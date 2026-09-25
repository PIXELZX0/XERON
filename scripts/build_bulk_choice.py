#!/usr/bin/env python3
"""Bulk builder: many HF decision datasets -> Laya-native JSONL for XERON-1.0 (encoder expansion).

Contract enforced on every emitted row (same as the existing builders):

    {"guid", "workflow", "state", "questions", "gold"}      # state/questions/gold = JSON strings

    questions = {"decision": {"type": choice|score|noul, "instructions": str, "criteria": {...}}}
    gold      = {"decision": {"probabilities": {...}}}      # sums to 1.0

Invariants (verified later by scripts/verify_bulk.py):
  * gold option sits at a random label position (no positional shortcut),
  * `state` never contains an answer marker ("Answer: B", answerKey, 정답, ...),
  * for label-name datasets with k > max_native, distractors are drawn from the SAME
    lexical-overlap bucket as the gold, so "option whose words appear in the text" is
    not a discriminating feature,
  * exact (state, questions) duplicates are dropped within the run and against
    --exclude keys (existing data/*.jsonl + the 231 public JevBench eval items).

Shapes
------
  nli            premise + hypothesis + int label          -> choice(3)
  label_int      text field(s) + int label (+ names)       -> choice(k)
  choices_list   question/context + list options + answer  -> choice(k)
  dict_choices   question + dict options + answerKey       -> choice(k)
  abc_columns    question + A..D columns + int answer      -> choice(k)
  ordinal        text + int label in [lo, hi]              -> score(lo..hi)
  soft_binary    text + float score in [0,1]               -> noul(soft gold)
  pair_binary    text1 + text2 (+ int label)               -> choice(2)

Usage
-----
    PYTHONPATH=scripts python scripts/build_bulk_choice.py --outdir data/bulk \
        --only xnli_en,massive,snli --exclude-existing
"""
import argparse
import hashlib
import heapq
import json
import os
import random
import re
import sys

from datasets import load_dataset, load_dataset_builder

LABELS = "ABCDEFGHIJKLMNOP"
MIN_OPTS, MAX_OPTS = 2, 6                     # native (full-k) choice budget
MAX_K = len(LABELS)                           # hard cap on shown options

# custom per-spec adapters contributed by a --specs-file (export ADAPTERS dict there)
ADAPTERS = {}

LEAK_PATTERNS = [
    re.compile(r"(?i)\banswer\s*[:=]\s*[A-P0-9]\b"),
    re.compile(r"(?i)\banswerkey\b"),
    re.compile(r"(?i)\banswer\s+key\b"),
    re.compile(r"(?i)\bcorrect\s+(answer|option|choice|label)\s*(is|:|=)"),
    re.compile(r"(?i)\bthe\s+answer\s+is\s*[A-P0-9]\b"),
    re.compile(r"(?i)\b(gold|correct|true)\s+(label|index|option|choice)\b"),
    re.compile(r"정답\s*[:：]?"),
    re.compile(r"(?i)\bground\s*truth\b"),
    re.compile(r"(?i)\"(answer|answerkey|label|target|gold)\"\s*:"),
]
TOK_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------------------
# SPECS
# --------------------------------------------------------------------------------------
# target   : rows per (config, split) — the row budget for that slice
# small    : (lo, hi) distractor count when the label set overflows the native head
# bucket   : same-lexical-bucket distractor control (label-name datasets only)
# stream   : reservoir-sample from a stream (no full download) for huge sources
SPECS = [
    # ---------------- multilingual routing / intent ----------------
    dict(name="massive", workflow="massive-intent", path="mteb/amazon_massive_intent",
         configs="ALL", splits=["train"], shape="label_str", text=["text"],
         label="label", lang_field="lang",
         instructions="Which intent does the user's utterance express?",
         target=11000, small=(3, 5), bucket=True, stream=False),
    # ---------------- multilingual NLI ----------------
    dict(name="xnli", workflow="xnli-multilingual", path="facebook/xnli",
         configs=["en", "fr", "es", "de", "el", "bg", "ru", "tr", "ar", "vi", "th", "zh",
                  "hi", "sw", "ur"],
         splits=["train"], shape="nli", lang_from_config=True,
         label_names=["entailment", "neutral", "contradiction"],
         instructions="What is the logical relation between `premise` and `hypothesis`?",
         target=25000, stream=False),
    # ---------------- english NLI / fact verification ----------------
    dict(name="snli", workflow="snli-nli", path="stanfordnlp/snli", splits=["train"],
         shape="nli", label_names=["entailment", "neutral", "contradiction"],
         instructions="What is the logical relation between `premise` and `hypothesis`?",
         target=30000, stream=False),
    dict(name="mnli", workflow="mnli-nli", path="nyu-mll/multi_nli", splits=["train"],
         shape="nli", label_names=["entailment", "neutral", "contradiction"],
         instructions="What is the logical relation between `premise` and `hypothesis`?",
         target=30000, stream=False),
    dict(name="anli", workflow="anli-nli", path="facebook/anli",
         splits=["train_r1", "train_r2", "train_r3"], shape="nli",
         label_names=["entailment", "neutral", "contradiction"],
         instructions="What is the logical relation between `premise` and `hypothesis`?",
         target=50000, stream=False),
    dict(name="vitaminc", workflow="vitaminc-factcheck", path="tals/vitaminc", splits=["train"],
         shape="label_int", text=["claim", "evidence"], json_text=["evidence"],
         label_names=["supports", "refutes", "not enough info"],
         instructions="Does `evidence` support, refute, or fail to establish `claim`?",
         target=40000, stream=False),
    # ---------------- english reading / science / math reasoning ----------------
    dict(name="race", workflow="race-reading", path="ehovy/race", configs=["all"],
         splits=["train"], shape="choices_list", text=["article"], question="question",
         choices="options", answer="answer", target=80000, stream=False),
    dict(name="qasc", workflow="qasc-science", path="allenai/qasc", splits=["train"],
         shape="dict_choices", text=["fact1", "fact2"], question="question",
         choices="choices", answer="answerKey", target=8000, stream=False),
    dict(name="aqua_rat", workflow="aqua-rat", path="deepmind/aqua_rat", splits=["train"],
         shape="choices_list", text=[], question="question", choices="options", answer="correct",
         target=90000, stream=False),
    dict(name="winogrande", workflow="winogrande-coref", path="allenai/winogrande",
         configs=["winogrande_xl", "winogrande_l", "winogrande_m", "winogrande_s",
                  "winogrande_debiased"],
         splits=["train"], shape="choices_list", text=["sentence"], question="sentence",
         choices=("option1", "option2"), answer="answer", target=8000, stream=False),
    dict(name="mmlu_pro", workflow="mmlu-pro-choice", path="TIGER-Lab/MMLU-Pro",
         splits=["test"], shape="choices_list", text=["category"], question="question",
         choices="options", answer="answer_index", target=12000, stream=False),
    dict(name="super_glue", workflow="superglue-{config}",
         path="aps/super_glue",
         configs=["boolq", "copa", "rte", "wic", "cb", "multirc", "record", "wsc.fixed", "axb", "axg"],
         splits=["train"], shape="auto", target=9000, stream=False),
    # ---------------- sentiment / topic / toxicity (volume, multi-difficulty) ----------------
    dict(name="amazon_polarity", workflow="amazon-polarity", path="fancyzhx/amazon_polarity",
         splits=["train"], shape="label_int", text=["title", "content"],
         label_names=["negative", "positive"],
         instructions="Is the review positive or negative?",
         target=60000, stream="shuffle"),
    dict(name="yelp_polarity", workflow="yelp-polarity", path="fancyzhx/yelp_polarity",
         splits=["train"], shape="label_int", text=["text"],
         label_names=["negative", "positive"],
         instructions="Is the review positive or negative?",
         target=40000, stream="shuffle"),
    dict(name="dbpedia_14", workflow="dbpedia-topic", path="fancyzhx/dbpedia_14",
         splits=["train"], shape="label_int", text=["title", "content"],
         label_names=["company", "educational institution", "artist", "athlete",
                      "office holder", "mean of transportation", "building",
                      "natural place", "village", "animal", "plant", "album", "film",
                      "written work"],
         instructions="Which topic does the document belong to?",
         target=60000, small=(3, 5), bucket=True, stream="shuffle"),
    dict(name="ag_news", workflow="agnews-topic", path="fancyzhx/ag_news", splits=["train"],
         shape="label_int", text=["text"], label_names=["world", "sports", "business", "sci/tech"],
         instructions="Which news category does the article belong to?",
         target=30000, stream="shuffle"),
    dict(name="civil_comments", workflow="civil-comments-toxicity",
         path="google/civil_comments", splits=["train"], shape="soft_binary",
         text=["text"], score="toxicity",
         instructions="Is this comment toxic? Answer with P(toxic).",
         target=80000, stream="shuffle"),
    dict(name="measuring_hate", workflow="measuring-hate-speech",
         path="ucberkeley-dlab/measuring-hate-speech", splits=["train"], shape="soft_binary",
         text=["text"], score="hate_speech_score", scale=(-2.0, 2.0),
         instructions="Does this comment express hate speech? Answer with P(hate speech).",
         target=60000, stream="shuffle"),
    dict(name="tweet_eval", workflow="tweet-{config}", path="cardiffnlp/tweet_eval",
         configs=["emoji", "emotion", "offensive", "hate", "irony", "sentiment", "stance_abortion",
                  "stance_atheism", "stance_climate", "stance_feminist", "stance_hillary"],
         splits=["train"], shape="auto", target=8000, stream=False),
    dict(name="emotion", workflow="emotion-6", path="dair-ai/emotion", splits=["train"],
         shape="label_int", text=["text"],
         label_names=["sadness", "joy", "love", "anger", "fear", "surprise"],
         instructions="Which emotion does the text express?", target=16000, stream=False),
    dict(name="sst5", workflow="sst5-ordinal", path="SetFit/sst5", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4,
         levels=["very negative", "negative", "neutral", "positive", "very positive"],
         instructions="How positive is the sentiment of the text?", target=8544, stream=False),
    # ---------------- korean ----------------
    dict(name="korean_extra", workflow="korean-{config}", path="klue/klue",
         configs=["nli", "sts", "ynat"], splits=["train"], shape="auto", target=30000,
         stream=False),
    dict(name="kobest", workflow="kobest-{config}", path="skt/kobest_v1",
         configs=["boolq", "copa", "hellaswag", "sentineg", "wic"],
         splits=["train"], shape="auto", target=9000, stream=False),
]


def spec_by_name():
    return {s["name"]: s for s in SPECS}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def norm_state(state_json):
    try:
        s = json.loads(state_json)
    except Exception:                                     # noqa: BLE001
        s = state_json
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def key_of(state_json):
    return hashlib.sha256(norm_state(state_json).encode()).hexdigest()[:16]


def clean(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scan_leak(text):
    return [p.pattern for p in LEAK_PATTERNS if p.search(text)]


def toks(s):
    return {t for t in TOK_RE.findall(str(s).lower()) if len(t) > 2}


def label_names_of(ds, field="label"):
    try:
        feat = ds.features[field]
        names = getattr(feat, "names", None)
        if names:
            return [str(n) for n in names]
    except Exception:                                     # noqa: BLE001
        pass
    return None


def collect_label_universe(ds, field="label", cap=500):
    """Distinct string labels of a (small) split, sorted — deterministic distractor pool."""
    seen = set()
    try:
        for row in ds:
            seen.add(str(row[field]))
            if len(seen) >= cap:
                break
    except Exception:                                     # noqa: BLE001
        pass
    return sorted(seen)


def iter_split(path, config, split, stream):
    """Yield rows of one (config, split). streaming=reservoir handled by the caller."""
    if config:
        ds = load_dataset(path, config, split=split, streaming=stream)
    else:
        ds = load_dataset(path, split=split, streaming=stream)
    return ds


def split_langs(path, config=None):
    try:
        b = load_dataset_builder(path, config) if config else load_dataset_builder(path)
        return list(b.info.splits.keys()) if b.info.splits else []
    except Exception:                                     # noqa: BLE001
        return []


def config_list(spec, limit=None):
    cfgs = spec.get("configs")
    if cfgs == "ALL" or cfgs == ["ALL"]:
        from datasets import get_dataset_config_names
        try:
            cfgs = sorted(get_dataset_config_names(spec["path"]))
        except Exception as e:                             # noqa: BLE001
            print(f"  ! config enumeration failed for {spec['path']}: {e}")
            cfgs = []
    if cfgs:
        return cfgs[:limit] if limit else cfgs
    return [None]


# --------------------------------------------------------------------------------------
# shape adapters -> (state dict, question fields, gold)
# --------------------------------------------------------------------------------------
def extract(spec, row, cfg):
    """Return (state:dict, kind:str, payload) or None.

    kind == "choice": payload = (instructions, options:list[str], correct_idx:int)
    kind == "score":  payload = (instructions, levels:list[str], target_idx:int)
    kind == "noul":   payload = (instructions, criteria:dict, p_true:float)
    """
    ad = spec.get("adapter")
    if ad:
        fn = ADAPTERS.get(ad)
        if fn is None:
            raise KeyError(f"adapter '{ad}' not provided by --specs-file (export ADAPTERS)")
        return fn(spec, row, cfg)
    shape = spec["shape"]
    lang = cfg if spec.get("lang_from_config") else spec.get("lang") or "en"

    if shape == "nli":
        state = {"language": lang, "premise": clean(row["premise"]),
                 "hypothesis": clean(row["hypothesis"])}
        names = spec.get("label_names") or ["entailment", "neutral", "contradiction"]
        lab = int(row["label"])
        if lab >= len(names) or lab < 0:
            return None
        return state, "choice", (spec["instructions"], names, lab)

    if shape == "label_int":
        state = {"language": lang}
        jt = spec.get("json_text") or []
        for i, f in enumerate(spec["text"]):
            v = row[f]
            if f in jt and isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        v = " ".join(str(x) for x in parsed)
                    elif isinstance(parsed, dict):
                        v = " ".join(str(x) for x in parsed.values())
                except Exception:                         # noqa: BLE001
                    pass
            state["text" if i == 0 else f"text_{i+1}"] = clean(v)
        names = spec.get("label_names") or spec.get("_names") or []
        raw = row["label"]
        if isinstance(raw, str) and not str(raw).lstrip("-").isdigit():
            lab = next((i for i, n in enumerate(names)
                        if str(n).strip().lower() == str(raw).strip().lower()), -1)
        else:
            try:
                lab = int(raw)
            except (TypeError, ValueError):
                lab = -1
        if not names:
            return None
        if lab >= len(names) or lab < 0:
            return None
        return state, "choice", (spec["instructions"], names, lab)

    if shape == "label_str":
        lf = spec.get("lang_field")
        state = {"language": (row.get(lf) if lf else lang) or lang,
                 "text": clean(row[spec["text"][0]])}
        names = spec.get("label_names") or spec.get("_names") or []
        lab_s = str(row[spec["label"]])
        if not names or lab_s not in names:
            return None
        return state, "choice", (spec["instructions"], names, names.index(lab_s))

    if shape == "choices_list":
        state = {"language": lang}
        for i, f in enumerate(spec.get("text") or []):
            v = row.get(f)
            if isinstance(v, str):
                state[f] = clean(v)
        opts_field = spec["choices"]
        if isinstance(opts_field, tuple):
            options = [clean(row[o]) for o in opts_field]
        else:
            options = [clean(o) for o in row[opts_field]]
        if spec.get("question"):
            qv = row.get(spec["question"])
            if qv is not None and not isinstance(qv, (list, dict)):
                state["question"] = clean(qv)
        ans = row[spec["answer"]]
        idx = -1
        if isinstance(ans, int):
            idx = int(ans)
        else:
            a = str(ans).strip()
            if a.upper() in LABELS and len(a) == 1:
                idx = "ABCDEFGHIJKLMNOP".find(a.upper())
            elif a.isdigit():                        # 1-based index (winogrande / belebele)
                idx = int(a) - 1
            elif a in options:
                idx = options.index(a)
        if idx < 0 or idx >= len(options):
            return None
        return state, "choice", (spec.get("instructions") or
                                 "Which option is the correct answer?", options, idx)

    if shape == "dict_choices":
        state = {"language": lang}
        for f in spec.get("text") or []:
            v = row.get(f)
            if isinstance(v, str):
                state[f] = clean(v)
        if spec.get("question"):
            state["question"] = clean(row[spec["question"]])
        ch = row[spec["choices"]]
        if isinstance(ch, dict):
            if "text" in ch:                             # qasc-style: {"text": [...], "label": [...]}
                options = [clean(t) for t in ch["text"]]
                ans = str(row[spec["answer"]]).strip()
                idx = ("ABCDEFGHIJKLMNOP".find(ans.upper())
                       if len(ans) == 1 and ans.isalpha() else -1)
            else:
                keys = sorted([k for k in ch if k != "text"])
                options = [clean(ch[k]) for k in keys]
                ans = str(row[spec["answer"]]).strip()
                idx = keys.index(ans) if ans in keys else -1
        else:
            return None
        if idx < 0:
            return None
        return state, "choice", (spec.get("instructions") or
                                 "Which option is the correct answer?", options, idx)

    if shape == "copa":
        state = {"language": lang, "premise": clean(row.get(spec["premise"]))}
        if spec.get("question"):
            state["question"] = clean(row.get(spec["question"]))
        options = [clean(row[spec["choices"][0]]), clean(row[spec["choices"][1]])]
        lab = int(row[spec["answer"]])
        if lab not in (0, 1):
            return None
        return state, "choice", (spec.get("instructions") or "Which alternative fits?",
                                 options, lab)

    if shape == "story_cloze":
        ctx = " ".join(clean(row[f"input_sentence_{i}"]) for i in range(1, 5))
        options = [clean(row.get("sentence_quiz1")), clean(row.get("sentence_quiz2"))]
        lab = int(row[spec["answer"]]) - 1
        if lab not in (0, 1) or "" in options:
            return None
        state = {"language": lang, "story": ctx}
        return state, "choice", (spec.get("instructions") or
                                 "Which sentence is the correct ending of the story?",
                                 options, lab)

    if shape == "pair_binary":
        state = {"language": lang, spec["text"][0]: clean(row[spec["text"][0]]),
                 spec["text"][1]: clean(row[spec["text"][1]])}
        names = spec.get("label_names") or ["no", "yes"]
        lab = int(row[spec["label"]])
        if lab >= len(names):
            return None
        return state, "choice", (spec["instructions"], names, lab)

    if shape == "abc_columns":
        state = {"language": lang, "question": clean(row["question"])}
        cols = spec.get("answer_columns") or ["A", "B", "C", "D"]
        options = [clean(row[c]) for c in cols]
        idx = int(row["answer"])
        if idx >= len(options):
            return None
        return state, "choice", (spec["instructions"], options, idx)

    if shape == "ordinal":
        state = {"language": lang, "text": clean(row[spec["text"][0]])}
        lab = int(row["label"])
        levels = spec["levels"]
        if lab < 0 or lab >= len(levels):
            return None
        return state, "score", (spec["instructions"], levels, lab)

    if shape == "soft_binary":
        state = {"language": lang, "text": clean(row[spec["text"][0]])}
        v = float(row[spec["score"]])
        if spec.get("scale"):
            lo, hi = spec["scale"]
            v = min(1.0, max(0.0, (v - lo) / (hi - lo)))
        v = min(1.0, max(0.0, v))
        crit = {"false": "No — the property is absent.",
                "true": "Yes — the property is present."}
        return state, "noul", (spec["instructions"], crit, v)

    if shape == "auto":
        return extract_auto(spec, row, cfg)

    return None


def extract_auto(spec, row, cfg):
    """Best-effort adapter for heterogeneous suites (super_glue, tweet_eval, klue, kobest)."""
    lang = {"klue": "ko", "kobest": "ko"}.get(spec["name"], spec.get("lang") or "en")
    name = f"{spec['name']}/{cfg}"
    row = {k: v for k, v in row.items()}

    def g(k, default=None):
        return row.get(k, default)

    # per-config label-name overrides (for suites whose names are not in the features)
    nbc = (spec.get("names_by_config") or {}).get(cfg)
    if nbc:
        spec = dict(spec)
        spec["_names"] = nbc

    # ---- klue ----
    if spec["name"] == "korean_extra":
        if cfg == "ynat":
            names = spec.get("_names") or []
            lab = int(row["label"])
            if not names or lab >= len(names):
                return None
            state = {"language": "ko", "title": clean(row["title"])}
            return state, "choice", ("기사의 주제 분류는 무엇인가?", names, lab)
        if cfg == "nli":
            state = {"language": "ko", "premise": clean(row.get("premise")),
                     "hypothesis": clean(row.get("hypothesis"))}
            lab = {"entailment": 0, "neutral": 1, "contradiction": 2}.get(str(row["label"]).lower())
            if lab is None:
                return None
            return state, "choice", ("두 문장의 논리적 관계는 무엇인가?",
                                     ["함의(entailment)", "중립(neutral)", "모순(contradiction)"], lab)
        if cfg == "sts":
            v = float(row["labels"]["label"]) if isinstance(row.get("labels"), dict) else float(row["label"])
            v = min(5.0, max(0.0, v)) / 5.0
            crit = {"false": "두 문장은 의미가 다르다.", "true": "두 문장은 의미가 같다."}
            state = {"language": "ko", "sentence1": clean(row["sentence1"]),
                     "sentence2": clean(row["sentence2"])}
            return state, "noul", ("두 문장의 의미가 같은가? P(같다)로 답하라.", crit, v)

    # ---- kobest ----
    if spec["name"] == "kobest":
        if cfg == "boolq":
            lab = int(row["label"])
            state = {"language": "ko", "paragraph": clean(row["paragraph"]),
                     "question": clean(row["question"])}
            return state, "choice", ("지문에 근거할 때 질문의 답은?", ["아니오", "예"], lab)
        if cfg == "copa":
            lab = int(row["label"])
            state = {"language": "ko",
                     "premise": clean(row.get("premise") or row.get("sentence")),
                     "question": clean(row.get("question") or "")}
            opts = [clean(row.get("alternative_1") or row.get("option1") or ""),
                    clean(row.get("alternative_2") or row.get("option2") or "")]
            if "" in opts:
                return None
            return state, "choice", ("인과관계상 알맞은 대안은?", opts, lab)
        if cfg == "sentineg":
            lab = int(row["label"]) if "label" in row else int(row.get("labels", 0))
            state = {"language": "ko", "text": clean(row.get("sentence") or row.get("text"))}
            return state, "choice", ("문장의 감정은?", ["부정", "긍정"], lab)
        if cfg == "hellaswag":
            lab = int(row["label"]) if "label" in row else -1
            ctx = clean(row.get("context") or row.get("sentence") or "")
            endings = row.get("endings") or []
            if lab < 0 or lab >= len(endings):
                return None
            state = {"language": "ko", "context": ctx}
            return state, "choice", ("이어질 가장 자연스러운 문장은?",
                                     [clean(e) for e in endings], lab)
        if cfg == "wic":
            lab = int(row["label"])
            state = {"language": "ko", "sentence1": clean(row.get("sentence1")),
                     "sentence2": clean(row.get("sentence2")),
                     "word": clean(row.get("word"))}
            return state, "choice", ("두 문장에서 단어의 의미가 같은가?", ["다르다", "같다"], lab)

    # ---- super_glue ----
    if spec["name"] == "super_glue":
        if cfg == "boolq":
            state = {"passage": clean(g("passage")), "question": clean(g("question"))}
            return state, "choice", ("지문에 근거해 질문에 답하라.", ["no", "yes"], int(g("label")))
        if cfg == "copa":
            state = {"premise": clean(g("premise")), "question": clean(g("question"))}
            return state, "choice", ("알맞은 선택지는?", [clean(g("choice1")), clean(g("choice2"))],
                                     int(g("label")))
        if cfg in ("rte", "cb", "axb", "axg"):
            lab = int(g("label"))
            names = {"rte": ["entailment", "not_entailment"],
                     "cb": ["entailment", "contradiction", "neutral"],
                     "axb": ["entailment", "not_entailment"],
                     "axg": ["entailment", "not_entailment"]}[cfg]
            if lab >= len(names):
                return None
            hy = g("hypothesis") if cfg != "axb" else g("sentence2")
            pr = g("premise") if cfg != "axb" else g("sentence1")
            state = {("sentence1" if cfg == "axb" else "premise"): clean(pr),
                     ("sentence2" if cfg == "axb" else "hypothesis"): clean(hy)}
            return state, "choice", ("가설과 전제의 관계는?", names, lab)
        if cfg == "wic":
            lab = int(g("label"))
            state = {"sentence1": clean(g("sentence1")), "sentence2": clean(g("sentence2")),
                     "word": clean(g("word"))}
            return state, "choice", ("두 문장에서 단어의 의미가 같은가?", ["different", "same"], lab)
        if cfg == "multirc":
            lab = int(g("label"))
            state = {"paragraph": clean(g("paragraph")), "question": clean(g("question")),
                     "answer": clean(g("answer"))}
            return state, "choice", ("제시된 답이 지문에 근거해 맞는가?", ["false", "true"], lab)
        if cfg == "wsc.fixed":
            lab = int(g("label"))
            state = {"text": clean(g("text")), "span1": clean(g("span1_text")),
                     "span2": clean(g("span2_text"))}
            return state, "choice", ("대명사가 가리키는 것은?", [clean(g("span1_text")),
                                                          clean(g("span2_text"))], lab)
        if cfg == "record":
            return None                                  # span-extraction, not a decision
        return None

    # ---- tweet_eval ----
    if spec["name"] == "tweet_eval":
        text = clean(g("text"))
        names = spec.get("_names") or []
        if cfg in ("emoji",):
            if not names:
                return None
            return {"text": text}, "choice", ("Which emoji best matches the tweet?", names,
                                              int(g("label")))
        if cfg == "emotion":
            return {"text": text}, "choice", ("Which emotion does the tweet express?", names,
                                              int(g("label")))
        if cfg in ("offensive", "hate", "irony"):
            if not names:
                return None
            return {"text": text}, "choice", (f"Is the tweet {cfg}?", names, int(g("label")))
        if cfg == "sentiment":
            return {"text": text}, "choice", ("What is the sentiment of the tweet?", names,
                                              int(g("label")))
        if cfg.startswith("stance_"):
            return {"text": text, "target": cfg.split("stance_")[1]}, "choice",
            ("What is the stance toward the target?", names, int(g("label")))
    return None


# --------------------------------------------------------------------------------------
# row emission
# --------------------------------------------------------------------------------------
def shortcut_pool(options, correct_idx, state_text, lo, hi, rng):
    """Sample distractors from the same lexical-overlap bucket as the gold option."""
    st = toks(state_text)
    bkt = [min(len(toks(o) & st), 2) for o in options]
    gb = bkt[correct_idx]
    same = [i for i in range(len(options)) if i != correct_idx and bkt[i] == gb]
    if len(same) < lo:
        return None
    nd = min(rng.randint(lo, hi), len(same))
    return rng.sample(same, nd)


def emit(spec, state, kind, payload, guid, rng, workflow):
    state = dict(state)
    state.setdefault("language", spec.get("lang") or "en")

    if kind == "choice":
        instructions, options, correct_idx = payload
        options = [clean(o) for o in options]
        if any(o == "" for o in options) or len(set(options)) != len(options):
            return None
        k = len(options)
        if spec.get("small") and k > MAX_OPTS or (spec.get("small") and spec.get("force_small")):
            pass
        if k <= MAX_OPTS and not spec.get("force_small"):
            idxs = list(range(k))
            rng.shuffle(idxs)
            crit = {LABELS[p]: options[i] for p, i in enumerate(idxs)}
            lab = LABELS[idxs.index(correct_idx)]
        else:
            lo, hi = spec.get("small") or (3, 5)
            st_text = " ".join(str(v) for v in state.values())
            if spec.get("bucket"):
                dist = shortcut_pool(options, correct_idx, st_text, lo, hi, rng)
                if dist is None:
                    return None
            else:
                pool = [i for i in range(k) if i != correct_idx]
                if len(pool) < lo:
                    return None
                dist = rng.sample(pool, min(rng.randint(lo, hi), len(pool)))
            shown = [correct_idx] + dist
            rng.shuffle(shown)
            crit = {LABELS[p]: options[i] for p, i in enumerate(shown)}
            lab = LABELS[shown.index(correct_idx)]
        # state keeps the labelled option map (same labels as criteria)
        state["choices"] = dict(crit)
        q = {"decision": {"type": "choice", "instructions": instructions, "criteria": crit}}
        g = {"decision": {"probabilities": {l: (1.0 if l == lab else 0.0) for l in crit}}}
    elif kind == "score":
        instructions, levels, correct_idx = payload
        crit = {str(i): clean(l) for i, l in enumerate(levels)}
        q = {"decision": {"type": "score", "instructions": instructions, "criteria": crit}}
        g = {"decision": {"probabilities": {k: (1.0 if k == str(correct_idx) else 0.0)
                                            for k in crit}}}
    elif kind == "noul":
        instructions, crit, p_true = payload
        crit = {k: clean(v) for k, v in crit.items()}
        q = {"decision": {"type": "noul", "instructions": instructions, "criteria": crit}}
        g = {"decision": {"probabilities": {"false": 1.0 - p_true, "true": p_true}}}
    else:
        return None

    state_json = json.dumps(state, ensure_ascii=False)
    if scan_leak(state_json):
        return None
    probs = g["decision"]["probabilities"]
    if not probs or abs(sum(probs.values()) - 1.0) > 1e-6:
        return None
    if kind == "choice" and (len(probs) < MIN_OPTS or len(probs) > MAX_K):
        return None
    return {"guid": guid, "workflow": workflow, "state": state_json,
            "questions": json.dumps(q, ensure_ascii=False),
            "gold": json.dumps(g, ensure_ascii=False)}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def reservoir(rows_iter, budget, rng, seen_counter):
    heap = []
    seen = 0
    for row in rows_iter:
        seen += 1
        k = rng.random()
        if len(heap) < budget:
            heapq.heappush(heap, (k, seen, row))
        elif k > heap[0][0]:
            heapq.heapreplace(heap, (k, seen, row))
    seen_counter[0] += seen
    return [r for _, _, r in heap]


def existing_keys(paths):
    keys = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r, dict) and "state" in r:
                        keys.add(key_of(r["state"]))
        except Exception as e:                             # noqa: BLE001
            print(f"  ! exclude {p}: {e}")
    return keys


def public_eval_keys(pubdir="/tmp/jevbench/datasets/public"):
    keys = set()
    if not os.path.isdir(pubdir):
        return keys
    for fn in os.listdir(pubdir):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(pubdir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    it = json.loads(line)
                except json.JSONDecodeError:
                    continue
                st = it.get("state")
                if isinstance(st, str):
                    keys.add(hashlib.sha256(re.sub(r"\s+", " ", st.lower()).strip()
                                            .encode()).hexdigest()[:16])
    return keys


def build_spec(spec, outdir, rng, ex_keys, ex_texts, per_config_limit=None, verbose=True):
    name = spec["name"]
    rows, stats = [], {"spec": name, "slices": [], "dropped_leak": 0, "dropped_dup": 0,
                       "dropped_pool": 0, "dropped_shape": 0}
    for cfg in config_list(spec, per_config_limit):
        for split in spec["splits"]:
            try:
                ds = iter_split(spec["path"], cfg, split, spec.get("stream", False))
            except Exception as e:                         # noqa: BLE001
                msg = f"{type(e).__name__}: {str(e)[:120]}"
                print(f"  [{name}/{cfg}/{split}] LOAD FAIL {msg}", flush=True)
                stats["slices"].append({"config": cfg, "split": split, "error": msg})
                continue
            names = label_names_of(ds) or []
            spec_local = dict(spec)
            spec_local["_names"] = names
            if spec.get("shape") == "label_str" and not spec_local["_names"]:
                spec_local["_names"] = collect_label_universe(ds, spec["label"])
            budget = spec.get("target", 10000)
            counter = [0]
            src = ds
            if spec.get("stream") == "shuffle":
                # fast path for multi-million-row sources: buffered shuffle + take
                src = ds.shuffle(seed=7, buffer_size=10000).take(budget)
                counter[0] = budget
            elif spec.get("stream", False):
                src = reservoir(ds, budget, rng, counter)
            else:
                n = len(ds)
                n = min(n, budget)
                src = ds.select(range(n))
                counter[0] = n
            kept = 0
            for i, row in enumerate(src):
                try:
                    res = extract(spec_local, row, cfg)
                except Exception:                          # noqa: BLE001
                    stats["dropped_shape"] += 1
                    continue
                if res is None:
                    stats["dropped_shape"] += 1
                    continue
                state, kind, payload = res
                wf = spec["workflow"].format(config=cfg) if "{config}" in spec["workflow"] \
                    else spec["workflow"]
                rec = emit(spec_local, state, kind, payload, f"{name}/{cfg or '-'}/{split}/{i}",
                           rng, wf)
                if rec is None:
                    stats["dropped_pool"] += 1
                    continue
                kk = key_of(rec["state"])
                if kk in ex_keys:
                    stats["dropped_dup"] += 1
                    continue
                ex_keys.add(kk)
                rec["_k"] = kk
                rows.append(rec)
                kept += 1
            stats["slices"].append({"config": cfg, "split": split, "rows": kept,
                                    "seen": counter[0]})
            if verbose:
                print(f"  [{name}/{cfg}/{split}] kept {kept:,} / seen {counter[0]:,}", flush=True)
    for r in rows:
        r.pop("_k", None)
    out = os.path.join(outdir, f"{name}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows"] = len(rows)
    stats["file"] = out
    print(f"[{name}] wrote {len(rows):,} rows -> {out}", flush=True)
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/bulk")
    ap.add_argument("--only", default="", help="comma list of spec names")
    ap.add_argument("--exclude-existing", action="store_true", default=True)
    ap.add_argument("--no-exclude-existing", dest="exclude_existing", action="store_false")
    ap.add_argument("--exclude-glob", default="data/*.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--config-limit", type=int, default=0)
    ap.add_argument("--rebuild-exclude-cache", action="store_true")
    ap.add_argument("--cache-only", action="store_true",
                    help="rebuild the exclusion-key cache and exit (no build)")
    ap.add_argument("--specs-file", default="",
                    help="python file exporting SPECS (list of spec dicts) to run instead")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    specs_all = SPECS
    if args.specs_file:
        import importlib.util
        sp = args.specs_file
        if not os.path.isabs(sp):
            here = os.path.dirname(os.path.abspath(__file__))
            # accept both "scripts/specs_x.py" and "specs_x.py" (relative to repo root or scripts/)
            cand = [sp, os.path.join(here, sp), os.path.join(here, os.path.basename(sp))]
            sp = next((c for c in cand if os.path.exists(c)), os.path.join(here, sp))
        module_spec = importlib.util.spec_from_file_location("extra_specs", sp)
        mod = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(mod)
        specs_all = list(mod.SPECS)
        ADAPTERS.update(getattr(mod, "ADAPTERS", {}) or {})
        print(f"loaded {len(specs_all)} specs (+{len(ADAPTERS)} adapters) from {sp}", flush=True)
    specs = [s for s in specs_all if not wanted or s["name"] in wanted]
    if not specs:
        print(f"no spec matched --only={args.only}")
        return 1

    import glob
    cache_path = os.path.join(args.outdir, "_exclude_keys.json")
    ex_paths = sorted(glob.glob(args.exclude_glob)) if args.exclude_existing else []
    ex_paths += sorted(glob.glob(os.path.join(args.outdir, "*.jsonl")))
    if os.path.exists(cache_path) and not args.rebuild_exclude_cache:
        with open(cache_path) as f:
            ex_keys = set(json.load(f))
        print(f"exclusion keys (cached): {len(ex_keys):,}", flush=True)
    else:
        print(f"exclusion sources: {len(ex_paths)} jsonl files", flush=True)
        ex_keys = existing_keys(ex_paths) if ex_paths else set()
        ex_keys |= public_eval_keys()
        print(f"exclusion keys: {len(ex_keys):,}", flush=True)
        with open(cache_path, "w") as f:
            json.dump(sorted(ex_keys), f)
        print(f"cached -> {cache_path}", flush=True)
    if args.cache_only:
        print("cache-only: done")
        return 0

    summaries = []
    for spec in specs:
        _, st = build_spec(spec, args.outdir, rng, ex_keys, set(),
                           args.config_limit or None)
        summaries.append(st)
        with open(os.path.join(args.outdir, f"{spec['name']}.manifest.json"), "w") as f:
            json.dump(st, f, ensure_ascii=False, indent=1)

    total = sum(s["rows"] for s in summaries)
    print(f"\n=== TOTAL {total:,} rows across {len(summaries)} specs ===")
    with open(os.path.join(args.outdir, "_build_summary.json"), "w") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

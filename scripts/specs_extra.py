#!/usr/bin/env python3
"""XERON-1.0 wave-3 registry (extra): Japanese / Chinese / Korean + reasoning·math + GLUE.

Loaded by:  build_bulk_choice.py --specs-file scripts/specs_extra.py

All "Dataset scripts are no longer supported" (datasets>=4 dropped .py loading)
sources from data/candidate_probe.json are recovered here through **script-free
parquet mirrors / community re-uploads**, each verified by probing on 2026-09-25
(field names, config list, row counts, label balance of the first rows):

  allenai/social_i_qa      -> tasksource/social_i_qa        (33,410 rows, answerA-C)
  allenai/cosmos_qa        -> Samsoup/cosmos_qa             (25,262 rows, answer0-3)
  allenai/math_qa          -> tasksource/math_qa            (29,837 rows, options str)
  lucasmccabe/logiqa       -> jeggers/logiqa2_formatted     (LogiQA2, 12,567 rows)
  fever/fever              -> pietrolesci/nli_fever         (208,346 NLI rows)
  e9t/nsmc                 -> Blpeng/nsmc                   (150,000 rows)
  shunk031/JGLUE (JNLI)    -> zenless-lab/jnli              (20,073 rows)
  shunk031/JGLUE (JSTS)    -> zenless-lab/jsts              (12,451 rows, float 0-5)
  shunk031/JGLUE (MARC-ja) -> yangwang825/marc-ja           (187,528 rows)
  clue (tnews/iflytek/…)   -> clue/clue                     (parquet-native repo)
  shibing624/C3            -> dataset-org/c3                (mixed, 3,138 rows)
  google/go_emotions       -> google-research-datasets/go_emotions
  takala/financial_phrasebank -> atrost/… + warwickai/… mirror
  fancyzhx/yahoo_answers_topics -> community-datasets/yahoo_answers_topics (1.4M)
  mteb/mtop_*              -> tasksource/mtop               (73,928, 6 languages)

Also added: HAERAE-HUB/KMMLU (45 subjects), kakaobrain/kor_nli, nyu-mll/glue set.
Deliberately NOT built (already in the main registry / another wave):
  ARC, MMLU-Pro, QASC, Aqua-RAT, Winogrande, SuperGLUE, KLUE, KoBEST, MASSIVE, XNLI.
"""

import re
import zlib

LABELS = "ABCDEFGHIJKLMNOP"
_WS = re.compile(r"\s+")

# ---------------------------------------------------------------------------- helpers
def clean(t):
    if t is None:
        return ""
    return _WS.sub(" ", str(t)).strip()


def _lab(row, key="label"):
    try:
        return int(row[key])
    except Exception:                                     # noqa: BLE001
        return -1


def _ok(lab, names):
    return isinstance(lab, int) and 0 <= lab < len(names)


def _ans1(v, k):
    """1-based answer column (KMMLU) -> 0-based index."""
    try:
        i = int(v)
    except Exception:                                     # noqa: BLE001
        return -1
    if 1 <= i <= k:
        return i - 1
    if 0 <= i < k:                                        # already 0-based
        return i
    return -1


# ---------------------------------------------------------------------------- NLI (localized options)
_EN2LOC = {
    "entailment":    {"ko": "함의", "ja": "含意", "zh": "蕴含"},
    "neutral":       {"ko": "중립", "ja": "中立", "zh": "中立"},
    "contradiction": {"ko": "모순", "ja": "矛盾", "zh": "矛盾"},
}


def nli_loc(spec, row, cfg):
    """NLI with localized option names.

    Option order = spec['label_names'] (explicit, probed) > dataset feature names
    (unavailable when streaming) > standard order.  Handles both the
    premise/hypothesis layout and the sentence1/sentence2 layout (CLUE).
    """
    lang = spec.get("lang", "en")
    names = (spec.get("label_names") or spec.get("_names")
             or ["entailment", "neutral", "contradiction"])
    lab = _lab(row)
    if not _ok(lab, names):
        return None
    disp = [f"{_EN2LOC.get(n, {}).get(lang, n)}({n})" for n in names]
    if row.get("premise") is not None and row.get("hypothesis") is not None:
        keys = ("premise", "hypothesis")
    else:
        keys = ("sentence1", "sentence2")
    state = {"language": lang, keys[0]: clean(row.get(keys[0])),
             keys[1]: clean(row.get(keys[1]))}
    if not state[keys[0]] or not state[keys[1]]:
        return None
    return state, "choice", (spec["instructions"], disp, lab)


# ---------------------------------------------------------------------------- Korean
def kmmlu(spec, row, cfg):
    cols = ["A", "B", "C", "D"]
    opts = [clean(row.get(c)) for c in cols]
    idx = _ans1(row.get("answer"), len(opts))
    if idx < 0 or any(o == "" for o in opts):
        return None
    state = {"language": "ko", "question": clean(row.get("question")),
             "subject": clean(row.get("Category") or cfg or "")}
    return state, "choice", ("다음 문제의 알맞은 답을 고르시오.", opts, idx)


# ---------------------------------------------------------------------------- Japanese
def marc_ja(spec, row, cfg):
    names = spec.get("_names") or ["positive", "negative"]
    lab = _lab(row)
    if not _ok(lab, names):
        return None
    loc = {"positive": "ポジティブ", "negative": "ネガティブ", "neutral": "中立"}
    disp = [loc.get(n, n) for n in names]
    state = {"language": "ja", "text": clean(row.get("text"))}
    if not state["text"]:
        return None
    return state, "choice", ("このレビューの感情はどちらですか?", disp, lab)


def jsts(spec, row, cfg):
    try:
        v = float(row["label"]) / 5.0
    except Exception:                                     # noqa: BLE001
        return None
    v = min(1.0, max(0.0, v))
    crit = {"false": "二つの文は意味が異なる。", "true": "二つの文は意味が同じである。"}
    state = {"language": "ja", "sentence1": clean(row.get("sentence1")),
             "sentence2": clean(row.get("sentence2"))}
    if not state["sentence1"] or not state["sentence2"]:
        return None
    return state, "noul", ("二つの文の意味は同じですか? P(同じ)で答えてください。", crit, v)


# ---------------------------------------------------------------------------- Chinese
# TNEWS category codes (CLUE) -> readable names
_TNEWS = {
    "100": "新闻故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经",
    "106": "房产", "107": "汽车", "108": "教育", "109": "科技", "110": "军事",
    "112": "旅游", "113": "国际", "114": "股票", "115": "农业", "116": "游戏",
}


def tnews(spec, row, cfg):
    codes = spec.get("_names") or []
    lab = _lab(row)
    if not _ok(lab, codes):
        return None
    disp = [_TNEWS.get(str(c), str(c)) for c in codes]
    if len(set(disp)) != len(disp):
        return None
    state = {"language": "zh", "text": clean(row.get("sentence"))}
    if not state["text"]:
        return None
    return state, "choice", ("这条新闻属于哪个类别?", disp, lab)


def afqmc(spec, row, cfg):
    lab = _lab(row)
    if lab not in (0, 1):
        return None
    state = {"language": "zh", "sentence1": clean(row.get("sentence1")),
             "sentence2": clean(row.get("sentence2"))}
    if not state["sentence1"] or not state["sentence2"]:
        return None
    return state, "choice", ("这两个句子表达的意思相同吗?", ["意思不同", "意思相同"], lab)


def clue_wsc(spec, row, cfg):
    lab = _lab(row)
    if lab not in (0, 1):
        return None
    tgt = row.get("target") or {}
    s1 = clean(tgt.get("span1_text"))
    s2 = clean(tgt.get("span2_text"))
    if not s1 or not s2:
        return None
    state = {"language": "zh", "text": clean(row.get("text")), "span1": s1, "span2": s2}
    if not state["text"]:
        return None
    return state, "choice", (f"文中的「{s2}」指的是「{s1}」吗?", ["否", "是"], lab)


def c3(spec, row, cfg):
    docs = row.get("documents") or []
    q = row.get("questions") or {}
    qs = list(q.get("question") or [])
    ans = list(q.get("answer") or [])
    chs = list(q.get("choice") or [])
    if not qs:
        return None
    # deterministic pick: the question with the widest option set (most informative)
    cand = [i for i in range(len(qs)) if i < len(ans) and i < len(chs)]
    if not cand:
        return None
    best = max(cand, key=lambda i: (len(chs[i]), -i))
    opts = [clean(c) for c in chs[best]]
    if not (2 <= len(opts) <= 4) or any(o == "" for o in opts):
        return None
    a = clean(ans[best])
    if a not in opts:
        return None
    state = {"language": "zh",
             "documents": clean(" ".join(str(d) for d in docs))[:3000],
             "question": clean(qs[best])}
    return state, "choice", ("根据以上材料回答问题。", opts, opts.index(a))


# ---------------------------------------------------------------------------- English reasoning / math
def math_qa(spec, row, cfg):
    opt_str = clean(row.get("options"))
    if not opt_str:
        return None
    parts = re.split(r"\s*,\s*(?=[a-eA-E]\s*\))", opt_str)
    keys, opts = [], []
    for p in parts:
        m = re.match(r"^\s*([a-eA-E])\s*\)\s*(.*)$", p.strip())
        if not m:
            return None
        keys.append(m.group(1).lower())
        opts.append(m.group(2).strip())
    corr = str(row.get("correct", "")).strip().lower()
    if corr not in keys or not (2 <= len(opts) <= 6) or any(o == "" for o in opts):
        return None
    state = {"language": "en", "question": clean(row.get("Problem"))}
    if not state["question"]:
        return None
    return state, "choice", ("Solve the math problem and pick the correct option.",
                             opts, keys.index(corr))


def social_i_qa(spec, row, cfg):
    lab = _lab(row)
    if lab not in (0, 1, 2):
        return None
    opts = [clean(row.get("answerA")), clean(row.get("answerB")), clean(row.get("answerC"))]
    if any(o == "" for o in opts):
        return None
    state = {"language": "en", "context": clean(row.get("context")),
             "question": clean(row.get("question"))}
    return state, "choice", ("Which answer is the most plausible for the question?",
                             opts, lab)


# 28-class multi-label -> soft binary "does this text express <emotion>?"
_GO_EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"]


def go_emotions(spec, row, cfg):
    text = clean(row.get("text"))
    labs = sorted({int(x) for x in (row.get("labels") or [])})
    if not text or not labs:
        return None
    h = zlib.crc32(text.encode("utf-8"))
    if h % 2 == 0:                                        # positive query
        idx = labs[(h // 2) % len(labs)]
        p = 0.85
    else:                                                 # negative query
        negs = [i for i in range(len(_GO_EMOTIONS)) if i not in set(labs)]
        if not negs:
            return None
        idx = negs[(h // 2) % len(negs)]
        p = 0.15
    emo = _GO_EMOTIONS[idx]
    crit = {"false": f"The comment does not express {emo}.",
            "true": f"The comment expresses {emo}."}
    state = {"language": "en", "text": text, "emotion": emo}
    return state, "noul", (f"Does the comment express the emotion '{emo}'? "
                           "Answer with P(true).", crit, p)


def stsb(spec, row, cfg):
    try:
        v = float(row["label"]) / 5.0
    except Exception:                                     # noqa: BLE001
        return None
    v = min(1.0, max(0.0, v))
    crit = {"false": "The two sentences have different meanings.",
            "true": "The two sentences have the same meaning."}
    state = {"language": "en", "sentence1": clean(row.get("sentence1")),
             "sentence2": clean(row.get("sentence2"))}
    if not state["sentence1"] or not state["sentence2"]:
        return None
    return state, "noul", ("Do the two sentences have the same meaning? "
                           "Answer with P(same).", crit, v)


# ---------------------------------------------------------------------------- multilingual MTOP
# top-30 most frequent MTOP intents (85% of train rows); rare intents are skipped
_MTOP_INTENTS = [
    "IN:GET_WEATHER", "IN:CREATE_CALL", "IN:CREATE_REMINDER", "IN:CREATE_ALARM",
    "IN:GET_STORIES_NEWS", "IN:SEND_MESSAGE", "IN:PLAY_MUSIC", "IN:GET_EVENT",
    "IN:GET_CONTACT", "IN:GET_INFO_RECIPES", "IN:GET_RECIPES", "IN:CREATE_TIMER",
    "IN:DELETE_REMINDER", "IN:GET_REMINDER", "IN:UPDATE_CALL", "IN:GET_AVAILABILITY",
    "IN:GET_TIMER", "IN:GET_MESSAGE", "IN:GET_LOCATION", "IN:END_CALL",
    "IN:QUESTION_NEWS", "IN:GET_ALARM", "IN:ADD_TIME_TIMER", "IN:GET_EMPLOYER",
    "IN:UPDATE_REMINDER_DATE_TIME", "IN:GET_INFO_CONTACT", "IN:SET_UNAVAILABLE",
    "IN:PAUSE_TIMER", "IN:GET_TRACK_INFO_MUSIC", "IN:SNOOZE_ALARM",
]
_MTOP_DOMAINS = ["alarm", "calling", "event", "messaging", "music", "news",
                 "people", "recipes", "reminder", "timer", "weather"]


def _mtop_lang(row):
    v = str(row.get("lang") or "en_XX")
    return v.split("_")[0] or "en"


def mtop_intent(spec, row, cfg):
    it = clean(row.get("intent"))
    if it not in _MTOP_INTENTS:
        return None
    state = {"language": _mtop_lang(row), "text": clean(row.get("question")), "task": "intent"}
    if not state["text"]:
        return None
    return state, "choice", ("Which intent does the user's utterance express?",
                             list(_MTOP_INTENTS), _MTOP_INTENTS.index(it))


def mtop_domain(spec, row, cfg):
    d = clean(row.get("domain"))
    if d not in _MTOP_DOMAINS:
        return None
    state = {"language": _mtop_lang(row), "text": clean(row.get("question")), "task": "domain"}
    if not state["text"]:
        return None
    return state, "choice", ("Which domain does the user's utterance belong to?",
                             list(_MTOP_DOMAINS), _MTOP_DOMAINS.index(d))


# ---------------------------------------------------------------------------- generic label_int with a configurable label column
def label_int_field(spec, row, cfg):
    """`label_int` shape with spec['label_field'] (e.g. yahoo's 'topic' column).

    Needed because the built-in label_int shape hardcodes row['label'].
    """
    lf = spec.get("label_field", "label")
    names = spec.get("label_names") or spec.get("_names") or []
    lab = _lab(row, lf)
    if not names or not _ok(lab, names):
        return None
    state = {"language": spec.get("lang", "en")}
    for i, f in enumerate(spec.get("text") or []):
        state["text" if i == 0 else f"text_{i+1}"] = clean(row.get(f))
    if not state.get("text"):
        return None
    return state, "choice", (spec["instructions"], names, lab)


ADAPTERS = {
    "nli_loc": nli_loc,
    "kmmlu": kmmlu,
    "marc_ja": marc_ja,
    "jsts": jsts,
    "tnews": tnews,
    "afqmc": afqmc,
    "clue_wsc": clue_wsc,
    "c3": c3,
    "math_qa": math_qa,
    "social_i_qa": social_i_qa,
    "go_emotions": go_emotions,
    "stsb": stsb,
    "mtop_intent": mtop_intent,
    "mtop_domain": mtop_domain,
    "label_int_field": label_int_field,
}


# ---------------------------------------------------------------------------- SPECS
SPECS = [
    # =============================== Korean ===============================
    dict(name="kmmlu", workflow="kmmlu-{config}", path="HAERAE-HUB/KMMLU", configs="ALL",
         splits=["train"], shape="auto", adapter="kmmlu", lang="ko", target=700,
         stream=False),
    dict(name="nsmc", workflow="nsmc-sentiment", path="Blpeng/nsmc", splits=["train"],
         shape="label_int", text=["document"], label_names=["부정", "긍정"], lang="ko",
         instructions="이 영화 리뷰의 감정은 긍정인가 부정인가?", target=30000,
         stream=False),
    dict(name="kor_nli_snli", workflow="kornli-snli-ko", path="kakaobrain/kor_nli",
         configs=["snli"], splits=["train"], shape="auto", adapter="nli_loc", lang="ko",
         label_names=["entailment", "neutral", "contradiction"],
         instructions="두 문장의 논리적 관계는 무엇인가?", target=20000, stream="shuffle"),
    dict(name="kor_nli_mnli", workflow="kornli-mnli-ko", path="kakaobrain/kor_nli",
         configs=["multi_nli"], splits=["train"], shape="auto", adapter="nli_loc", lang="ko",
         label_names=["entailment", "neutral", "contradiction"],
         instructions="두 문장의 논리적 관계는 무엇인가?", target=20000, stream="shuffle"),

    # =============================== Japanese ===============================
    dict(name="jnli", workflow="jnli-ja", path="zenless-lab/jnli", configs=["default"],
         splits=["train"], shape="auto", adapter="nli_loc", lang="ja",
         label_names=["entailment", "neutral", "contradiction"],
         instructions="二つの文の論理的な関係は何ですか?", target=15000, stream=False),
    dict(name="marc_ja", workflow="marc-ja-sentiment", path="yangwang825/marc-ja",
         splits=["train"], shape="auto", adapter="marc_ja", lang="ja", target=20000,
         stream=False),
    dict(name="jsts", workflow="jsts-ja", path="zenless-lab/jsts", splits=["train"],
         shape="auto", adapter="jsts", lang="ja", target=12000, stream=False),

    # =============================== Chinese ===============================
    dict(name="clue_tnews", workflow="clue-tnews", path="clue/clue", configs=["tnews"],
         splits=["train"], shape="auto", adapter="tnews", lang="zh", target=20000,
         stream=False),
    dict(name="clue_ocnli", workflow="clue-ocnli", path="clue/clue", configs=["ocnli"],
         splits=["train"], shape="auto", adapter="nli_loc", lang="zh",
         label_names=["neutral", "entailment", "contradiction"],
         instructions="两句话之间的逻辑关系是什么?", target=20000, stream=False),
    dict(name="clue_cmnli", workflow="clue-cmnli", path="clue/clue", configs=["cmnli"],
         splits=["train"], shape="auto", adapter="nli_loc", lang="zh",
         label_names=["neutral", "entailment", "contradiction"],
         instructions="两句话之间的逻辑关系是什么?", target=25000, stream="shuffle"),
    dict(name="clue_afqmc", workflow="clue-afqmc", path="clue/clue", configs=["afqmc"],
         splits=["train"], shape="auto", adapter="afqmc", lang="zh", target=15000,
         stream=False),
    dict(name="clue_wsc", workflow="clue-wsc2020", path="clue/clue", configs=["cluewsc2020"],
         splits=["train"], shape="auto", adapter="clue_wsc", lang="zh", target=1200,
         stream=False),
    dict(name="c3", workflow="c3-zh-reading", path="dataset-org/c3", configs=["mixed"],
         splits=["train"], shape="auto", adapter="c3", lang="zh", target=3000, stream=False),

    # ==================== English reasoning / math / knowledge ====================
    dict(name="logiqa2", workflow="logiqa2-reasoning", path="jeggers/logiqa2_formatted",
         splits=["train"], shape="choices_list", text=["text"], question="question",
         choices="options", answer="answer",
         instructions="Read the passage and pick the option that must be true.",
         target=12000, stream=False),
    dict(name="math_qa", workflow="math-qa", path="tasksource/math_qa", splits=["train"],
         shape="auto", adapter="math_qa", target=25000, stream=False),
    dict(name="social_i_qa", workflow="social-i-qa", path="tasksource/social_i_qa",
         splits=["train"], shape="auto", adapter="social_i_qa", target=25000, stream=False),
    dict(name="cosmos_qa", workflow="cosmos-qa-commonsense", path="Samsoup/cosmos_qa",
         splits=["train"], shape="choices_list", text=["context"], question="question",
         choices=("answer0", "answer1", "answer2", "answer3"), answer="label",
         instructions="Pick the option that best answers the question about the context.",
         target=20000, stream=False),
    dict(name="fever_nli", workflow="fever-fact-verification", path="pietrolesci/nli_fever",
         splits=["train"], shape="auto", adapter="nli_loc",
         label_names=["entailment", "neutral", "contradiction"],
         instructions="What is the logical relation between `premise` and `hypothesis`?",
         target=30000, stream="shuffle"),
    dict(name="yahoo_topics", workflow="yahoo-answers-topic",
         path="community-datasets/yahoo_answers_topics", configs=["yahoo_answers_topics"],
         splits=["train"], shape="auto", adapter="label_int_field", label_field="topic",
         text=["question_title", "question_content"],
         label_names=["Society & Culture", "Science & Mathematics", "Health",
                      "Education & Reference", "Computers & Internet", "Sports",
                      "Business & Finance", "Entertainment & Music", "Family & Relationships",
                      "Politics & Government"],
         instructions="Which topic does the question belong to?", target=20000, stream=True),
    dict(name="fin_phrasebank_atrost", workflow="financial-phrasebank-sentiment",
         path="atrost/financial_phrasebank", splits=["train"], shape="label_int",
         text=["sentence"],
         instructions="What is the sentiment of the financial sentence?", target=3000,
         stream=False),
    dict(name="fin_phrasebank_warwick", workflow="financial-phrasebank-sentiment-2",
         path="warwickai/financial_phrasebank_mirror", splits=["train"], shape="label_int",
         text=["sentence"], label_names=["negative", "neutral", "positive"],
         instructions="What is the sentiment of the financial sentence?", target=4800,
         stream=False),
    dict(name="go_emotions", workflow="go-emotions-binary",
         path="google-research-datasets/go_emotions", configs=["simplified"],
         splits=["train"], shape="auto", adapter="go_emotions", target=20000, stream=False),

    # =============================== GLUE set ===============================
    dict(name="glue_mrpc", workflow="glue-mrpc", path="nyu-mll/glue", configs=["mrpc"],
         splits=["train"], shape="pair_binary", label="label", text=["sentence1", "sentence2"],
         label_names=["not_equivalent", "equivalent"],
         instructions="Do the two sentences express the same meaning?", target=3600),
    dict(name="glue_qqp", workflow="glue-qqp", path="nyu-mll/glue", configs=["qqp"],
         splits=["train"], shape="pair_binary", label="label", text=["question1", "question2"],
         label_names=["not_duplicate", "duplicate"],
         instructions="Are the two questions duplicates of each other?", target=20000,
         stream="shuffle"),
    dict(name="glue_sst2", workflow="glue-sst2", path="nyu-mll/glue", configs=["sst2"],
         splits=["train"], shape="label_int", text=["sentence"],
         instructions="Is the sentiment of the movie review positive or negative?",
         target=20000),
    dict(name="glue_cola", workflow="glue-cola", path="nyu-mll/glue", configs=["cola"],
         splits=["train"], shape="label_int", text=["sentence"],
         instructions="Is the sentence grammatically acceptable?", target=8000),
    dict(name="glue_qnli", workflow="glue-qnli", path="nyu-mll/glue", configs=["qnli"],
         splits=["train"], shape="pair_binary", label="label", text=["question", "sentence"],
         label_names=["entailment", "not_entailment"],
         instructions="Does the sentence contain the answer to the question?",
         target=15000, stream="shuffle"),
    dict(name="glue_rte", workflow="glue-rte", path="nyu-mll/glue", configs=["rte"],
         splits=["train"], shape="pair_binary", label="label", text=["sentence1", "sentence2"],
         label_names=["entailment", "not_entailment"],
         instructions="Does the first sentence entail the second one?", target=2400),
    dict(name="glue_wnli", workflow="glue-wnli", path="nyu-mll/glue", configs=["wnli"],
         splits=["train"], shape="pair_binary", label="label", text=["sentence1", "sentence2"],
         label_names=["not_entailment", "entailment"],
         instructions="Does the first sentence entail the second one?", target=600),
    dict(name="glue_stsb", workflow="glue-stsb", path="nyu-mll/glue", configs=["stsb"],
         splits=["train"], shape="auto", adapter="stsb", target=5700),

    # ======================= multilingual routing (MTOP) =======================
    dict(name="mtop_intent", workflow="mtop-intent", path="tasksource/mtop",
         splits=["train"], shape="auto", adapter="mtop_intent", target=20000,
         stream=True),
    dict(name="mtop_domain", workflow="mtop-domain", path="tasksource/mtop",
         splits=["train"], shape="auto", adapter="mtop_domain", target=20000,
         stream=True),
]

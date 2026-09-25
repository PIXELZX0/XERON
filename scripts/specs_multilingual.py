#!/usr/bin/env python3
"""W2 spec registry: multilingual language-coverage expansion for XERON-1.0.

Loaded by scripts/build_bulk_choice.py --specs-file scripts/specs_multilingual.py
Exports SPECS (list[dict]) and ADAPTERS (dict[str, callable]).

Every spec here was written against a *probed* schema (scripts/probe_w2*.py) — no guessing.

Custom adapter contract (see build_bulk_choice.extract):
    fn(spec, row, cfg) -> (state:dict, kind:str, payload) | None
      kind="choice" -> payload = (instructions, options:list[str], correct_idx:int)
      kind="score"  -> payload = (instructions, levels:list[str], target_idx:int)
      kind="noul"   -> payload = (instructions, criteria:dict, p_true:float)

Notes
-----
* `shape` is still set on adapter specs so the builder's label-universe collection
  (`_names`) kicks in where needed (shape == "label_str").
* seed=7 is fixed by the builder; nothing here is random.
"""
import re

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def _clean(v):
    if v is None:
        return ""
    return _WS.sub(" ", str(v)).strip()


def _dedup_opts(opts):
    """Drop empty/placeholder options, keep order."""
    out = []
    for o in opts:
        o = _clean(o)
        if o == "" or o.lower() in ("none", "n/a", "na", "null"):
            continue
        out.append(o)
    return out


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
XWINO_LANG = {"en": "en", "fr": "fr", "jp": "ja", "pt": "pt", "ru": "ru", "zh": "zh"}


def xwinograd(spec, row, cfg):
    """Muennighoff/xwinograd: sentence with `_` blank + option1/option2 + answer '1'/'2'."""
    lang = XWINO_LANG.get(cfg, cfg or "en")
    sentence = _clean(row.get("sentence"))
    opts = [_clean(row.get("option1")), _clean(row.get("option2"))]
    if "" in opts or opts[0] == opts[1]:
        return None
    a = _clean(row.get("answer"))
    idx = int(a) - 1 if a.isdigit() else -1
    if idx not in (0, 1):
        return None
    state = {"language": lang, "sentence": sentence}
    return state, "choice", (
        "Which noun phrase does the blank `_` in the sentence refer to?", opts, idx)


def xcopa(spec, row, cfg):
    """cambridgeltl/xcopa: premise + choice1/choice2 + question(cause|effect) + label."""
    premise = _clean(row.get("premise"))
    opts = [_clean(row.get("choice1")), _clean(row.get("choice2"))]
    if "" in opts or opts[0] == opts[1]:
        return None
    lab = int(row["label"])
    if lab not in (0, 1):
        return None
    rel = _clean(row.get("question")).lower()
    if rel == "cause":
        ins = "Which option is the cause of the situation described in `premise`?"
    elif rel == "effect":
        ins = "Which option is the effect of the situation described in `premise`?"
    else:
        ins = "Which option best fits the situation described in `premise`?"
        rel = "unknown"
    state = {"language": cfg, "premise": premise, "relation": rel}
    return state, "choice", (ins, opts, lab)


def mmlu_prox(spec, row, cfg):
    """li-lab/MMLU-ProX: 10 option slots (some are 'None' placeholders) + answer_index."""
    raw = [str(row.get(f"option_{i}", "") or "").strip() for i in range(10)]
    kept = [i for i, v in enumerate(raw)
            if v != "" and v.lower() not in ("none", "n/a", "na", "null")]
    ai = row.get("answer_index")
    if ai is None or len(kept) < 2:
        return None
    ai = int(ai)
    if ai not in kept:
        return None
    opts = [raw[i] for i in kept]
    if len(set(opts)) != len(opts):
        return None
    state = {"language": cfg,
             "category": _clean(row.get("category")),
             "question": _clean(row.get("question"))}
    return state, "choice", (
        spec.get("instructions") or "Which option is the correct answer to the question?",
        opts, kept.index(ai))


def masakhanews(spec, row, cfg):
    """masakhane/masakhanews: headline + body text -> category (label universe via shape=label_str)."""
    names = spec.get("_names") or []
    lab_s = _clean(row.get("category"))
    if not names or lab_s not in names:
        return None
    state = {"language": cfg,
             "headline": _clean(row.get("headline")),
             "text": _clean(row.get("text"))}
    return state, "choice", (
        spec.get("instructions") or "Which news category does the article belong to?",
        names, names.index(lab_s))


ADAPTERS = {
    "xwinograd": xwinograd,
    "xcopa": xcopa,
    "mmlu_prox": mmlu_prox,
    "masakhanews": masakhanews,
}

# ---------------------------------------------------------------------------
# explicit config lists (probed; avoids 'default'/'translation-*' surprises)
# ---------------------------------------------------------------------------
XCOPA_LANGS = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]
XSTORY_LANGS = ["en", "ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]
PAWSX_LANGS = ["de", "en", "es", "fr", "ja", "ko", "zh"]
XWINO_CFGS = ["en", "fr", "jp", "pt", "ru", "zh"]
MASSIVE_SCENARIO_LANGS = [
    "af", "am", "ar", "az", "bn", "cy", "da", "de", "el", "en", "es", "fa", "fi", "fr",
    "he", "hi", "hu", "hy", "id", "is", "it", "ja", "jv", "ka", "km", "kn", "ko", "lv",
    "ml", "mn", "ms", "my", "nb", "nl", "pl", "pt", "ro", "ru", "sl", "sq", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "ur", "vi", "zh-CN", "zh-TW",
]
MMLU_PROX_LANGS = [  # 'en' excluded: W1 already ships TIGER-Lab/MMLU-Pro (English)
    "af", "ar", "bn", "cs", "de", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "mr",
    "ne", "pt", "ru", "sr", "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zh", "zu",
]
AML_MIRRORS = [  # SetFit mirrors of amazon_reviews_multi (script-free parquet)
    ("de", "SetFit/amazon_reviews_multi_de"),
    ("en", "SetFit/amazon_reviews_multi_en"),
    ("es", "SetFit/amazon_reviews_multi_es"),
    ("fr", "SetFit/amazon_reviews_multi_fr"),
    ("ja", "SetFit/amazon_reviews_multi_ja"),
    ("zh", "SetFit/amazon_reviews_multi_zh"),
]
MLNLI_LANGS = ["ar", "bn", "de", "es", "fa", "fr", "he", "hi", "id", "it", "ja", "ko",
               "mr", "nl", "pl", "ps", "pt", "ru", "sv", "sw", "ta", "tr", "uk", "ur",
               "vi", "zh"]
MLNLI_SOURCES = ["anli", "fever", "ling", "mnli", "wanli"]

NLI_NAMES = ["entailment", "neutral", "contradiction"]
NLI_INSTR = "What is the logical relation between `premise` and `hypothesis`?"

# ---------------------------------------------------------------------------
# SPECS
# ---------------------------------------------------------------------------
SPECS = [
    # ======================= 1. reading comprehension, 122 languages =================
    dict(name="belebele", workflow="belebele-reading",
         path="facebook/belebele", configs="ALL", splits=["test"],
         shape="choices_list", text=["flores_passage"], question="question",
         choices=("mc_answer1", "mc_answer2", "mc_answer3", "mc_answer4"),
         answer="correct_answer_num", lang_from_config=True,
         instructions="Read `flores_passage`, then choose the correct answer to `question`.",
         target=900, stream=False),

    # ======================= 2. topic classification, 204 languages ==================
    dict(name="sib200", workflow="sib200-topic",
         path="Davlan/sib200", configs="ALL",
         splits=["train", "validation", "test"],
         shape="label_str", label="category", text=["text"], lang_from_config=True,
         instructions="Which topic does the text belong to?",
         target=1000, small=(3, 5), stream=False),

    # ======================= 3. knowledge QA, 42 languages ==========================
    dict(name="global_mmlu", workflow="global-mmlu-{config}",
         path="CohereForAI/Global-MMLU", configs="ALL", splits=["test"],
         shape="choices_list", text=["subject"], question="question",
         choices=("option_a", "option_b", "option_c", "option_d"), answer="answer",
         lang_from_config=True,
         instructions="Choose the correct answer to the question.",
         target=3500, stream=False),

    # ======================= 4. assistant-scenario routing, 51 languages =============
    dict(name="massive_scenario", workflow="massive-scenario",
         path="mteb/amazon_massive_scenario", configs=MASSIVE_SCENARIO_LANGS,
         splits=["train"],
         shape="label_str", label="label", text=["text"], lang_field="lang",
         instructions="Which virtual-assistant scenario does the user's utterance belong to?",
         target=1800, small=(4, 5), stream=False),

    # ======================= 5. hard reasoning QA, 28 languages ======================
    dict(name="mmlu_prox", workflow="mmlu-prox-{config}",
         path="li-lab/MMLU-ProX", configs=MMLU_PROX_LANGS, splits=["test"],
         shape="choices_list", adapter="mmlu_prox", text=["category"], question="question",
         lang_from_config=True,
         instructions="Which option is the correct answer to the question?",
         target=2500, small=(4, 5), stream=False),

    # ======================= 6. culturally-grounded QA, 44 languages =================
    dict(name="include_lite_44", workflow="include-44",
         path="CohereLabs/include-lite-44", configs="ALL", splits=["test"],
         shape="choices_list", text=["subject"], question="question",
         choices="choices", answer="answer", lang_from_config=True,
         instructions="Which option is the correct answer to the question?",
         target=250, stream=False),

    # ======================= 7. African-language sentiment (15 languages) ============
    dict(name="afrisenti", workflow="afrisenti",
         path="masakhane/afrisenti", configs="ALL", splits=["train"],
         shape="label_str", label="label", text=["tweet"], lang_from_config=True,
         instructions="What is the sentiment of the tweet?",
         target=2500, stream=False),

    # ======================= 8. African-language news topic (16 languages) ===========
    dict(name="masakhanews", workflow="masakhanews",
         path="masakhane/masakhanews", configs="ALL", splits=["train", "validation"],
         shape="label_str", adapter="masakhanews", label="category",
         instructions="Which news category does the article belong to?",
         target=1500, small=(4, 5), stream=False),

    # ======================= 9. XCOPA causal reasoning (11 languages) ================
    dict(name="xcopa", workflow="xcopa",
         path="cambridgeltl/xcopa", configs=XCOPA_LANGS, splits=["validation", "test"],
         shape="copa", adapter="xcopa",
         target=600, stream=False),

    # ======================= 10. XStoryCloze (11 languages) ==========================
    dict(name="xstory_cloze", workflow="xstory-cloze",
         path="juletxara/xstory_cloze", configs=XSTORY_LANGS, splits=["train", "eval"],
         shape="story_cloze", answer="answer_right_ending", lang_from_config=True,
         instructions="Which sentence is the correct ending of the story?",
         target=2000, stream=False),

    # ======================= 11. PAWS-X paraphrase detection (7 languages) ===========
    dict(name="paws_x", workflow="paws-x",
         path="google-research-datasets/paws-x", configs=PAWSX_LANGS, splits=["train"],
         shape="pair_binary", text=["sentence1", "sentence2"], label="label",
         label_names=["not paraphrase", "paraphrase"], lang_from_config=True,
         instructions="Are `sentence1` and `sentence2` paraphrases of each other?",
         target=8000, stream="shuffle"),

    # ======================= 12. multilingual review rating (6 languages) ============
    # (mteb/amazon_reviews_multi is script-only -> per-language SetFit mirrors)
    dict(name="amazon_reviews_de", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_de", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="de",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),
    dict(name="amazon_reviews_en", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_en", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="en",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),
    dict(name="amazon_reviews_es", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_es", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="es",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),
    dict(name="amazon_reviews_fr", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_fr", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="fr",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),
    dict(name="amazon_reviews_ja", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_ja", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="ja",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),
    dict(name="amazon_reviews_zh", workflow="amazon-reviews-multi",
         path="SetFit/amazon_reviews_multi_zh", splits=["train"],
         shape="ordinal", text=["text"], label_lo=0, label_hi=4, lang="zh",
         levels=["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
         instructions="How many stars did the reviewer give?",
         target=12000, stream="shuffle"),

    # ======================= 13. XWinograd coreference (6 languages) =================
    dict(name="xwinograd", workflow="xwinograd",
         path="Muennighoff/xwinograd", configs=XWINO_CFGS, splits=["test"],
         shape="choices_list", adapter="xwinograd",
         target=2400, stream=False),
]

# ======================= 14. multilingual NLI, 26 languages (5 sources each) =========
# splits are named '{lang}_{source}' -> one spec per language so `lang` is exact.
for _lang in MLNLI_LANGS:
    SPECS.append(dict(
        name=f"mlnli_{_lang}", workflow="multilingual-nli",
        path="MoritzLaurer/multilingual-NLI-26lang-2mil7",
        splits=[f"{_lang}_{s}" for s in MLNLI_SOURCES],
        shape="nli", label_names=NLI_NAMES, lang=_lang,
        instructions=NLI_INSTR, target=250, stream="shuffle"))

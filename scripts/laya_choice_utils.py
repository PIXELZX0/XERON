#!/usr/bin/env python3
"""Shared helpers for building Laya-format multiple-choice decision datasets.

Every emitted row follows the Laya native format consumed by scripts/preprocess.py:

    {"guid", "workflow", "state", "questions", "gold"}     (state/questions/gold = JSON strings)

Key invariants enforced here:

* the correct option is placed at a *random* label position (no positional shortcut),
* the option texts are written both into `state` (as a labelled mapping) and into
  `questions[qid].criteria` (so each [MASK] marker carries its own option text),
* the label letters used in `state` and in `criteria` are always the same mapping,
* `gold` is a one-hot distribution over exactly those labels and sums to 1.0,
* no answer marker (`Answer: B`, `answerKey`, `정답`, `label`, ...) is ever copied
  into `state`.

Used by build_clinc_dataset.py / build_mmlu_dataset.py / build_hellaswag_dataset.py /
build_arc_dataset.py.
"""
import json
import re

LABELS = "ABCDEF"          # Laya choice head budget: 2..6 options
MIN_OPTS, MAX_OPTS = 2, 6

# Patterns that would betray the gold answer if they leaked into `state`.
# Deliberately narrow: bare words like "correct"/"gold"/"answer" occur constantly in
# ordinary question text ("Find the exact answer: 942 / 3"), so matching them would only
# produce noise. What matters is an *answer key* — a marker bound to a label.
LEAK_PATTERNS = [
    re.compile(r"(?i)\banswer\s*[:=]\s*[A-F0-9]\b"),      # "Answer: B"
    re.compile(r"(?i)\banswerkey\b"),                      # raw field name
    re.compile(r"(?i)\banswer\s+key\b"),
    re.compile(r"(?i)\bcorrect\s+(answer|option|choice|label)\s*(is|:|=)"),
    re.compile(r"(?i)\bthe\s+answer\s+is\s*[A-F0-9]\b"),
    re.compile(r"(?i)\b(gold|correct|true)\s+(label|index|option|choice)\b"),
    re.compile(r"정답\s*[:：]?"),
    re.compile(r"(?i)\bground\s*truth\b"),
    re.compile(r"(?i)\"(answer|answerkey|label|target|gold)\"\s*:"),
]


def scan_leak(text):
    """Return the list of leak patterns matched in `text` (empty = clean)."""
    return [p.pattern for p in LEAK_PATTERNS if p.search(text)]


def _clean(text):
    """Collapse whitespace and strip answer-marker artefacts from free text."""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    # ARC/MMLU style trailing answer keys that occasionally live inside the question text
    text = re.sub(r"(?i)\s*\banswer\s*[:=]\s*[A-F0-9]\b\.?\s*$", "", text)
    return text.strip()


def build_row(guid, workflow, state, instructions, options, correct_idx, rng,
              choices_key="choices"):
    """Build one Laya row with a randomly permuted option order.

    state        : dict of non-answer context (question / context / utterance ...)
    options      : list[str], canonical order
    correct_idx  : index into `options` of the gold option
    rng          : random.Random instance (seeded) used for the permutation
    choices_key  : key under which the labelled option map is stored in `state`
    """
    n = len(options)
    if not (MIN_OPTS <= n <= MAX_OPTS):
        return None
    if not (0 <= correct_idx < n):
        return None

    opts = [_clean(o) for o in options]
    if any(o == "" for o in opts):
        return None
    # an option repeated verbatim makes the gold ambiguous -> drop the row
    if len(set(opts)) != n:
        return None

    order = list(range(n))
    rng.shuffle(order)
    labels = list(LABELS[:n])

    criteria, shown, correct_label = {}, {}, None
    for pos, oi in enumerate(order):
        lab = labels[pos]
        criteria[lab] = opts[oi]
        shown[lab] = opts[oi]
        if oi == correct_idx:
            correct_label = lab
    if correct_label is None:
        return None

    state = dict(state)
    state[choices_key] = shown

    q = {"decision": {"type": "choice", "instructions": instructions, "criteria": criteria}}
    g = {"decision": {"probabilities": {l: (1.0 if l == correct_label else 0.0) for l in labels}}}
    return {
        "guid": guid,
        "workflow": workflow,
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps(q, ensure_ascii=False),
        "gold": json.dumps(g, ensure_ascii=False),
    }


def dedupe_key(state_json):
    """Canonical text key of a state (same normalization as scripts/audit_duplicates.py)."""
    try:
        s = json.loads(state_json)
    except Exception:                                     # noqa: BLE001
        s = state_json
    if isinstance(s, dict):
        s = " || ".join(f"{k}={v}" for k, v in sorted(s.items()))
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

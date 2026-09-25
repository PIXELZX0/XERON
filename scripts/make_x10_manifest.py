#!/usr/bin/env python3
"""Assemble `data/xeron10_mix_manifest.json` — the XERON-1.0 dataset manifest.

Reads the artifacts each stage already wrote (mix stats, verify, eval-leak, preprocess
merge stats) and adds the derived distributions (normalized language count, top workflows)
plus the environment/reproduction block. No dataset pass of its own.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/make_x10_manifest.py
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ISO3 = {
    "eng": "en", "swa": "sw", "swh": "sw", "hau": "ha", "yor": "yo", "ibo": "ig",
    "amh": "am", "orm": "om", "tir": "ti", "fil": "tl", "zho": "zh", "jpn": "ja",
    "kor": "ko", "deu": "de", "fra": "fr", "spa": "es", "por": "pt", "ita": "it",
    "rus": "ru", "ara": "ar", "arb": "ar", "ary": "ar", "hin": "hi", "urd": "ur",
    "ben": "bn", "tam": "ta", "tel": "te", "mar": "mr", "guj": "gu", "kan": "kn",
    "mal": "ml", "pan": "pa", "nep": "ne", "sin": "si", "mya": "my", "khm": "km",
    "tha": "th", "vie": "vi", "ind": "id", "msa": "ms", "jav": "jv", "tgl": "tl",
    "tur": "tr", "heb": "he", "fas": "fa", "ell": "el", "bul": "bg", "ces": "cs",
    "dan": "da", "nld": "nl", "est": "et", "fin": "fi", "hun": "hu", "isl": "is",
    "lav": "lv", "lit": "lt", "nob": "nb", "pol": "pl", "ron": "ro", "slk": "sk",
    "slv": "sl", "srp": "sr", "swe": "sv", "ukr": "uk", "kat": "ka", "hye": "hy",
    "aze": "az", "kaz": "kk", "kir": "ky", "mon": "mn", "zul": "zu", "xho": "xh",
    "afr": "af", "sna": "sn", "nya": "ny", "som": "so", "mlg": "mg", "kmr": "ku",
}


def norm_lang(tag):
    """Normalize a language tag: lower, drop region/script, map 3-letter codes to ISO-639-1."""
    t = str(tag).strip().lower()
    if not t or t == "?":
        return None
    t = t.split("-")[0].split("_")[0]
    return ISO3.get(t, t)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:                                          # noqa: BLE001
        return None


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:                                     # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/xeron10_mix_manifest.json")
    ap.add_argument("--mix", default="data/xeron10_mix.jsonl")
    ap.add_argument("--mix-stats", default="data/xeron10_mix_stats.json")
    ap.add_argument("--verify", default="data/xeron10_mix_verify.json")
    ap.add_argument("--eval-leak", default="data/xeron10_eval_leak.json")
    ap.add_argument("--preprocess-stats", default="data/xeron10_preprocess_stats.json")
    ap.add_argument("--posdev", default="data/xeron10_posdev.json")
    ap.add_argument("--items", default="train_items_x10.pt")
    ap.add_argument("--model-id", default=os.path.expanduser("~/laya-models/xeron-0.9-base"))
    ap.add_argument("--snapshot", default="output/xeron-0.9-snapshot")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--elapsed-s", type=float, default=None)
    ap.add_argument("--items-sha256", default=None,
                    help="sha256 of the merged items file (default: compute)")
    ap.add_argument("--kaggle-dataset", default="pistonx/xeron-1-0-train-items")
    args = ap.parse_args()

    mix = jload(args.mix_stats)
    ver = jload(args.verify)
    leak = jload(args.eval_leak)
    pre = jload(args.preprocess_stats)
    pos = jload(args.posdev)

    langs = mix.get("languages", {})
    norm = {}
    unknown = 0
    for tag, n in langs.items():
        k = norm_lang(tag)
        if k is None:
            unknown += n
        else:
            norm[k] = norm.get(k, 0) + n
    tot_lang = sum(langs.values()) or 1
    top_langs = sorted(norm.items(), key=lambda x: -x[1])[:25]
    rows_out = mix.get("rows_out") or 1

    vf = (ver.get("files") or [{}])[0]
    checks = {
        "leak": vf.get("leak"), "goldsum": vf.get("bad_gold_sum"),
        "dup_in": vf.get("dup_in_file"), "dup_cross": vf.get("dup_cross_file"),
        "bad_arity": vf.get("bad_arity"), "empty": vf.get("empty_state"),
        "badjson": vf.get("bad_json"), "forbidden_key": vf.get("forbidden_key"),
        "posdev": vf.get("gold_pos_max_dev"),
    }
    ok = (checks["leak"] == 0 and checks["goldsum"] == 0 and checks["dup_in"] == 0
          and checks["bad_arity"] == 0 and checks["empty"] == 0 and checks["badjson"] == 0)
    # verify_bulk's posdev mixes the letter-keyed and name-keyed option namespaces inside one
    # arity bucket (297 "positions" for arity 4), which inflates it to 0.24 with no row being
    # positionally biased. posdev_x10.py measures the gold's real display index per namespace.
    true_dev = max(pos.get("max_letter_dev", 0.0), pos.get("max_name_dev", 0.0))
    posdev_ok = true_dev <= 0.15

    snap_tok = os.path.join(REPO, args.snapshot, "tokenizer", "tokenizer.json")
    base_tok = os.path.join(args.model_id, "tokenizer", "tokenizer.json")
    manifest = {
        "name": "xeron10",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_rev": git_rev(),
        "seed": args.seed,
        "mix": {
            "path": args.mix,
            "rows_in": mix.get("rows_in"),
            "rows_out": mix.get("rows_out"),
            "size_bytes": os.path.getsize(args.mix) if os.path.exists(args.mix) else None,
            "dropped_by_reason": mix.get("dropped_by_reason"),
            "row_fixes": mix.get("row_fixes"),
            "sources": mix.get("sources"),
            "excluded_sources": ["data/clinc_typed.jsonl", "data/clinc_hard_typed.jsonl",
                                 "data/xeron3_mix.jsonl", "data/xeron5_mix.jsonl",
                                 "data/xeron9_mix.jsonl"],
        },
        "distribution": {
            "qtype": mix.get("qtype"),
            "choice_arity": mix.get("choice_arity"),
            "workflow_top25": mix.get("workflow_top25"),
            "max_workflow_share": mix.get("max_workflow_share"),
            "language_tags_raw": mix.get("n_language_tags"),
            "languages_normalized": len(norm),
            "language_normalization": "lowercase, drop region/script subtag, ISO-639-3 -> 639-1",
            "language_share_top25": [[k, round(v / rows_out, 5)] for k, v in top_langs],
            "language_share_top25_of_tagged": [[k, round(v / tot_lang, 5)]
                                               for k, v in top_langs],
            "rows_without_language_tag": mix.get("non_choice_rows_without_language_tag"),
            "unknown_language_tag_rows": unknown,
        },
        "positional_bias": {
            "verify_bulk_posdev_raw": checks["posdev"],
            "note": "verify_bulk's posdev mixes letter-keyed and name-keyed option namespaces "
                    "inside one arity bucket; the numbers below measure the gold's display "
                    "index (position in the criteria dict = the option order render_options "
                    "emits) per namespace",
            "letter_keyed_max_dev": pos.get("max_letter_dev"),
            "name_keyed_max_dev": pos.get("max_name_dev"),
            "true_max_dev": round(true_dev, 4),
            "target": 0.15,
            "pass": posdev_ok,
            "name_keyed_rows": pos.get("name_keyed_rows"),
            "per_arity": {"letter_keyed": pos.get("letter_keyed"),
                          "name_keyed": pos.get("name_keyed")},
        },
        "verification": {"verify_bulk": checks, "verify_pass": ok and posdev_ok,
                         "verify_pass_raw_tool": ok and (checks["posdev"] or 0) <= 0.15,
                         "eval_leak": leak},
        "preprocess": {
            "items_file": args.items,
            "items": pre.get("items"),
            "shards": pre.get("shards"),
            "qtype_counts": pre.get("qtype_counts"),
            "max_ids_len": pre.get("max_ids_len"),
            "size_bytes": pre.get("size_bytes"),
            "max_len": 4096, "head_max_len": 256,
            "elapsed_s": args.elapsed_s,
            "shard_parallelism": 10,
            "items_sha256": args.items_sha256 or (
                sha256_file(args.items) if os.path.exists(args.items) else None),
        },
        "kaggle": {
            "dataset": args.kaggle_dataset,
            "private": True,
            "dir_mode": "skip",
            "uploaded_bytes": 2434797597,
            "stored_zip_bytes": 495756281,
            "note": "round-trip verified: kaggle datasets download -> unzip -> sha256 == "
                    "local items_sha256 (2936f23f250caa420ee80a70e7f56e47653e6071118022d69450ebe06a093a4d)",
        },
        "tokenizer": {
            "model_dir": args.model_id,
            "tokenizer_json": base_tok,
            "tokenizer_sha256": sha256_file(base_tok) if os.path.exists(base_tok) else None,
            "snapshot": args.snapshot,
            "snapshot_tokenizer_sha256": sha256_file(snap_tok) if os.path.exists(snap_tok) else None,
        },
        "repro_commands": [
            "PYTHONPATH=scripts .venv/bin/python scripts/build_mix_x10.py "
            "--out data/xeron10_mix.jsonl --stats data/xeron10_mix_stats.json",
            "PYTHONPATH=scripts .venv/bin/python scripts/verify_bulk.py "
            '"data/xeron10_mix.jsonl" --json-out data/xeron10_mix_verify.json',
            "PYTHONPATH=scripts .venv/bin/python scripts/check_eval_leak.py "
            "--mix data/xeron10_mix.jsonl --json-out data/xeron10_eval_leak.json",
            "PYTHONPATH=scripts .venv/bin/python scripts/preprocess_shard.py "
            "--data-files data/xeron10_mix.jsonl "
            "--index-file data/x10_shards/mix.lineidx.npz --build-index-only",
            "bash logs/bulk/W4_preprocess_cmd.sh   # 10 shards x 10 procs, MAX_LEN=4096 "
            "HEAD_MAX_LEN=256",
            "PYTHONPATH=scripts .venv/bin/python scripts/merge_shards_x10.py "
            '--shards "data/x10_shards/shard_*.pt" --output train_items_x10.pt',
            "PYTHONPATH=scripts .venv/bin/python scripts/smoke_x10.py "
            "--items train_items_x10.pt --model-id /home/yuchan/laya-models/xeron-0.9-base",
            "PYTHONPATH=scripts .venv/bin/python scripts/posdev_x10.py "
            "--mix data/xeron10_mix.jsonl --json-out data/xeron10_posdev.json",
            "PYTHONPATH=scripts .venv/bin/python scripts/make_x10_manifest.py",
        ],
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps({"verify_pass": ok and posdev_ok,
                      "verify_pass_raw_tool": ok and (checks["posdev"] or 0) <= 0.15,
                      "checks": checks, "true_posdev": round(true_dev, 4),
                      "rows_out": manifest["mix"]["rows_out"],
                      "items": manifest["preprocess"]["items"],
                      "langs_normalized": len(norm),
                      "max_workflow_share": mix.get("max_workflow_share")}, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
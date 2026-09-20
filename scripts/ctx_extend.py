#!/usr/bin/env python3
"""Context-extension helper: raise the default context limits of a Laya model dir.

RoPE-based encoders (ModernBERT, sans_pos) have no positional table, so
`max_position_embeddings` is only a validation cap — raising it to 32768 makes
contexts up to 32K trainable/inferable. Also bumps `rl_agent_config.max_len`.

Usage (imported by preprocess.py / train_ddp.py):
    from ctx_extend import ensure_long_context
    ensure_long_context(model_dir, max_len=2048, cap=32768)

The patch is idempotent (never shrinks) and writes back into the model dir;
a fresh `snapshot_download` restores the original if ever needed.
"""
import json
import os

CAP_DEFAULT = 32768  # 4096 x 8 — long-context headroom


def ensure_long_context(model_dir: str, max_len: int, cap: int = CAP_DEFAULT,
                        head_max_len: int = 256) -> dict:
    """Extend encoder max_position_embeddings + rl_agent_config lengths.

    The encoder cap is raised to `cap` (default 32,768 = 4096x8) so RoPE-based
    long contexts up to that length are always allowed; actual training length
    stays controlled by max_len / MICRO_BATCH / MAX_TOKENS_BATCH.

    Returns updated rl_agent_config dict (with max_len/head_max_len applied).
    """
    cfg = {}
    rp = os.path.join(model_dir, "rl_agent_config.json")
    if os.path.exists(rp):
        with open(rp) as f:
            cfg = json.load(f)

    # --- encoder/config.json: max_position_embeddings up to cap ---
    ep = os.path.join(model_dir, "encoder", "config.json")
    if os.path.exists(ep):
        with open(ep) as f:
            ecfg = json.load(f)
        cur = int(ecfg.get("max_position_embeddings", 0) or 0)
        target = max(cur, int(cap))
        if target > cur:
            ecfg["max_position_embeddings"] = target
            with open(ep, "w") as f:
                json.dump(ecfg, f, indent=2)
            print(f"[CTX] encoder max_position_embeddings: {cur} -> {target}")
        else:
            print(f"[CTX] encoder max_position_embeddings already {cur} (>= {target})")

    # --- rl_agent_config.json: max_len / head_max_len ---
    changed = False
    if int(cfg.get("max_len", 0) or 0) < int(max_len):
        cfg["max_len"] = int(max_len)
        changed = True
    if int(cfg.get("head_max_len", 0) or 0) < int(head_max_len):
        cfg["head_max_len"] = int(head_max_len)
        changed = True
    cfg.setdefault("max_len", int(max_len))
    if changed and os.path.exists(rp):
        with open(rp, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[CTX] rl_agent_config max_len -> {cfg['max_len']} "
              f"head_max_len -> {cfg['head_max_len']}")

    return cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--cap", type=int, default=CAP_DEFAULT)
    args = ap.parse_args()
    ensure_long_context(args.model_dir, args.max_len, args.cap)
    print("done")
#!/usr/bin/env python3
"""XERON-1.0 architecture throughput benchmark (A100 80GB).

Measures the *real* XERON training step (same collate, same RLCD loss, same
optimizer step, same DDP wrapper as scripts/train_ddp.py) on a **randomly
initialized** model of the target architecture — no weights are needed for a
throughput / memory benchmark.

Target arch (master-confirmed): ModernBERT encoder hidden_size=1536,
num_hidden_layers=44, intermediate_size=2304, 24 heads, vocab 256000,
MAX_LEN 8192, head_size=1024, head_layers=4, bf16.

Everything is env-parameterized so scripts/a100/run_a100_10.sh can sweep the
matrix and each config is one fresh process (so an OOM in one config does not
poison the others).

    PARAMS    HIDDEN LAYERS INTER HEADS VOCAB      (default 1536 44 2304 24 256000)
    HEAD_SIZE HEAD_LAYERS                          (default 1024 4)
    MAX_LEN   MAXPOS                               (default 8192 32768)
    MICRO_BATCH GRAD_ACCUM GROUP_SIZE              (default 4 8 4)
    GRAD_CKPT OPT8BIT DTYPE DEVICE                 (default 1 0 bf16 cuda)
    STEPS WARMUP SEED                              (default 30 5 42)
    DATA_MODE=file|synth  DATA_FILE  SAMPLE_ITEMS  SYNTH_LEN
    OUT_JSONL  TAG
"""
import gc
import json
import os
import random
import statistics
import sys
import time
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from laya.common import proper_reward, QTYPES  # noqa: E402
from train_ddp import collate_train_batch  # noqa: E402  (identical batch shaping)

# ------------------------------------------------------------------ env
def _i(n, d):
    return int(os.environ.get(n, d))


def _f(n, d):
    return float(os.environ.get(n, d))


def _b(n, d):
    return str(os.environ.get(n, d)).strip().lower() in ("1", "true", "yes", "on")


HIDDEN = _i("HIDDEN", 1536)
LAYERS = _i("LAYERS", 44)
INTER = _i("INTER", 2304)
HEADS = _i("HEADS", 24)
VOCAB = _i("VOCAB", 256000)
HEAD_SIZE = _i("HEAD_SIZE", 1024)
HEAD_LAYERS = _i("HEAD_LAYERS", 4)
MAX_LEN = _i("MAX_LEN", 8192)
MAXPOS = _i("MAXPOS", 32768)
LOCAL_ATTENTION = _i("LOCAL_ATTENTION", 128)
GLOBAL_EVERY = _i("GLOBAL_EVERY", 3)     # mmBERT pattern: every 3rd layer is full attention
MICRO_BATCH = _i("MICRO_BATCH", 4)
GRAD_ACCUM = _i("GRAD_ACCUM", 8)
GROUP_SIZE = _i("GROUP_SIZE", 4)
GRAD_CKPT = _b("GRAD_CKPT", True)
OPT8BIT = _b("OPT8BIT", False)
DTYPE = os.environ.get("DTYPE", "bf16").lower()
DEVICE = os.environ.get("DEVICE", "cuda")
STEPS = _i("STEPS", 30)
WARMUP = _i("WARMUP", 5)
SEED = _i("SEED", 42)
LR_ENCODER = _f("LR_ENCODER", 2e-5)
LR_HEAD = _f("LR_HEAD", 1e-4)
WEIGHT_DECAY = _f("WEIGHT_DECAY", 0.02)
SIGMA = _f("SIGMA", 0.3)
DATA_MODE = os.environ.get("DATA_MODE", "file")
DATA_FILE = os.environ.get("DATA_FILE", os.path.join(ROOT, "train_items_x10.pt"))
SAMPLE_ITEMS = _i("SAMPLE_ITEMS", 40000)
SYNTH_LEN = _i("SYNTH_LEN", 0)
ONLY_PREP = _b("ONLY_PREP", False)
OUT_JSONL = os.environ.get("OUT_JSONL", os.path.join(ROOT, "outputs", "xeron10_a100_bench.jsonl"))
TAG = os.environ.get("TAG", "bench")

AMP_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16

# Base ModernBert config values taken from the mmBERT-base derived snapshot
# (~/laya-models/xeron-0.9-base/encoder/config.json, encoder jhu-clsp/mmBERT-base).
BASE_CFG = dict(
    attention_bias=False,
    attention_dropout=0.0,
    classifier_activation="gelu",
    classifier_bias=False,
    classifier_dropout=0.0,
    classifier_pooling="mean",
    decoder_bias=True,
    deterministic_flash_attn=False,
    embedding_dropout=0.0,
    hidden_activation="gelu",
    initializer_cutoff_factor=2.0,
    initializer_range=0.02,
    layer_norm_eps=1e-05,
    mlp_bias=False,
    mlp_dropout=0.0,
    norm_bias=False,
    norm_eps=1e-05,
    position_embedding_type="sans_pos",
    repad_logits_with_grad=False,
    sparse_pred_ignore_index=-100,
    sparse_prediction=False,
    tie_word_embeddings=True,
    pad_token_id=0,
    bos_token_id=2,
    cls_token_id=1,
    eos_token_id=1,
    sep_token_id=1,
    mask_token_id=4,
    rope_parameters={
        "full_attention": {"rope_theta": 160000, "rope_type": "default"},
        "sliding_attention": {"rope_theta": 160000, "rope_type": "default"},
    },
)


def make_encoder_config():
    from transformers import ModernBertConfig

    cfg = dict(BASE_CFG)
    cfg.update(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        intermediate_size=INTER,
        num_attention_heads=HEADS,
        vocab_size=VOCAB,
        max_position_embeddings=MAXPOS,
        local_attention=LOCAL_ATTENTION,
        global_attn_every_n_layers=GLOBAL_EVERY,
        layer_types=["full_attention" if (i % GLOBAL_EVERY == 0) else "sliding_attention"
                     for i in range(LAYERS)],
    )
    return ModernBertConfig(**cfg)


def build_model():
    from transformers import AutoModel
    from model_xeron import WideHeadDecisionModel

    ecfg = make_encoder_config()
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    model = WideHeadDecisionModel(enc, head_layers=HEAD_LAYERS, head_size=HEAD_SIZE,
                                  n_act=2, dropout=0.1)
    if GRAD_CKPT:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    return model


def param_report(model):
    enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder."))
    head = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))
    total = sum(p.numel() for p in model.parameters())
    return {"params_total": total, "params_encoder": enc, "params_head": head,
            "params_total_m": round(total / 1e6, 2)}


# ------------------------------------------------------------------ data
def synth_items(n):
    """Synthetic items whose lengths follow the measured train_items_x10 histogram."""
    hist = None
    for hp in (os.path.join(ROOT, "configs", "xeron10_len_stats.json"),
               os.path.join(ROOT, "data", "xeron10_len_stats.json")):
        if os.path.exists(hp):
            with open(hp) as f:
                hist = json.load(f)
            break
    rng = random.Random(SEED)
    lengths = []
    if SYNTH_LEN > 0:
        lengths = [SYNTH_LEN] * n
    elif hist:
        bins = sorted((int(k), v) for k, v in hist["hist"].items())
        w = hist["bin_width"]
        keys = [b for b, _ in bins]
        weights = [c for _, c in bins]
        for _ in range(n):
            b = rng.choices(keys, weights=weights, k=1)[0]
            lengths.append(max(8, rng.randint(b, b + w - 1)))
        # append the long tail (>= p99) so the occasionally huge sample is represented
        long_n = max(1, int(n * 0.005))
        for _ in range(long_n):
            lengths.append(rng.choice([1024, 2048, 4096]))
    else:
        lengths = [max(8, int(rng.gauss(172, 90))) for _ in range(n)]

    items = []
    for L in lengths:
        k = rng.randint(2, 6)
        pos = sorted(rng.sample(range(1, max(2, L)), min(k, max(1, L - 1))))
        tgt = [0.0] * k
        tgt[rng.randrange(k)] = 1.0
        items.append({
            "ids": [rng.randrange(2, VOCAB) for _ in range(L)],
            "markers": pos,
            "qtype": rng.choice([0, 0, 0, 1, 2]),
            "target": tgt,
            "label": rng.randrange(k),
        })
    return items


def load_items():
    if DATA_MODE == "synth":
        return synth_items(SAMPLE_ITEMS)
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"DATA_FILE not found: {DATA_FILE}")
    t0 = time.time()
    all_items = torch.load(DATA_FILE, weights_only=False)
    print(f"[data] loaded {len(all_items)} items from {DATA_FILE} in {time.time()-t0:.1f}s",
          flush=True)
    if SAMPLE_ITEMS and len(all_items) > SAMPLE_ITEMS:
        rng = random.Random(SEED)
        all_items = rng.sample(all_items, SAMPLE_ITEMS)
        print(f"[data] subsampled -> {len(all_items)} items", flush=True)
    return all_items


def prep_sample_only():
    items = load_items()
    out = os.path.join(ROOT, "data", "xeron10_bench_sample.pt")
    tmp = out + ".tmp"
    torch.save(items, tmp)
    os.replace(tmp, out)
    mean_len = sum(len(it["ids"]) for it in items) / len(items)
    print(f"[prep] wrote {out} n={len(items)} mean_len={mean_len:.1f}", flush=True)


# ------------------------------------------------------------------ main
def main():
    dist_on = DEVICE.startswith("cuda") and "RANK" in os.environ
    rank, world_size = 0, 1
    if dist_on:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    device = torch.device(DEVICE if not dist_on else f"cuda:{local_rank}")

    if ONLY_PREP:
        prep_sample_only()
        return

    rec = {"tag": TAG, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "arch": {"hidden": HIDDEN, "layers": LAYERS, "inter": INTER, "heads": HEADS,
                    "vocab": VOCAB, "head_size": HEAD_SIZE, "head_layers": HEAD_LAYERS,
                    "max_len": MAX_LEN, "local_attention": LOCAL_ATTENTION,
                    "global_every": GLOBAL_EVERY},
           "run": {"micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
                   "group_size": GROUP_SIZE, "grad_ckpt": int(GRAD_CKPT),
                   "opt8bit": int(OPT8BIT), "dtype": DTYPE, "world_size": world_size,
                   "steps": STEPS, "warmup": WARMUP, "data_mode": DATA_MODE}}
    t_start = time.time()
    try:
        model = build_model()
        rec.update(param_report(model))
        if rank == 0:
            print(f"[bench] {TAG} params={rec['params_total_m']}M "
                  f"(enc {rec['params_encoder']/1e6:.1f}M / head {rec['params_head']/1e6:.1f}M) "
                  f"mb={MICRO_BATCH} ga={GRAD_ACCUM} ckpt={int(GRAD_CKPT)} "
                  f"8bit={int(OPT8BIT)} L={MAX_LEN} mode={DATA_MODE} dev={device}", flush=True)

        model.to(device)
        model.train()
        if dist_on:
            import torch.distributed as dist  # noqa: F811
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            ddp_model = model

        items = load_items()
        rec["sample_items"] = len(items)
        rec["sample_mean_len"] = round(sum(len(it["ids"]) for it in items) / len(items), 2)
        my_items = items[rank::world_size]

        enc_p = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
        head_p = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
        groups = [{"params": enc_p, "lr": LR_ENCODER}, {"params": head_p, "lr": LR_HEAD}]
        if OPT8BIT:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(groups, weight_decay=WEIGHT_DECAY)
            rec["optimizer"] = "AdamW8bit"
        else:
            optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
            rec["optimizer"] = "AdamW"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * 1),
            eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == "fp16"))

        rng = random.Random(SEED)
        order = list(range(len(my_items)))
        rng.shuffle(order)
        cursor = 0

        def next_batch():
            nonlocal cursor
            chunk = []
            for _ in range(MICRO_BATCH):
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                chunk.append(my_items[order[cursor]])
                cursor += 1
            return collate_train_batch(chunk, 0)

        total_steps = WARMUP + STEPS
        timings = []
        padded_tok = 0
        real_tok = 0
        sequences = 0
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        if DEVICE.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        n_oom = 0
        for step in range(total_steps):
            t0 = time.time()
            batch = next_batch()
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            mpos = batch["marker_pos"].to(device)
            mmask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)

            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=DEVICE.startswith("cuda")):
                logits, act = ddp_model(ids, att, mpos, mmask, qtype)

            logits = logits.float()
            k = mmask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * SIGMA * mmask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mmask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mmask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mmask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mmask).sum(-1) / (2 * SIGMA ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mmask, -1e4), -1)).sum(-1).mean()
            loss = (0.5 * loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum += 1
            if accum % GRAD_ACCUM == 0 or (step + 1) == total_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if DEVICE.startswith("cuda"):
                torch.cuda.synchronize()
            dt = time.time() - t0
            if step >= WARMUP:
                timings.append(dt)
                padded_tok += int(ids.numel())
                real_tok += int(att.sum().item())
                sequences += int(ids.shape[0])
            if rank == 0 and step < WARMUP:
                print(f"[bench] warmup {step+1}/{WARMUP} {dt*1000:.0f}ms", flush=True)

        if timings:
            ms = sorted(t * 1000 for t in timings)
            tot = sum(timings)
            rec["timing"] = {
                "n": len(timings),
                "step_ms_mean": round(tot / len(timings) * 1000, 2),
                "step_ms_p10": round(ms[int(0.10 * (len(ms) - 1))], 2),
                "step_ms_p50": round(ms[len(ms) // 2], 2),
                "step_ms_p90": round(ms[int(0.90 * (len(ms) - 1))], 2),
            }
            rec["throughput"] = {
                "tok_s_padded": round(padded_tok / tot, 1),
                "tok_s_real": round(real_tok / tot, 1),
                "seq_s": round(sequences / tot, 2),
                "opt_steps_per_s": round(len(timings) / GRAD_ACCUM / tot, 4),
                "tokens_per_seq_padded": round(padded_tok / max(1, sequences), 1),
            }
            rec["optimizer_steps"] = len(timings) // GRAD_ACCUM
        if DEVICE.startswith("cuda"):
            rec["mem"] = {
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
                "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
                "device": torch.cuda.get_device_name(0),
            }
        rec["ok"] = True
    except RuntimeError as e:
        msg = str(e)
        rec["ok"] = False
        if "out of memory" in msg.lower():
            rec["oom"] = True
            try:
                rec["mem"] = {
                    "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
                    "device": torch.cuda.get_device_name(0),
                }
            except Exception:
                pass
        rec["error"] = msg[:800]
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:800]
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["elapsed_s"] = round(time.time() - t_start, 1)

    if rank == 0:
        os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        print("[bench] " + json.dumps({k: v for k, v in rec.items()
                                       if k in ("tag", "ok", "oom", "throughput", "mem",
                                                "timing", "optimizer_steps", "params_total_m",
                                                "error")}, ensure_ascii=False), flush=True)
    if dist_on:
        import torch.distributed as dist  # noqa: F811
        dist.barrier()
        dist.destroy_process_group()
    # exit 0 even on OOM so the sweep continues
    return 0


if __name__ == "__main__":
    sys.exit(main())

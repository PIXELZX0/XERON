#!/usr/bin/env python3
"""XERON 학습 — Cloud TPU v5e-8 SPMD(FSDPv2) 트레이너.

Kaggle TPU 용. `train_ddp.py`(CUDA/DDP/fp16)의 RLCD 손실·콜레이트·스케줄러를 그대로 이식하고,
학습 병렬화만 DDP → **SPMD + FSDPv2** 로 바꾼다. 이유:

  v5e는 칩당 HBM이 15.75GiB 뿐이라 1.33B 모델을 DDP로 복제하면 정적 메모리만
  15.7GB(파라미터2.6+그래드2.6+AdamW fp32 10.5) 라서 **어떤 배치로도 OOM**.
  파라미터/그래디언트/옵티마이저 상태를 fsdp 축으로 8등분하면 칩당 ~1.3GB → AdamW 그대로 사용 가능.
  (실측: seq128 68.75 seq/s, seq256 33.0, seq512 14.5, seq1024 8.0)

반드시 지킬 것 (전부 실제로 물려서 확인한 함정):
  1. **bf16 캐스팅은 FSDPv2 래핑 전에** — 후에 `.to()` 하면 샤딩 메타데이터가 깨져 35배 느려진다.
  2. 입력 샤딩은 `pl.MpDeviceLoader(input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)))` 로만.
     직접 `xs.mark_sharding` 하면 HF embedding 안에서 `XLAShardedTensor has no attribute global_tensor`.
     → 배치는 **모든 텐서가 rank-2** 인 튜플로 만든다(그래서 qtype을 (B,1)로 패딩).
  3. `xs.set_global_mesh(mesh)` 필수, mesh 축 이름은 반드시 `fsdp`.
  4. `torch_xla.utils.checkpoint`는 `use_reentrant=True`만 지원.
  5. SPMD 모드에서 `xm.get_memory_info`는 실패한다.

데이터: 길이 오름차순 파일을 **등시간 샤드**로 나눠 세션당 하나씩 학습한다(길이 버킷팅 →
collate가 배치 내 최대 길이로 패딩하므로 낭비 최소). `SHARD_INDEX`/`SHARD_COUNT`로 선택.

env: BASE_DIR ITEMS OUT_DIR SHARD_INDEX SHARD_COUNT EPOCHS MICRO_BATCH GRAD_ACCUM
     LR_ENCODER LR_HEAD SIGMA_START SIGMA_END RL_WEIGHT WEIGHT_DECAY HEAD_LAYERS HEAD_SIZE
     MAX_LEN HEAD_MAX_LEN MAX_TOKENS_BATCH CKPT_EVERY RESUME_DCP DCP_DIR RESUME_SHARD
"""
import functools
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("PJRT_DEVICE", "TPU")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_sampler import plan_epoch as plan_batches_epoch  # noqa: E402
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr


def _i(n, d):
    return int(os.environ.get(n, d))


def _f(n, d):
    return float(os.environ.get(n, d))


EPOCHS = _i("EPOCHS", 1)
MICRO_BATCH = _i("MICRO_BATCH", 32)          # GLOBAL batch (sharded across chips)
GRAD_ACCUM = _i("GRAD_ACCUM", 8)
GROUP_SIZE = _i("GROUP_SIZE", 4)
LR_ENCODER = _f("LR_ENCODER", 2.0e-5)
LR_HEAD = _f("LR_HEAD", 5.0e-5)
SIGMA_START = _f("SIGMA_START", 0.3)
SIGMA_END = _f("SIGMA_END", 0.2)
RL_WEIGHT = _f("RL_WEIGHT", 0.5)
WEIGHT_DECAY = _f("WEIGHT_DECAY", 0.02)
HEAD_LAYERS = _i("HEAD_LAYERS", 4)
HEAD_SIZE = _i("HEAD_SIZE", 1024)
HEAD_DROPOUT = _f("HEAD_DROPOUT", 0.0)
MAX_LEN = _i("MAX_LEN", 1024)
HEAD_MAX_LEN = _i("HEAD_MAX_LEN", 256)
CTX_CAP = _i("CTX_CAP", 32768)
MAX_TOKENS_BATCH = _i("MAX_TOKENS_BATCH", 32768)
# Length-bucketed batches (see scripts/batch_sampler.py): groups similar lengths together
# so per-batch padding stays small and MAX_TOKENS_BATCH is enforced.
BATCH_MODE = os.environ.get("BATCH_MODE", "length").strip().lower()
BATCH_BUCKET_MULT = _i("BATCH_BUCKET_MULT", 64)

BASE_DIR = os.environ.get("BASE_DIR", "/kaggle/working/base")
ITEMS = os.environ.get("ITEMS", "/kaggle/input/xeron-1-0-train-items-8192/train_items_x10_8192.pt")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/output/xeron-1.0-short")
DCP_DIR = os.environ.get("DCP_DIR", "/kaggle/working/dcp")
SHARD_INDEX = _i("SHARD_INDEX", 0)
SHARD_COUNT = _i("SHARD_COUNT", 4)
CKPT_EVERY = _i("CKPT_EVERY", 1)
MAX_ITEMS = _i("MAX_ITEMS", 0)                # 0 = 샤드 전체
# 0 = 무제한. >0 이면 이 분(min)을 넘긴 **스텝 경계**에서 루프를 빠져나와 정상 종료한다
# (DCP 저장 + 스냅샷 + TRAIN_DONE 까지 수행). 커널이 같은 값을 하드 타임아웃으로 걸면
# 저장 도중 SIGKILL 되어 커널이 ERROR 로 끝난다(v6 스모크).
MAX_TRAIN_MIN = _i("MAX_TRAIN_MIN", 0)
RESUME_SHARD = _i("RESUME_SHARD", 1) == 1
LEN_MIN = _i("LEN_MIN", 0)
LEN_MAX = _i("LEN_MAX", 1024)                 # 이 트레이너는 ≤1024(숏) 전용
# XLA는 **텐서 shape 마다** 그래프를 새로 컴파일한다. 길이순 정렬 데이터는 micro-batch 마다
# 패딩 길이(=batch 내 최장)가 달라져서 거의 모든 micro-batch 가 새 그래프가 된다 —
# v8 스모크: 128 micro 에 2154s(1.9 seq/s, ~10-60 s/micro)로 실측 peak(8칩 ~8.5K tok/s)의
# 1/90 수준. shape 를 성글게 고정해 컴파일 횟수를 한 자릿수로 줄인다.
PAD_L_BUCKET = _i("PAD_L_BUCKET", 16)         # 토큰 축을 이 배수로 올림 패딩 (0=비활성)
MARKER_PAD = _i("MARKER_PAD", 16)             # 결정 헤드 폭 고정(실측 최대 마커 14)


# ---------------------------------------------------------------- data
def collate_train_batch(items, pad_id, l_bucket=0, k_pad=0):
    """train_ddp.py 와 동일. 단 모든 텐서를 rank-2 로 유지한다(SPMD 입력 샤딩 제약).

    l_bucket/k_pad 는 XLA 재컴파일을 줄이기 위한 shape 고정용이다(손실에는 영향 없음).
    """
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    if l_bucket:
        L = math.ceil(L / l_bucket) * l_bucket     # 데이터가 ≤LEN_MAX 로 필터되어 있다
    if k_pad:
        kmax = max(k_pad, kmax)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    qt = torch.tensor([[it["qtype"]] for it in items], dtype=torch.long)   # (n,1) rank-2
    return ids, att, mpos, mmask, target, qt


def load_shard(path, shard_index, shard_count, len_min, len_max, max_items=0):
    """긴 항목을 빼고 길이 오름차순 정렬한 뒤 **등토큰(등시간) 샤드**로 자른다."""
    items = torch.load(path, weights_only=False)
    print(f"[data] loaded {len(items):,} items from {path}", flush=True)
    sel = [it for it in items if len_min < len(it["ids"]) <= len_max]
    del items
    sel.sort(key=lambda it: len(it["ids"]))
    print(f"[data] {len_min}<len<={len_max}: {len(sel):,} items "
          f"({sum(len(it['ids']) for it in sel):,} tokens)", flush=True)
    # 등토큰 그리디 분배: 길이순으로 훑으며 가장 가벼운 샤드에 넣는다 → 샤드별 학습시간 균등
    loads = [0] * shard_count
    buckets = [[] for _ in range(shard_count)]
    for it in sel:
        j = min(range(shard_count), key=lambda k: loads[k])
        buckets[j].append(it)
        loads[j] += len(it["ids"])
    del sel
    mine = buckets[shard_index]
    print(f"[data] shard {shard_index}/{shard_count}: {len(mine):,} items "
          f"({loads[shard_index]:,} tokens; all={ [f'{l/1e6:.1f}M' for l in loads] })", flush=True)
    if max_items:
        mine = mine[:max_items]
        print(f"[data] truncated to {len(mine):,} items (MAX_ITEMS)", flush=True)
    return mine


# ---------------------------------------------------------------- model
def build_model():
    from transformers import AutoTokenizer
    from ctx_extend import ensure_long_context
    from model_xeron import build_wide_model
    from safetensors.torch import load_file

    cfg = ensure_long_context(BASE_DIR, MAX_LEN, CTX_CAP, HEAD_MAX_LEN)
    cfg["head_layers"] = HEAD_LAYERS
    cfg["dropout"] = HEAD_DROPOUT
    cfg["gradient_checkpointing"] = False   # XLA에서는 별도 래퍼로 처리(아래)
    cfg["max_tokens_per_batch"] = MAX_TOKENS_BATCH
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN

    tok = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "tokenizer"))
    model = build_wide_model(cfg, os.path.join(BASE_DIR, "encoder"),
                             head_layers=HEAD_LAYERS, head_size=HEAD_SIZE,
                             dropout=HEAD_DROPOUT)
    weights = load_file(os.path.join(BASE_DIR, "model.safetensors"))
    try:
        model.load_state_dict(weights, strict=True)
        print("[XERON] loaded all weights", flush=True)
    except RuntimeError as e:
        enc_only = {k: v for k, v in weights.items() if k.startswith("encoder.")}
        missing, _ = model.load_state_dict(enc_only, strict=False)
        print(f"[XERON] head rebuilt; encoder loaded, {len(missing)} head tensors fresh "
              f"({str(e).splitlines()[0][:100]})", flush=True)
    return model, tok, cfg


def main():
    t_start = time.time()
    xr.use_spmd()
    ndev = xr.global_runtime_device_count()
    dev = xm.xla_device()
    mesh = xs.Mesh(np.arange(ndev), (ndev, 1), ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    print(f"[spmd] devices={ndev} mesh=(fsdp,model) pjrt={os.environ.get('PJRT_DEVICE')}",
          flush=True)

    model, tok, cfg = build_model()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[XERON] params {nparam/1e6:.1f}M  max_len={MAX_LEN} head={HEAD_LAYERS}x{HEAD_SIZE}",
          flush=True)

    # 1) bf16 먼저 (FSDPv2 래핑 후 캐스팅하면 샤딩이 깨진다)
    model = model.to(torch.bfloat16)

    # 2) FSDPv2 샤딩 (mesh 축 'fsdp')
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2)
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.modernbert import modeling_modernbert as mbm
    layer_cls = getattr(mbm, "ModernBertEncoderLayer", None) or getattr(mbm, "ModernBertLayer")

    def shard_output(out, mesh):
        """Shard every model output along the FSDP axis, matching the tensor rank.

        The decision head returns rank-2 tensors (logits [B, T], act_logits [B, A]),
        so a fixed 3-entry partition spec asserts out (rank 3 != rank 2).
        Build the spec from `dim()` instead so it stays correct either way.
        """
        def _shard(t):
            if not torch.is_tensor(t):
                return
            xs.mark_sharding(t, mesh, ("fsdp",) + (None,) * (t.dim() - 1))

        if isinstance(out, (tuple, list)):
            for t in out:
                _shard(t)
        else:
            _shard(out)

    auto_wrap = functools.partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={layer_cls})
    model = FSDPv2(model, mesh=mesh, auto_wrap_policy=auto_wrap, shard_output=shard_output)
    model.train()

    # 3) 데이터 (길이 버킷 → 등토큰 샤드)
    items = load_shard(ITEMS, SHARD_INDEX, SHARD_COUNT, LEN_MIN, LEN_MAX, MAX_ITEMS)

    class BatchDS(torch.utils.data.Dataset):
        """Yields one pre-planned micro-batch per index (length-bucketed, budgeted)."""

        def __init__(self, plan, rows):
            self.plan = plan
            self.rows = rows

        def __len__(self):
            return len(self.plan)

        def __getitem__(self, i):
            return collate_train_batch([self.rows[j] for j in self.plan[i]], tok.pad_token_id,
                                       PAD_L_BUCKET, MARKER_PAD)

    def collate_tuple(rows):
        # rows: list[(ids,att,mpos,mmask,target,qt)] each of shape (1, ...) → cat
        return tuple(torch.cat([r[j] for r in rows], dim=0) for j in range(6))

    row_lengths = [len(it["ids"]) for it in items]

    def make_loader(epoch: int, device):
        plan, st = plan_batches_epoch(row_lengths, MICRO_BATCH, MAX_TOKENS_BATCH,
                                      BATCH_MODE, BATCH_BUCKET_MULT, seed=1000 + epoch)
        dl = torch.utils.data.DataLoader(
            BatchDS(plan, items), batch_size=1, shuffle=False, num_workers=4,
            collate_fn=collate_tuple)
        return pl.MpDeviceLoader(
            dl, device, input_sharding=xs.ShardingSpec(mesh, ("fsdp", None))), st, plan

    dev_loader, plan_stats, plan = make_loader(0, dev)
    print(f"[batch] mode={BATCH_MODE} micro_batch(global)={MICRO_BATCH} "
          f"max_tokens_batch={MAX_TOKENS_BATCH} items={plan_stats['items']:,} "
          f"batches={plan_stats['batches']:,} real_tok={plan_stats['real_tokens']:,} "
          f"padded_tok={plan_stats['padded_tokens']:,} waste={plan_stats['waste_pct']}% "
          f"max_batch_padded={plan_stats['max_padded_per_batch']:,}", flush=True)
    # XLA 컴파일 횟수 = 서로 다른 (토큰 패딩, 마커 패딩) shape 의 개수.
    def _shapes_report(plan):
        if PAD_L_BUCKET or MARKER_PAD:
            l_b = {max(row_lengths[j] for j in b) for b in plan}
            if PAD_L_BUCKET:
                l_b = {math.ceil(m / PAD_L_BUCKET) * PAD_L_BUCKET for m in l_b}
            return f"[shapes] distinct padded token widths={sorted(l_b)[:20]}{'...' if len(l_b) > 20 else ''} " \
                   f"(count={len(l_b)}) marker_pad={MARKER_PAD}\n"
        return f"[shapes] dynamic padding, distinct batch widths={len({max(row_lengths[j] for j in b) for b in plan})}\n"

    print(_shapes_report(plan)[:-1], flush=True)

    # steps_per_epoch must use the planner's actual batch count, not
    # len(items)//(MICRO_BATCH*GRAD_ACCUM).  When MAX_TOKENS_BATCH caps a batch below
    # MICRO_BATCH the planner emits more (smaller) batches, so the naive formula
    # underestimates steps_per_epoch → T_max too small → LR decays too fast.
    steps_per_epoch = max(1, plan_stats["batches"] // GRAD_ACCUM)
    print(f"[data] {len(items):,} items | micro_batch(global)={MICRO_BATCH} "
          f"grad_accum={GRAD_ACCUM} | batches/epoch={plan_stats['batches']:,} "
          f"steps/epoch={steps_per_epoch}", flush=True)

    # 4) 옵티마이저: encoder/head LR 분리 (train_ddp 와 동일)
    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    print(f"[opt] encoder tensors={len(enc_params)} head tensors={len(head_params)}", flush=True)
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER},
         {"params": head_params, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
    total_updates = steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_updates, eta_min=1e-6)

    # 5) DCP 복원
    start_epoch = 0
    if RESUME_SHARD and os.path.isdir(DCP_DIR) and os.listdir(DCP_DIR):
        try:
            import torch.distributed.checkpoint as dist_cp
            import torch_xla.experimental.distributed_checkpoint as xc
            ckpt = os.path.join(DCP_DIR, f"shard{SHARD_INDEX}")
            if os.path.isdir(ckpt) and os.listdir(ckpt):
                sd = {"model": model.state_dict(), "optim": opt.state_dict()}
                dist_cp.load(state_dict=sd, storage_reader=dist_cp.FileSystemReader(ckpt),
                             planner=xc.SPMDLoadPlanner())
                model.load_state_dict(sd["model"])
                opt.load_state_dict(sd["optim"])
                meta = os.path.join(ckpt, "meta.json")
                if os.path.exists(meta):
                    start_epoch = json.load(open(meta)).get("epoch_done", 0)
                print(f"[dcp] resumed shard{SHARD_INDEX} at epoch {start_epoch}", flush=True)
        except Exception as ex:
            print(f"[dcp] resume skipped: {type(ex).__name__}: {str(ex)[:200]}", flush=True)

    from laya.common import proper_reward

    def save_dcp(epoch_done):
        try:
            import torch.distributed.checkpoint as dist_cp
            import torch_xla.experimental.distributed_checkpoint as xc
            ckpt = os.path.join(DCP_DIR, f"shard{SHARD_INDEX}")
            os.makedirs(ckpt, exist_ok=True)
            t0 = time.time()
            dist_cp.save(state_dict={"model": model.state_dict(), "optim": opt.state_dict()},
                         storage_writer=dist_cp.FileSystemWriter(ckpt),
                         planner=xc.SPMDSavePlanner())
            json.dump({"epoch_done": epoch_done, "shard": SHARD_INDEX,
                       "shard_count": SHARD_COUNT, "items": len(items),
                       "max_len": MAX_LEN, "saved_s": round(time.time() - t0, 1)},
                      open(os.path.join(ckpt, "meta.json"), "w"))
            print(f"[dcp] saved {ckpt} in {time.time()-t0:.1f}s", flush=True)
        except Exception as ex:
            print(f"[dcp] save FAILED: {type(ex).__name__}: {str(ex)[:200]}", flush=True)

    def snapshot(tag=""):
        """추론용 스냅샷 (HF 포맷) — 평가/HF 업로드용."""
        from safetensors.torch import save_file
        os.makedirs(OUT_DIR, exist_ok=True)
        sd = {k: v.to(torch.float32).cpu() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(OUT_DIR, "model.safetensors"))
        cfg_out = dict(cfg)
        json.dump(cfg_out, open(os.path.join(OUT_DIR, "rl_agent_config.json"), "w"), indent=2)
        for sub in ("encoder", "tokenizer"):
            src = os.path.join(BASE_DIR, sub)
            if os.path.isdir(src):
                os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
                for fn in os.listdir(src):
                    with open(os.path.join(src, fn), "rb") as fi, \
                         open(os.path.join(OUT_DIR, sub, fn), "wb") as fo:
                        fo.write(fi.read())
        print(f"[snapshot] {OUT_DIR} {tag}", flush=True)

    # 6) 학습 루프 (train_ddp 의 RLCD 손실을 그대로 이식)
    # 진행 감시 스레드: XLA 컴파일/콜렉티브에서 멈추면 stdout 이 완전히 끊겨
    # "멈춘 것"과 "느린 컴파일"을 구분할 수 없다(v5 스모크: 스텝 로그 0줄로 40분 타임아웃).
    # 2분마다 현재 위치를 남겨 어느 단계에 물려 있는지 알 수 있게 한다.
    progress = {"phase": "loop-start", "micro": -1, "nstep": 0}

    def _watch():
        t0 = time.time()
        while True:
            time.sleep(120)
            print(f"[watch] t={time.time()-t0:.0f}s phase={progress['phase']} "
                  f"micro={progress['micro']} nstep={progress['nstep']}", flush=True)

    import threading
    threading.Thread(target=_watch, daemon=True).start()

    rng = random.Random(42 + SHARD_INDEX)
    budget_stop = False
    for epoch in range(start_epoch, EPOCHS):
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, EPOCHS - 1))
        opt.zero_grad(set_to_none=True)
        accum, nstep, t0 = 0, 0, time.time()
        losses = []
        tok_seen = 0
        # rebuild the plan each epoch so bucket order/content is reshuffled
        if epoch > start_epoch:
            dev_loader, epoch_stats, plan = make_loader(epoch, dev)
            print(f"[batch] epoch {epoch+1} plan: batches={epoch_stats['batches']:,} "
                  f"waste={epoch_stats['waste_pct']}%", flush=True)
        for micro, batch in enumerate(dev_loader):
            ids, att, mpos, mmask, target, qt = batch
            tok_seen += ids.numel()          # 패딩 포함 토큰(장치가 실제로 계산하는 양)
            t_micro = time.time()
            progress.update(phase="fwd", micro=micro)
            with torch.amp.autocast("cuda", enabled=False):
                logits, act = model(ids, att, mpos, mmask, qt.squeeze(-1))
            logits = logits.float()
            # logits 은 rank-2 [B, T] (결정 헤드) 이므로 마스크도 rank-2 로 유지한다.
            # unsqueeze(-1) 하면 eps (G,B,T) * mask (B,T,1) 에서 T vs B 로 밀려
            # XLA mul 이 "Check failed: dim1 == dim2 ..." 로 죽는다(v3 스모크).
            mask = mmask
            k = mask.sum(-1, keepdim=True).float()

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=logits.device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qt.squeeze(-1),
                                  mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            # logp·adv 는 마커 축으로 이미 축약되어 (G,B) 다. 결정 헤드 폭을 MARKER_PAD 로
            # 넓혀도 패딩 위치는 mask=0 이라 logp 에 0 을 더할 뿐이라 손실은 동일하다.
            # (로컬 CPU 검증: 자연폭(5) vs 패딩(16) rl/ce 차이 < 1e-6)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target *
                        torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (RL_WEIGHT * loss_rl + 1.0 * loss_ce) / GRAD_ACCUM
            progress["phase"] = "bwd"
            loss.backward()
            accum += 1

            if accum % GRAD_ACCUM == 0:
                progress["phase"] = "opt"
                # 근사 클리핑: 로컬 샤드 노름 → 전역 스케일(√ndev)로 보정
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    1.0 * math.sqrt(ndev))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                nstep += 1
                progress.update(phase="step-done", nstep=nstep)
                if nstep <= 5 or nstep % 20 == 0:
                    el = time.time() - t0
                    print(f"  ep{epoch+1}/{EPOCHS} step {nstep}/{total_updates//EPOCHS} "
                          f"loss={loss.item()*GRAD_ACCUM:.4f} reward={r.mean().item():.3f} "
                          f"gn={float(gn):.2f} lr={sched.get_last_lr()[0]:.2e} "
                          f"{el/max(1,nstep):.2f}s/step "
                          f"({MICRO_BATCH*GRAD_ACCUM/max(1e-9, el/max(1,nstep)):.1f} seq/s)",
                          flush=True)
            # XLA 는 마지막 mark_step 이후의 연산을 **하나의 그래프**로 묶는다. 마이크로배치마다
            # 끊지 않으면 GRAD_ACCUM(8)회 fwd+bwd 가 한 HLO 로 전개되어 컴파일이 수십 분 걸리거나
            # 끝나지 않는다(v5 스모크: 스텝 로그 0줄 상태로 40분 타임아웃). 1회 fwd+bwd 로 제한한다.
            xm.mark_step()
            if micro < 3:
                print(f"  [micro {micro}] {time.time()-t_micro:.1f}s "
                      f"loss={float(loss)*GRAD_ACCUM:.4f} accum={accum}", flush=True)
            # shape 별 처리량을 직접 볼 수 있게 주기적으로 속도를 남긴다
            # (스모크에서 micro 당 고정비용이 지배적인지를 판단하는 근거).
            if (micro + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  [rate] micro={micro+1} {el/(micro+1):.2f}s/micro "
                      f"{tok_seen/max(1e-9, el):.0f} tok/s shape={tuple(int(s) for s in ids.shape)}",
                      flush=True)
            losses.append(loss.item() * GRAD_ACCUM)
            if MAX_TRAIN_MIN and (time.time() - t_start) > MAX_TRAIN_MIN * 60:
                print(f"[budget] MAX_TRAIN_MIN={MAX_TRAIN_MIN} 도달 — micro {micro+1} "
                      f"(step {nstep}) 에서 정상 종료 → DCP 저장", flush=True)
                budget_stop = True
                break

        # Flush trailing partial grad-accum group on normal epoch completion (not budget stop).
        # Each micro-loss was divided by GRAD_ACCUM, so accumulated grads carry a 1/GRAD_ACCUM
        # factor.  Rescale by GRAD_ACCUM/partial so the update is a proper mean over the
        # partial group, matching the semantics of a full GRAD_ACCUM step.
        partial = accum % GRAD_ACCUM
        if not budget_stop and partial != 0:
            scale = GRAD_ACCUM / partial
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            progress["phase"] = "opt-trailing"
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * math.sqrt(ndev))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            nstep += 1
            xm.mark_step()
            print(f"  [trailing] flushed {partial}/{GRAD_ACCUM} micro-batches as step {nstep} "
                  f"gn={float(gn):.2f}", flush=True)

        dt = time.time() - t0
        print(f"=== epoch {epoch+1}/{EPOCHS} done {dt/60:.1f}min | "
              f"avg loss {sum(losses)/max(1,len(losses)):.4f} | "
              f"{(len(items)*EPOCHS)/max(1e-9, dt):.1f} seq/s ===", flush=True)
        if CKPT_EVERY and (epoch + 1) % CKPT_EVERY == 0:
            save_dcp(epoch + 1)
            snapshot(f"epoch{epoch+1}")
        if budget_stop:
            break

    # epoch 단위 저장이 마지막 epoch 를 이미 덮었는데(EPOCHS % CKPT_EVERY == 0) final 저장을
    # 한 번 더 하면 DCP 저장(~700s, v8 실측 694.9s)이 중복되어 커널 하드 타임아웃에 잘린다.
    if not budget_stop and (CKPT_EVERY <= 0 or EPOCHS % CKPT_EVERY != 0):
        save_dcp(EPOCHS)
        snapshot("final")
    # TRAIN_DONE must always print: kaggle/kernel/kernel.py gates HF upload on this line.
    print(f"TRAIN_DONE shard={SHARD_INDEX}/{SHARD_COUNT} total={(time.time()-t_start)/60:.1f}min",
          flush=True)


if __name__ == "__main__":
    main()

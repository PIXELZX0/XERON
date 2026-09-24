#!/usr/bin/env python3
"""DDP fine-tuning of convaiinnovations/laya with RLCD (policy gradient + proper scoring rules).

Usage:
    torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py \
        convaiinnovations/laya ./output/xeron ./train_items.pt

Hyperparameters can be overridden via environment variables:
    EPOCHS, MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE, LR_ENCODER, LR_HEAD, SIGMA_START, SIGMA_END
"""
import json
import math
import os
import random
import sys
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward, QTYPES
from ctx_extend import ensure_long_context


# ---------------------------------------------------------------- hyperparams
def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


EPOCHS = _env_int("EPOCHS", 4)
MICRO_BATCH = _env_int("MICRO_BATCH", 8)      # sequences per fwd pass per GPU
GRAD_ACCUM = _env_int("GRAD_ACCUM", 4)        # effective batch = MICRO_BATCH * GPUs * GRAD_ACCUM
GROUP_SIZE = _env_int("GROUP_SIZE", 4)        # GRPO baseline samples
LR_ENCODER = _env_float("LR_ENCODER", 2.5e-5)
LR_HEAD = _env_float("LR_HEAD", 1.0e-4)
SIGMA_START = _env_float("SIGMA_START", 0.4)  # exploration noise
SIGMA_END = _env_float("SIGMA_END", 0.1)
RL_WEIGHT = _env_float("RL_WEIGHT", 1.0)      # policy-gradient term weight (0 = pure CE/SFT)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.01)
HEAD_LAYERS = _env_int("HEAD_LAYERS", 0)      # 0 = keep the base model's head depth; >0 rebuilds the head
HEAD_DROPOUT = _env_float("HEAD_DROPOUT", 0.0)  # 0 = keep base value
CHECKPOINT_EVERY = _env_int("CHECKPOINT_EVERY", 0)  # save ckpt every N epochs (0=off, Colab: 1)
RESUME = os.environ.get("RESUME", "")              # checkpoint path or "auto" (latest in output_dir)
MAX_LEN = _env_int("MAX_LEN", 2048)                # context: seq length (RoPE up to 32768)
HEAD_MAX_LEN = _env_int("HEAD_MAX_LEN", 256)       # decision-head marker window
MAX_TOKENS_BATCH = _env_int("MAX_TOKENS_BATCH", 4096)  # max tokens per micro-batch (memory bound)
CTX_CAP = _env_int("CTX_CAP", 32768)                # encoder max_position_embeddings ceiling (4096x8)
DTYPE = os.environ.get("DTYPE", "fp16").lower()    # fp16 (T4/V100) | bf16 (A100/H100, native, no scaler)
assert DTYPE in ("fp16", "bf16"), f"DTYPE must be fp16 or bf16, got {DTYPE}"
AMP_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
STORE_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16


def _latest_checkpoint(output_dir):
    import glob
    cks = sorted(glob.glob(os.path.join(output_dir, "checkpoint_epoch*.pt")))
    return cks[-1] if cks else None


def snapshot_model(output_dir, model, tok, cfg, temps=None):
    """Write an inference-ready checkpoint (model + encoder cfg + tokenizer + agent cfg).

    Called after every epoch checkpoint as well as at the end of training, so a lost
    Colab VM never costs more than one epoch: the snapshot can be loaded directly with
    `laya.load(path)` or used as the base of a resumed run.
    """
    os.makedirs(output_dir, exist_ok=True)
    sd = {k: v.to(STORE_DTYPE).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(output_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))
    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
    c = dict(cfg)
    c["fine_tuned"] = True
    c["model_name"] = "XERON"
    if temps is not None:
        c["temperature"] = temps
    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(c, f, indent=2)
    print(f"[XERON] inference-ready snapshot written to {output_dir}")


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch):
    sd = {k: v.to(STORE_DTYPE).contiguous().cpu() for k, v in model.state_dict().items()}
    torch.save({
        "model": sd,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.float()  # ckpt stored in fp16 -> back to fp32 for stable training
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    scaler.load_state_dict(ck["scaler"])
    return ck["epoch"] + 1


# ---------------------------------------------------------------- utilities
def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
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
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


# ---------------------------------------------------------------- main
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model_id = sys.argv[1]
    output_dir = sys.argv[2]
    items_path = sys.argv[3] if len(sys.argv) > 3 else "train_items.pt"

    # model_id is expected to be a LOCAL directory (e.g. from snapshot_download
    # of convaiinnovations/laya, or a previous fine-tune output)
    # ---- context extension: raise encoder cap + config lengths ----
    cfg = ensure_long_context(model_id, MAX_LEN, CTX_CAP, HEAD_MAX_LEN)
    if HEAD_LAYERS:
        cfg["head_layers"] = HEAD_LAYERS
    if HEAD_DROPOUT:
        cfg["dropout"] = HEAD_DROPOUT
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = MAX_TOKENS_BATCH
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN

    tok = AutoTokenizer.from_pretrained(os.path.join(model_id, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_id, "encoder"))

    weights = load_file(os.path.join(model_id, "model.safetensors"))
    try:
        model.load_state_dict(weights, strict=True)
        if rank == 0:
            print(f"[XERON] loaded all weights (head_layers={cfg.get('head_layers')})")
    except RuntimeError as e:
        # Head shape changed (HEAD_LAYERS override): keep the encoder, rebuild the head.
        enc_only = {k: v for k, v in weights.items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(enc_only, strict=False)
        if rank == 0:
            print(f"[XERON] head rebuilt (head_layers={cfg.get('head_layers')}); "
                  f"encoder weights loaded, {len(missing)} head tensors freshly initialized")
            print(f"[XERON] (original mismatch: {str(e).splitlines()[0][:120]})")

    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    all_items = torch.load(items_path, weights_only=False)
    my_items = all_items[rank::world_size]

    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER},
         {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )
    total_updates = (len(my_items) // (MICRO_BATCH * GRAD_ACCUM)) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == "fp16"))

    # ---- checkpoint resume (all ranks load the same file) ----
    start_epoch = 0
    if RESUME:
        ck_path = RESUME if RESUME != "auto" else _latest_checkpoint(output_dir)
        if ck_path and os.path.exists(ck_path):
            start_epoch = load_checkpoint(ck_path, ddp_model.module, optimizer,
                                          scheduler, scaler, device)
            dist.barrier()
            if rank == 0:
                print(f"[XERON] resumed from {ck_path} at epoch {start_epoch}")
        elif rank == 0:
            print(f"[XERON] RESUME={RESUME} but no checkpoint found; training from scratch")

    cfg["training"] = {
        "updates": total_updates,
        "epochs": EPOCHS,
        "world_size": world_size,
        "fine_tuned_from_checkpoint": True,
        "base_model": os.path.basename(os.path.normpath(model_id)),
        "train_sequences": len(all_items),
        "max_len": MAX_LEN,
        "dtype": DTYPE,
        "micro_batch": MICRO_BATCH,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": MICRO_BATCH * world_size * GRAD_ACCUM,
        "group_size": GROUP_SIZE,
        "lr_encoder": LR_ENCODER,
        "lr_head": LR_HEAD,
        "sigma_start": SIGMA_START,
        "sigma_end": SIGMA_END,
        "rl_weight": RL_WEIGHT,
        "weight_decay": WEIGHT_DECAY,
        "head_layers": cfg.get("head_layers"),
        "head_reinitialized": bool(HEAD_LAYERS),
    }

    if rank == 0:
        print(f"XERON DDP training: {len(all_items)} items | {len(my_items)}/rank "
              f"| {EPOCHS} epochs | eff batch {MICRO_BATCH * world_size * GRAD_ACCUM}")
        print(f"  lr_enc={LR_ENCODER} lr_head={LR_HEAD} sigma={SIGMA_START}->{SIGMA_END} "
              f"rl_w={RL_WEIGHT} wd={WEIGHT_DECAY} dtype={DTYPE} head_layers={cfg.get('head_layers')}")
        print("실시간 진행률/손실/남은 시간(ETA)이 표시됩니다...")
    t0 = time.time()
    t_epoch = t0

    for epoch in range(start_epoch, EPOCHS):
        random.seed(42 + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        t_epoch = time.time()  # 에폭별 시간 측정 시작

        progress = epoch / max(1, EPOCHS - 1)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * progress

        # 실시간 진행바 (rank 0만): 진행률 + 손실 + ETA 자동 표시
        pbar = None
        if rank == 0:
            done_pct = (epoch - start_epoch) / max(1, EPOCHS - start_epoch) * 100
            pbar = tqdm(
                total=len(my_items),
                desc=f"Epoch {epoch+1}/{EPOCHS} (전체 {done_pct:.0f}%)",
                unit="seq",
                dynamic_ncols=True,
                leave=True,
                mininterval=1.0,
            )

        for b_idx in range(0, len(my_items), MICRO_BATCH):
            chunk = my_items[b_idx : b_idx + MICRO_BATCH]
            if not chunk:
                continue

            batch = collate_train_batch(chunk, tok.pad_token_id)

            with torch.autocast("cuda", dtype=AMP_DTYPE):
                logits, act = ddp_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )

            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions (zero-mean projection)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Proper scoring reward (w_sph=0.75 for soft target matching)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                  w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient + full soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (RL_WEIGHT * loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1

            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1

            if pbar is not None:
                pbar.set_postfix(loss=f"{loss.item()*GRAD_ACCUM:.4f}",
                                 reward=f"{r.mean().item():.3f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.1e}")
                pbar.update(len(chunk))

            if rank == 0 and (n_batches % 100) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {n_batches} | Loss: "
                      f"{loss.item()*GRAD_ACCUM:.4f} | Reward: {r.mean().item():.3f} | LR: {cur_lr:.2e}")

        if rank == 0:
            if pbar is not None:
                pbar.close()
            epoch_secs = time.time() - t_epoch
            done_epochs = epoch + 1 - start_epoch
            todo_epochs = EPOCHS - start_epoch
            eta_min = epoch_secs * max(0, todo_epochs - done_epochs) / 60.0
            print(f"=== Epoch {epoch+1}/{EPOCHS} done in {epoch_secs:.1f}s | "
                  f"Avg Loss: {epoch_loss/max(1, n_batches):.4f} | "
                  f"예상 남은 시간: 약 {eta_min:.0f}분 (총 경과 {(time.time()-t0)/60:.0f}분) ===")
            if CHECKPOINT_EVERY and (epoch + 1) % CHECKPOINT_EVERY == 0:
                os.makedirs(output_dir, exist_ok=True)
                ck_path = os.path.join(output_dir, f"checkpoint_epoch{epoch}.pt")
                save_checkpoint(ck_path, model, optimizer, scheduler, scaler, epoch)
                print(f"[XERON] checkpoint saved: {ck_path}")
                # inference-ready snapshot too: a lost VM costs at most one epoch
                snapshot_model(output_dir, model, tok, cfg)
        dist.barrier()

    # Post-training temperature calibration on rank 0
    if rank == 0:
        print("\nFitting post-training calibration temperatures...")
        del optimizer, scaler, scheduler
        torch.cuda.empty_cache()
        model.eval()
        calib_items = all_items[::15][:400]
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx : c_idx + 16]
                cb = collate_train_batch(c_chunk, tok.pad_token_id)
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    l_sub, _ = model(
                        cb["input_ids"].to(device),
                        cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device),
                        cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r_i, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r_i, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]
        try:
            for qt in range(3):
                sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
                if sel:
                    fitted_temps[qt] = fit_one_temp(sel)
            print("Fitted calibration temperatures (choice, score, noul):",
                  [round(t, 3) for t in fitted_temps])
        except Exception as e:
            print("Temperature fitting fallback:", e)

        snapshot_model(output_dir, model, tok, cfg, temps=fitted_temps)
        print(f"XERON model saved to {output_dir}!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
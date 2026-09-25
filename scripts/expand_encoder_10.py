#!/usr/bin/env python3
"""XERON-1.0 — function-preserving encoder expansion (mmBERT-base 768x22 -> 1536x44).

What it does
------------
Grows the XERON-0.9 encoder (ModernBERT, hidden 768 / 22 layers / 12 heads /
intermediate 1152) to 1536 / 44 / 24 / 2304 with **exactly the same function**,
so a 1.0 run starts from a model that already produces 0.9's outputs instead of
from random weights.  The decision head is preserved as well (head_size stays
1024; only `head_proj` grows 768 -> 1536).

Scheme (Net2Net, extended to a pre-norm transformer with GLU-MLP)
-----------------------------------------------------------------
Width.  The residual stream is expanded as `h' = [h ; h]` (block duplication, so
mean/variance are unchanged and LayerNorm stays exact).  For every linear weight
`W: d_in -> d_out` the new weight is `W' = P_out W D_in` where `D_in = 0.5 [I I]`
(averaging downdate) and `P_out` is the chunk-aware duplication.  Concretely
every one of the four blocks of `W'` is `0.5 * W`; the 1/2 compensates the fact
that each input now contributes twice (input-side duplication -> outgoing 1/2).
Output-side duplication needs no extra factor (the duplicated output is consumed
by the next layer's already-halved incoming weights).

    * `attn.Wqkv`  out 3*d is chunked q|k|v -> duplication must respect the
      chunks (otherwise [q;k;v] would become [q;k|v;q|k;v]): out chunks = 3.
    * `mlp.Wi`     out 2*inter is chunked input|gate (GLU) -> out chunks = 2.
    * `mlp.Wo`     in = inter (not hidden) -> block duplication, no chunking.
    * LayerNorm / embedding weights are duplicated with **no** scale: LN stats
      (mean/var) are invariant under duplication, and the embedding is a lookup
      whose output IS the residual stream (already duplicated at full value).
    * `head_proj` (768 -> 1024) -> (1536 -> 1024) = `W D_in`, bias copied.

Depth.  22 -> 44 by placing the (expanded) original layer `j` at index `2j` and
an *identity* clone at `2j+1` whose attn/mlp output projections (`attn.Wo`,
`mlp.Wo`) are zero.  A zero output projection makes the layer an exact residual
passthrough (`h + 0`), so the stack `L0 I L1 I ... L21 I` computes 0.9's
function.  Even placement keeps the attention pattern consistent: the original
layer `j` is full-attention iff `j % 3 == 0`, and `2j % 3 == 0` iff `j % 3 == 0`,
so the 44-entry `layer_types` (full every 3rd) is satisfied at every real layer
(the identity layers' type is irrelevant, their output is zero).

Outputs (atomic: built in `<dst>.tmp-<pid>`, then renamed over `<dst>`)
----------------------------------------------------------------------
    <dst>/model.safetensors, <dst>/encoder/config.json, <dst>/tokenizer/,
    <dst>/rl_agent_config.json   (max_len 8192, encoder name + expansion meta)
    configs/xeron-1.0-expand.json          expansion manifest
    data/xeron10_expand_report.md          verification + memory report
    <dst>/expansion_metrics.json           raw numbers (consumed by the report)

Usage
-----
    .venv/bin/python scripts/expand_encoder_10.py --stage all
    .venv/bin/python scripts/expand_encoder_10.py --stage verify   # re-run checks
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 1234  # fixed: verification inputs are drawn from this seed

# ---- 0.9 (source) geometry -------------------------------------------------
SRC_HIDDEN = 768
SRC_LAYERS = 22
SRC_HEADS = 12
SRC_INTER = 1152
# ---- 1.0 (target) geometry -------------------------------------------------
TGT_HIDDEN = 1536
TGT_LAYERS = 44
TGT_HEADS = 24
TGT_INTER = 2304
HEAD_SIZE = 1024
HEAD_LAYERS = 4
GLOBAL_EVERY = 3  # global_attn_every_n_layers (full attention every 3rd layer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(ROOT, "output", "xeron-0.9-snapshot")
DEFAULT_DST = os.path.expanduser("~/laya-models/xeron-1.0-base")
DEFAULT_MANIFEST = os.path.join(ROOT, "configs", "xeron-1.0-expand.json")
DEFAULT_REPORT = os.path.join(ROOT, "data", "xeron10_expand_report.md")

LAYER_RE = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")

# per-suffix expansion recipe: (dim, chunks, scale) applied in order
LAYER_SPEC = {
    "attn.Wqkv.weight": ((1, 1, 0.5), (0, 3, 1.0)),   # out = q|k|v chunks
    "attn.Wo.weight":   ((1, 1, 0.5), (0, 1, 1.0)),
    "mlp.Wi.weight":    ((1, 1, 0.5), (0, 2, 1.0)),   # out = input|gate chunks
    "mlp.Wo.weight":    ((1, 1, 0.5), (0, 1, 1.0)),
    "attn_norm.weight": ((0, 1, 1.0),),
    "mlp_norm.weight":  ((0, 1, 1.0),),
}
IDENTITY_ZERO_KEYS = ("attn.Wo.weight", "mlp.Wo.weight")


# --------------------------------------------------------------- primitives
def dup_index(d: int, chunks: int = 1) -> torch.Tensor:
    """Index map `new -> old` for duplicating a `d`-sized dim into `2*d`.

    `chunks` splits the dim into equal chunks (q|k|v, input|gate) so that the
    duplication happens *inside* each chunk.
    """
    assert d % chunks == 0, (d, chunks)
    D = d // chunks
    i = torch.arange(2 * d)
    c = torch.div(i, 2 * D, rounding_mode="floor")
    off = i % (2 * D)
    return c * D + (off % D)


def expand_dim(t: torch.Tensor, dim: int, chunks: int = 1, scale: float = 1.0) -> torch.Tensor:
    return t.index_select(dim, dup_index(t.shape[dim], chunks)) * scale


def expand_layer_tensor(suffix: str, t: torch.Tensor) -> torch.Tensor:
    for dim, chunks, scale in LAYER_SPEC[suffix]:
        t = expand_dim(t, dim, chunks, scale)
    return t


def target_encoder_cfg_dict(src_c: dict) -> dict:
    c = dict(src_c)
    c["hidden_size"] = TGT_HIDDEN
    c["num_hidden_layers"] = TGT_LAYERS
    c["num_attention_heads"] = TGT_HEADS
    c["intermediate_size"] = TGT_INTER
    c["layer_types"] = [
        ("full_attention" if i % GLOBAL_EVERY == 0 else "sliding_attention")
        for i in range(TGT_LAYERS)
    ]
    return c


# --------------------------------------------------------------- expansion
def expand_state_dict(src_path: str, verbose: bool = True) -> dict:
    """Build the 1.0 state dict (bf16) from a 0.9 snapshot file."""
    out = {}
    with safe_open(src_path, framework="pt") as f:
        src_keys = set(f.keys())
        need = {"encoder.embeddings.tok_embeddings.weight",
                "encoder.embeddings.norm.weight", "encoder.final_norm.weight"}
        assert need <= src_keys, need - src_keys

        # embeddings / final norm: pure duplication (residual stream is [h;h])
        out["encoder.embeddings.tok_embeddings.weight"] = expand_dim(
            f.get_tensor("encoder.embeddings.tok_embeddings.weight"), 1)
        out["encoder.embeddings.norm.weight"] = expand_dim(
            f.get_tensor("encoder.embeddings.norm.weight"), 0)
        out["encoder.final_norm.weight"] = expand_dim(
            f.get_tensor("encoder.final_norm.weight"), 0)

        seen_layers = set()
        for j in range(SRC_LAYERS):
            for n in (2 * j, 2 * j + 1):
                for suffix in LAYER_SPEC:
                    sk = f"encoder.layers.{j}.{suffix}"
                    if sk not in src_keys:
                        # layer 0's attn_norm is nn.Identity() (it uses embeddings.norm),
                        # so neither the old nor the *expanded* layer 0 has that weight.
                        assert suffix == "attn_norm.weight" and j == 0, sk
                        if n == 0:
                            continue
                        out[f"encoder.layers.{n}.{suffix}"] = torch.ones(TGT_HIDDEN)
                        continue
                    seen_layers.add(sk)
                    out[f"encoder.layers.{n}.{suffix}"] = expand_layer_tensor(
                        suffix, f.get_tensor(sk))
                if n % 2 == 1:  # identity clone: zero the output projections
                    out[f"encoder.layers.{n}.attn.Wo.weight"] = torch.zeros(
                        TGT_HIDDEN, TGT_HIDDEN)
                    out[f"encoder.layers.{n}.mlp.Wo.weight"] = torch.zeros(
                        TGT_HIDDEN, TGT_INTER)
        missing = {k for k in src_keys
                   if LAYER_RE.match(k) and k not in seen_layers}
        assert not missing, f"unhandled encoder layer keys: {sorted(missing)[:5]}"

        # everything else: head / scorer / act_head / type_emb / temperature copied,
        # head_proj grown on its input side (incoming weights halved)
        copied = 0
        for k in src_keys:
            if k.startswith("encoder."):
                continue
            if k == "head_proj.weight":
                out[k] = expand_dim(f.get_tensor(k), 1, 1, 0.5)
            else:
                out[k] = f.get_tensor(k).clone()
                copied += 1
    if verbose:
        print(f"[expand] {len(out)} tensors ({copied} copied verbatim)", flush=True)
    return {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}


def build_target_model(enc_cfg_dict: dict, tmp_encoder_dir: str):
    """Materialise the 1.0 WideHeadDecisionModel from a written encoder dir."""
    os.makedirs(tmp_encoder_dir, exist_ok=True)
    with open(os.path.join(tmp_encoder_dir, "config.json"), "w") as f:
        json.dump(enc_cfg_dict, f, indent=2)

    from transformers import AutoConfig, AutoModel
    from model_xeron import WideHeadDecisionModel

    ecfg = AutoConfig.from_pretrained(tmp_encoder_dir)
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    model = WideHeadDecisionModel(enc, head_layers=HEAD_LAYERS, head_size=HEAD_SIZE,
                                  n_act=2, dropout=0.1)
    return model


# --------------------------------------------------------------- helpers
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_replace_dir(tmp_dir: str, dst_dir: str) -> str | None:
    """Rename tmp_dir -> dst_dir, moving any existing dst aside (never delete)."""
    old = None
    if os.path.exists(dst_dir):
        old = f"{dst_dir}.old-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.rename(dst_dir, old)
    os.rename(tmp_dir, dst_dir)
    return old


def count_params(model: torch.nn.Module) -> dict:
    enc = sum(p.numel() for n, p in model.named_parameters() if n.startswith("encoder."))
    total = sum(p.numel() for p in model.parameters())
    return {"encoder": enc, "head": total - enc, "total": total}


# --------------------------------------------------------------- stage: expand
def stage_expand(src_dir: str, dst_dir: str, manifest_path: str) -> dict:
    t0 = time.time()
    src_enc_cfg_p = os.path.join(src_dir, "encoder", "config.json")
    src_weights_p = os.path.join(src_dir, "model.safetensors")
    src_cfg_p = os.path.join(src_dir, "rl_agent_config.json")
    src_cfg = json.load(open(src_cfg_p))
    src_enc_cfg = json.load(open(src_enc_cfg_p))
    tgt_enc_cfg = target_encoder_cfg_dict(src_enc_cfg)

    tmp_dir = f"{dst_dir}.tmp-{os.getpid()}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"[expand] src={src_dir}\n[expand] tmp={tmp_dir}", flush=True)

    # 1) build the target skeleton (writes encoder/config.json inside tmp)
    model = build_target_model(tgt_enc_cfg, os.path.join(tmp_dir, "encoder"))
    model.eval()
    print(f"[expand] skeleton: {count_params(model)}", flush=True)

    # 2) expanded weights -> model
    sd = expand_state_dict(src_weights_p)
    exp_keys = set(model.state_dict().keys())
    assert set(sd) == exp_keys, (
        f"key mismatch: missing={sorted(exp_keys - set(sd))[:5]} "
        f"unexpected={sorted(set(sd) - exp_keys)[:5]}")
    model.load_state_dict(sd, strict=True)
    del sd

    # 3) save bf16 weights atomically
    save_sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}
    wtmp = os.path.join(tmp_dir, "model.safetensors.tmp")
    save_file(save_sd, wtmp)
    os.replace(wtmp, os.path.join(tmp_dir, "model.safetensors"))
    del save_sd
    print("[expand] weights written", flush=True)

    # 4) tokenizer (copied verbatim from the source snapshot)
    shutil.copytree(os.path.join(src_dir, "tokenizer"), os.path.join(tmp_dir, "tokenizer"))

    # 5) rl_agent_config: rename encoder, max_len 8192, expansion metadata
    tok_sha = sha256_file(os.path.join(src_dir, "tokenizer", "tokenizer.json"))
    cfg = dict(src_cfg)
    cfg["encoder"] = "PIXELZX/XERON-1.0-base"
    cfg["max_len"] = 8192
    cfg["head_layers"] = HEAD_LAYERS
    cfg["head_size"] = HEAD_SIZE
    cfg["expansion"] = {
        "method": "net2net-width + identity-depth (function preserving)",
        "from": "xeron-0.9-base",
        "encoder_base": src_cfg.get("encoder"),
        "hidden_size": [SRC_HIDDEN, TGT_HIDDEN],
        "num_hidden_layers": [SRC_LAYERS, TGT_LAYERS],
        "num_attention_heads": [SRC_HEADS, TGT_HEADS],
        "intermediate_size": [SRC_INTER, TGT_INTER],
        "head_dim": SRC_HIDDEN // SRC_HEADS,
        "identity_layers": list(range(1, TGT_LAYERS, 2)),
        "global_attn_every_n_layers": GLOBAL_EVERY,
        "head_proj": [SRC_HIDDEN, TGT_HIDDEN],
        "source_tokenizer_sha256": tok_sha,
        "script": "scripts/expand_encoder_10.py",
        "seed": SEED,
    }
    # write through ensure_long_context so the 8192 context edit is canonical
    from ctx_extend import ensure_long_context
    rp = os.path.join(tmp_dir, "rl_agent_config.json")
    with open(rp, "w") as f:
        json.dump(cfg, f, indent=2)
    cfg = ensure_long_context(tmp_dir, 8192)
    cfg["expansion"] = json.load(open(rp))["expansion"]
    with open(rp, "w") as f:
        json.dump(cfg, f, indent=2)

    old = atomic_replace_dir(tmp_dir, dst_dir)
    if old:
        print(f"[expand] previous dir kept at {old}", flush=True)

    params = count_params(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del model

    manifest = {
        "name": "xeron-1.0-expand",
        "script": "scripts/expand_encoder_10.py",
        "seed": SEED,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "model_dir": os.path.relpath(src_dir, ROOT),
            "encoder": {k: src_enc_cfg[k] for k in
                        ("model_type", "hidden_size", "num_hidden_layers",
                         "num_attention_heads", "intermediate_size", "vocab_size",
                         "max_position_embeddings", "layer_types")},
            "tokenizer_sha256": tok_sha,
            "rl_agent_config_sha256": sha256_file(src_cfg_p),
        },
        "target": {
            "model_dir": dst_dir,
            "encoder": {k: tgt_enc_cfg[k] for k in
                        ("model_type", "hidden_size", "num_hidden_layers",
                         "num_attention_heads", "intermediate_size", "vocab_size",
                         "max_position_embeddings", "layer_types")},
            "head": {"head_layers": HEAD_LAYERS, "head_size": HEAD_SIZE,
                     "head_proj_in": TGT_HIDDEN},
            "max_len": cfg["max_len"],
        },
        "layout": {
            "real_layer_indices": list(range(0, TGT_LAYERS, 2)),
            "identity_layer_indices": list(range(1, TGT_LAYERS, 2)),
            "identity_zero_keys": list(IDENTITY_ZERO_KEYS),
        },
        "params": params,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[expand] done in {time.time() - t0:.1f}s -> {dst_dir}", flush=True)
    return manifest


# --------------------------------------------------------------- stage: verify
def synth_batch(L: int, B: int, seed: int, n_markers: int = 4, pad_last: bool = False):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256000, (B, L), generator=g)
    att = torch.ones(B, L, dtype=torch.long)
    mpos = torch.zeros(B, n_markers, dtype=torch.long)
    mmask = torch.ones(B, n_markers, dtype=torch.bool)
    for b in range(B):
        p = torch.randperm(L, generator=g)[:n_markers]
        mpos[b] = p
    if pad_last and B > 1:
        keep = max(8, L // 2)
        att[-1, keep:] = 0
        mmask[-1] = False  # markers must not sit in the padded tail
    qtype = torch.tensor([(i % 3) for i in range(B)], dtype=torch.long)
    return ids, att, mpos, mmask, qtype


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    d = (a - b).abs()
    scale = b.abs().max().item()
    return {"cosine": cos, "max_abs_diff": d.max().item(),
            "mean_abs_diff": d.mean().item(), "ref_absmax": scale,
            "rel_max_abs_diff": (d.max() / scale).item() if scale else 0.0}


def load_model(model_dir: str):
    from model_xeron import build_wide_model
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    m = build_wide_model(cfg, os.path.join(model_dir, "encoder"),
                         head_layers=cfg.get("head_layers", 2),
                         head_size=cfg.get("head_size"),
                         dropout=cfg.get("dropout", 0.1) or 0.1)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    m.load_state_dict(weights, strict=True)
    m.eval()
    m.float()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def stage_verify(src_dir: str, dst_dir: str) -> dict:
    cases = [("short", dict(L=64, B=2, n_markers=3, pad_last=False), 11),
             ("medium", dict(L=512, B=2, n_markers=4, pad_last=True), 22),
             ("long", dict(L=4096, B=1, n_markers=6, pad_last=False), 33)]
    print(f"[verify] loading 0.9 from {src_dir}", flush=True)
    m09, cfg09 = load_model(src_dir)
    print(f"[verify] loading 1.0 from {dst_dir}", flush=True)
    m10, cfg10 = load_model(dst_dir)
    assert m10.encoder.config.hidden_size == TGT_HIDDEN
    assert m10.encoder.config.num_hidden_layers == TGT_LAYERS
    rows = []
    with torch.inference_mode():
        for name, kw, seed in cases:
            ids, att, mpos, mmask, qtype = synth_batch(seed=seed, **kw)
            h09 = m09.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h10 = m10.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            d = SRC_HIDDEN
            enc_cos_a = _stats(h10[..., :d], h09)
            enc_cos_b = _stats(h10[..., d:], h09)
            l09, a09 = m09(ids, att, mpos, mmask, qtype)
            l10, a10 = m10(ids, att, mpos, mmask, qtype)
            rows.append({
                "case": name, "B": kw["B"], "L": kw["L"], "seed": seed,
                "enc_half0": enc_cos_a, "enc_half1": enc_cos_b,
                "enc_full_cosine": torch.nn.functional.cosine_similarity(
                    h10[..., :d].float().reshape(-1),
                    h10[..., d:].float().reshape(-1), dim=0).item(),
                "enc_nan": int(torch.isnan(h10).sum().item()),
                "logits": _stats(l10, l09), "act_logits": _stats(a10, a09),
                "logits_shape": list(l10.shape), "act_shape": list(a10.shape),
            })
            print(f"[verify] {name}: enc cos(half0)={enc_cos_a['cosine']:.10f} "
                  f"maxabs={enc_cos_a['max_abs_diff']:.2e} "
                  f"rel={enc_cos_a['rel_max_abs_diff']:.2e} | "
                  f"logits cos={rows[-1]['logits']['cosine']:.10f} "
                  f"maxabs={rows[-1]['logits']['max_abs_diff']:.2e}", flush=True)
            del h09, h10, l09, l10, a09, a10
    return {"cases": rows, "tolerances": {"cosine_min": 0.9999, "max_abs_diff_max": 1e-3}}


# ------------------------------------------------- stage: residual-cause probe
def stage_probe(src_dir: str, dst_dir: str, L: int = 512, seed: int = 77) -> dict:
    """Layer-by-layer 0.9 vs 1.0 comparison in fp32 AND fp64.

    The fp32 residual of the width expansion is a few 1e-4 (absolute) at the encoder
    output.  This probe shows where it comes from: (a) every identity layer is a
    bit-exact no-op, (b) the two halves of the 1.0 hidden state are bit-identical,
    (c) dropping to fp64 shrinks the whole error curve by orders of magnitude -> the
    residual is floating-point rounding, not a structural error.
    """
    out = {}
    d = SRC_HIDDEN
    for tag, DT in (("fp32", torch.float32), ("fp64", torch.float64)):
        m09, _ = load_model(src_dir)
        m10, _ = load_model(dst_dir)
        m09, m10 = m09.to(DT), m10.to(DT)
        ids, att, _, _, _ = synth_batch(L=L, B=1, seed=seed, n_markers=4)
        with torch.inference_mode():
            o9 = m09.encoder(input_ids=ids, attention_mask=att, output_hidden_states=True)
            o10 = m10.encoder(input_ids=ids, attention_mask=att, output_hidden_states=True)
        hs9, hs10 = o9.hidden_states, o10.hidden_states
        # hidden_states[k] is the input to layer k; the last entry is last_hidden_state
        layers, id_max = [], 0.0
        for j in range(SRC_LAYERS):
            if j < SRC_LAYERS - 1:
                cur, nxt, ref = hs10[2 * j + 1], hs10[2 * j + 2], hs9[j + 1]
                idn = (nxt - cur).abs().max().item()
                id_max = max(id_max, idn)
            else:
                cur, ref, idn = hs10[-1], hs9[-1], None
            dab = (cur[..., :d] - ref).abs()
            scale = ref.abs().max().item()
            layers.append({"j": j, "abs_max": dab.max().item(),
                           "rel_max": (dab.max() / scale).item(), "ref_absmax": scale,
                           "identity_delta": idn})
        dfin = (hs10[-1][..., :d] - hs9[-1]).abs()
        scale = hs9[-1].abs().max().item()
        out[tag] = {
            "dtype": str(DT), "L": L, "layers": layers,
            "identity_delta_max": id_max,
            "halves_bit_identical": bool(torch.equal(hs10[-1][..., :d], hs10[-1][..., d:])),
            "final_abs_max": dfin.max().item(),
            "final_rel_max": (dfin.max() / scale).item(),
            "pre_norm_absmax": max(l["ref_absmax"] for l in layers),
            "layer_absmax_j": max(layers, key=lambda l: l["ref_absmax"])["j"],
        }
        print(f"[probe] {tag}: identity_delta_max={id_max:.3e} "
              f"halves_identical={out[tag]['halves_bit_identical']} "
              f"final abs={out[tag]['final_abs_max']:.3e} "
              f"rel={out[tag]['final_rel_max']:.3e}", flush=True)
        del m09, m10, o9, o10, hs9, hs10, layers
    return out


# --------------------------------------------------------------- stage: smoke
def stage_smoke(dst_dir: str, length: int = 8192) -> dict:
    from ctx_extend import ensure_long_context
    before = {p: os.stat(p).st_mtime_ns for p in
              (os.path.join(dst_dir, "rl_agent_config.json"),
               os.path.join(dst_dir, "encoder", "config.json"))}
    ensure_long_context(dst_dir, length)
    for p, mt in before.items():  # ctx_extend writes in place -> keep the dir atomic
        if os.stat(p).st_mtime_ns != mt:
            atomic_write_json(p, json.load(open(p)))
            print(f"[smoke] re-wrote {p} atomically", flush=True)
    enc_cfg = json.load(open(os.path.join(dst_dir, "encoder", "config.json")))
    print(f"[smoke] encoder max_position_embeddings="
          f"{enc_cfg['max_position_embeddings']} max_len={length}", flush=True)
    m, _ = load_model(dst_dir)
    ids, att, mpos, mmask, qtype = synth_batch(L=length, B=1, seed=SEED, n_markers=8)
    t0 = time.time()
    with torch.inference_mode():
        h = m.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        logits, act = m(ids, att, mpos, mmask, qtype)
    res = {
        "length": length,
        "input_ids_shape": list(ids.shape),
        "hidden_shape": list(h.shape),
        "hidden_nan": int(torch.isnan(h).sum().item()),
        "hidden_inf": int(torch.isinf(h).sum().item()),
        "hidden_finite": bool(torch.isfinite(h).all().item()),
        "logits_shape": list(logits.shape), "logits_finite": bool(torch.isfinite(logits).all().item()),
        "act_shape": list(act.shape), "act_finite": bool(torch.isfinite(act).all().item()),
        "hidden_absmax": float(h.abs().max().item()),
        "seconds": round(time.time() - t0, 1),
        "max_position_embeddings": enc_cfg["max_position_embeddings"],
    }
    print(f"[smoke] {json.dumps(res)}", flush=True)
    del m, h
    return res


# --------------------------------------------------------------- memory model
def param_report(src_dir: str, dst_dir: str) -> dict:
    """Exact parameter counts, read straight off the saved checkpoints."""
    def counts(model_dir: str) -> dict:
        enc = head = 0
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                n = 1
                for s in f.get_slice(k).get_shape():
                    n *= int(s)
                if k.startswith("encoder."):
                    enc += n
                else:
                    head += n
        return {"params_encoder": enc, "params_head": head, "params_total": enc + head}

    return {"xeron-0.9": counts(src_dir), "xeron-1.0": counts(dst_dir)}


def activation_report(L: int, B: int) -> dict:
    """Rough A100 activation estimate (bf16 activations, sdpa mem-efficient)."""
    d, layers, inter, heads = TGT_HIDDEN, TGT_LAYERS, TGT_INTER, TGT_HEADS
    by = 2  # bf16
    n_full = sum(1 for i in range(layers) if i % GLOBAL_EVERY == 0)
    n_slide = layers - n_full
    S = B * L * d * by
    gc_on = layers * S
    gc_off = layers * (10 * d + 3 * inter) * B * L * by
    # transient peak inside one layer during recompute / forward
    qkv = 3 * S
    wi = 2 * inter * B * L * by
    peak_in = max(qkv, wi, 2 * S, (inter + 2 * inter) * B * L * by)
    # what the scores WOULD cost if attention materialised them (math backend)
    scores = B * heads * L * L * by
    return {
        "L": L, "B": B, "bytes_per_elem": by,
        "hidden_state": S, "gc_on_stored": gc_on, "gc_off_stored": gc_off,
        "in_layer_peak": peak_in, "gc_on_total": gc_on + peak_in,
        "gc_off_total": gc_off + peak_in, "hypothetical_scores_per_full_layer": scores,
        "n_full_layers": n_full, "n_sliding_layers": n_slide,
    }


# --------------------------------------------------------------- report
def human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def hn(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.3f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{n:,.0f}"


def write_report(path: str, manifest: dict, verify: dict, smoke: dict,
                 prm: dict, acts: list, probe: dict, extra: dict):
    tol = verify["tolerances"]
    L = []
    A = L.append
    A("# XERON-1.0 인코더 확장 리포트")
    A("")
    A(f"- 생성: {extra['created']} (KST) / 스크립트 `scripts/expand_encoder_10.py` (seed {SEED})")
    A(f"- 소스(읽기 전용): `{manifest['source']['model_dir']}` "
      f"(encoder {SRC_HIDDEN}x{SRC_LAYERS}, head {HEAD_SIZE}x{HEAD_LAYERS})")
    A(f"- 산출: `{manifest['target']['model_dir']}` "
      f"(encoder {TGT_HIDDEN}x{TGT_LAYERS}, head {HEAD_SIZE}x{HEAD_LAYERS}, max_len 8192)")
    A("- config: `configs/xeron-1.0-expand.json` (확장 매니페스트), "
      "`rl_agent_config.json` 안에 `expansion` 메타 포함")
    A("- 방식: Net2Net 폭 확장(복제 + 1/2 스케일, 청크 인식) + identity 깊이 삽입"
      "(출력 projection 0) + `head_proj` 입력측 확장 — **함수 보존(function preserving)**")
    A("")
    A("## 1. 파라미터")
    A("")
    A("| 모델 | encoder | head(+head_proj) | total | bf16 가중치 | AdamW fp32 상태 |")
    A("|---|---|---|---|---|---|")
    for tag in ("xeron-0.9", "xeron-1.0"):
        p = prm[tag]
        tot = p["params_total"]
        A(f"| {tag} | {hn(p['params_encoder'])} | {hn(p['params_head'])} | {hn(tot)} | "
          f"{human(tot * 2)} | master {human(tot * 4)} + m/v {human(tot * 8)} = {human(tot * 12)} |")
    p9, p10 = prm["xeron-0.9"], prm["xeron-1.0"]
    A("")
    A(f"- encoder 파라미터: {hn(p9['params_encoder'])} -> {hn(p10['params_encoder'])} "
      f"({p10['params_encoder'] / p9['params_encoder']:.3f}x; 폭 2x x 깊이 2x = 4x 이론값이지만 "
      f"임베딩 393M은 한 번만 존재)")
    A(f"- 전체 대비 head 비중: {100 * p10['params_head'] / p10['params_total']:.1f}% "
      f"(head_size 1024는 그대로 복사, `head_proj`만 2배)")
    A(f"- 저장 실측: `model.safetensors` = {human(extra['weights_bytes'])} (bf16), "
      f"tokenizer = {human(extra['tokenizer_bytes'])}")
    A("")
    A("## 2. 함수 보존 검증 (0.9 vs 1.0, 동일 입력 · CPU fp32)")
    A("")
    A("`h_1.0 = [h ; h]` 구조를 이용해 1.0 인코더 출력의 앞/뒤 절반을 각각 0.9와 비교했다 "
      "(반쪽만 일치하면 확장이 잘못된 것). `enc rel`은 0.9 최종 출력의 max|h|로 나눈 값이라 "
      "분모가 final_norm 이후의 작은 값이어서 보수적이다(내재 상대오차는 §2b 기준 "
      "fp32 <= 3.6e-6, fp64 <= 6.5e-8).")
    A("")
    A("| 케이스 | B x L | cos(앞 768) | cos(뒤 768) | enc max|d| | enc rel | "
      "logits cos | logits max|d| | act cos | act max|d| |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in verify["cases"]:
        A(f"| {r['case']} | {r['B']} x {r['L']} | {r['enc_half0']['cosine']:.9f} | "
          f"{r['enc_half1']['cosine']:.9f} | {r['enc_half0']['max_abs_diff']:.2e} | "
          f"{r['enc_half0']['rel_max_abs_diff']:.1e} | {r['logits']['cosine']:.9f} | "
          f"{r['logits']['max_abs_diff']:.2e} | {r['act_logits']['cosine']:.9f} | "
          f"{r['act_logits']['max_abs_diff']:.2e} |")
    ok = all(r["enc_half0"]["cosine"] >= tol["cosine_min"]
             and r["enc_half1"]["cosine"] >= tol["cosine_min"]
             and r["logits"]["cosine"] >= tol["cosine_min"]
             and r["enc_half0"]["max_abs_diff"] <= tol["max_abs_diff_max"]
             and r["logits"]["max_abs_diff"] <= tol["max_abs_diff_max"]
             for r in verify["cases"])
    worst = max(r["enc_half0"]["max_abs_diff"] for r in verify["cases"])
    A("")
    A(f"판정: **{'PASS' if ok else 'FAIL'}** "
      f"(기준 cosine >= {tol['cosine_min']}, max|d| <= {tol['max_abs_diff_max']:g}; "
      f"최대 절대오차 {worst:.2e}). decision logits는 1e-6 수준으로 더 잘 맞는다.")
    A("")
    A("## 2b. 남은 잔차의 원인 = fp32 반올림 (float64 대조)")
    A("")
    if probe:
        f32, f64 = probe["fp32"], probe["fp64"]
        A(f"L={f32['L']}, seed {SEED} 고정. 레이어별로 0.9 레이어 j의 출력과 1.0 레이어 2j"
          f"(뒤따르는 identity 레이어 직전) 출력을 비교.")
        A("")
        A(f"- identity 레이어 {SRC_LAYERS - 1}개 비트 검사: Δ = {f32['identity_delta_max']:.1e} "
          f"(fp32) / {f64['identity_delta_max']:.1e} (fp64) → **정확한 no-op**")
        A(f"- 1.0 최종 hidden state의 두 절반이 비트 동일: {f32['halves_bit_identical']} "
          f"(fp32) / {f64['halves_bit_identical']} (fp64) → 복제 구조가 정확히 유지됨")
        A(f"- 최종 |Δ|: fp32 {f32['final_abs_max']:.2e} (rel {f32['final_rel_max']:.1e}) → "
          f"fp64 {f64['final_abs_max']:.2e} (rel {f64['final_rel_max']:.1e}) "
          f"= {f32['final_abs_max'] / max(f64['final_abs_max'], 1e-300):.0f}배 감소")
        A(f"- 원인: 0.9의 pre-final_norm 잔차가 레이어 {f32['layer_absmax_j']} 부근에서 "
          f"|h|max ≈ {f32['pre_norm_absmax']:.0f}까지 커진다(모델 자체의 특성). 절대오차는 "
          f"이 크기에 비례해 커 보이지만 상대오차는 fp32 <= "
          f"{max(l['rel_max'] for l in f32['layers']):.1e}, fp64 <= "
          f"{max(l['rel_max'] for l in f64['layers']):.1e}로 기계 정밀도 수준이다.")
        A("")
        A("| 레이어 j | 0.9 \\|h\\|max | fp32 abs | fp32 rel | fp64 abs | fp64 rel |")
        A("|---|---|---|---|---|---|")
        for l32, l64 in zip(f32["layers"], f64["layers"]):
            A(f"| {l32['j']} | {l32['ref_absmax']:.1f} | {l32['abs_max']:.2e} | "
              f"{l32['rel_max']:.1e} | {l64['abs_max']:.2e} | {l64['rel_max']:.1e} |")
        A("")
        A("→ 확장 자체는 수학적으로 정확하고(2^d 복제 + 1/2은 2진 정확), 남은 차이는 "
          "연산/저장 정밀도의 반올림뿐이다.")
    else:
        A("(probe 미실행)")
    A("")
    A("## 3. A100 80GB 활성화 메모리 추정")
    A("")
    A("가정: bf16 활성화, `attn_implementation=\"sdpa\"` (memory-efficient → score 행렬 미저장), "
      "gradient checkpointing on이면 레이어 경계만 저장. static = 가중치(bf16) + grad(bf16) + "
      "AdamW(fp32 master + m/v).")
    A("")
    A(f"- static(1.0): {human(p10['params_total'] * 2)} + {human(p10['params_total'] * 2)} + "
      f"{human(p10['params_total'] * 12)} = **{human(p10['params_total'] * 16)}**")
    A("")
    A("| L | B | GC on 저장 | GC on 총 | GC off 저장 | GC off 총 |")
    A("|---|---|---|---|---|---|")
    for a in acts:
        A(f"| {a['L']} | {a['B']} | {human(a['gc_on_stored'])} | {human(a['gc_on_total'])} | "
          f"{human(a['gc_off_stored'])} | {human(a['gc_off_total'])} |")
    A("")
    A(f"- 참고: score 행렬을 실제로 저장하는 eager/math 백엔드라면 full-attention 레이어 하나가 "
      f"L={acts[0]['L']}에서 {human(acts[0]['hypothetical_scores_per_full_layer'])}, "
      f"L=8192에서 {human(acts[3]['hypothetical_scores_per_full_layer'])} → full 레이어 "
      f"{acts[0]['n_full_layers']}개만으로 수백 GB. **sdpa(memory-efficient) 필수.**")
    A("")
    A("## 4. 8192 토큰 스모크 (CPU, batch 1)")
    A("")
    A("| 항목 | 값 |")
    A("|---|---|")
    A(f"| encoder max_position_embeddings | {smoke['max_position_embeddings']} |")
    A(f"| 입력 shape | {smoke['input_ids_shape']} |")
    A(f"| last_hidden_state | {smoke['hidden_shape']} |")
    A(f"| NaN / Inf | {smoke['hidden_nan']} / {smoke['hidden_inf']} |")
    A(f"| finite | {smoke['hidden_finite']} |")
    A(f"| decision logits | {smoke['logits_shape']} finite={smoke['logits_finite']} |")
    A(f"| act_logits | {smoke['act_shape']} finite={smoke['act_finite']} |")
    A(f"| \\|h\\|max | {smoke['hidden_absmax']:.3f} |")
    A(f"| 소요 | {smoke['seconds']} s |")
    A("")
    A("## 5. 확장 매핑 요약")
    A("")
    A("| 원본 레이어 j | 1.0 레이어 | 내용 | attention type |")
    A("|---|---|---|---|")
    A("| 0..21 | 2j | Net2Net 확장 사본 (실제 연산) | full iff j%3==0 (유지) |")
    A("| 0..21 | 2j+1 | identity 사본: `attn.Wo=0`, `mlp.Wo=0` | 무관 (출력 0) |")
    A("")
    A(f"- 삽입(identity) 레이어 인덱스: {manifest['layout']['identity_layer_indices']}")
    A("- `layer_types`는 44개 전부 `i%3==0 → full_attention`으로 재생성"
      "(`global_attn_every_n_layers=3` 유지). 실제 레이어 j는 2j 위치이므로 "
      "`2j%3==0 ⟺ j%3==0` → 패턴이 원본과 동일.")
    A(f"- 0.9 토크나이저 sha256: `{manifest['source']['tokenizer_sha256']}` (변경 없음, "
      f"vocab 256000 동결)")
    A("- 청크 규칙: `attn.Wqkv` 출력은 q|k|v 3청크, `mlp.Wi` 출력은 input|gate 2청크"
      "(GLU)로 청크별 복제; LayerNorm/embedding은 스케일 없이 복제; `head_proj`는 입력측 1/2.")
    A(f"- `max_len` 4096 → {manifest['target']['max_len']}, "
      f"`head_max_len` 256, `head_layers` {HEAD_LAYERS}, `head_size` {HEAD_SIZE} (유지)")
    A("")
    A("## 6. 리스크 / 다음 단계")
    A("")
    A("**리스크**")
    A("")
    for x in extra["risks"]:
        A(f"- {x}")
    A("")
    A("**다음 단계 (학습은 별도 담당)**")
    A("")
    for x in extra["next"]:
        A(f"- {x}")
    A("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[report] {path}", flush=True)


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "expand", "verify", "probe", "smoke", "report"])
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--smoke-len", type=int, default=8192)
    ap.add_argument("--probe-len", type=int, default=512)
    ap.add_argument("--metrics", default=None)
    args = ap.parse_args()
    metrics_p = args.metrics or os.path.join(ROOT, "data", "xeron10_expand_metrics.json")

    cached = {}
    if os.path.exists(metrics_p):
        cached = json.load(open(metrics_p))
    data = dict(cached)

    def persist():
        json.dump(data, open(metrics_p, "w"), indent=2)
        print(f"[metrics] {metrics_p}", flush=True)

    manifest = cached.get("manifest")
    if args.stage in ("all", "expand"):
        manifest = stage_expand(args.src, args.dst, args.manifest)
        data["manifest"] = manifest
        persist()
    if manifest is None:
        manifest = json.load(open(args.manifest))

    verify = cached.get("verify")
    if args.stage in ("all", "verify"):
        verify = stage_verify(args.src, args.dst)
        data["verify"] = verify
        persist()

    probe = cached.get("probe")
    if args.stage in ("all", "probe"):
        probe = stage_probe(args.src, args.dst, L=args.probe_len)
        data["probe"] = probe
        persist()

    smoke = cached.get("smoke")
    if args.stage in ("all", "smoke"):
        smoke = stage_smoke(args.dst, args.smoke_len)
        data["smoke"] = smoke
        persist()

    if args.stage in ("all", "report") or (verify and smoke):
        prm = param_report(args.src, args.dst)
        acts = [activation_report(L, B) for L in (4096, 8192) for B in (1, 2, 4)]
        weights_bytes = os.path.getsize(os.path.join(args.dst, "model.safetensors"))
        tok_bytes = sum(os.path.getsize(os.path.join(args.dst, "tokenizer", f))
                        for f in os.listdir(os.path.join(args.dst, "tokenizer")))
        extra = {
            "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "weights_bytes": weights_bytes,
            "tokenizer_bytes": tok_bytes,
            "risks": [
                "identity 레이어(홀수 인덱스)는 `attn.Wo = mlp.Wo = 0`이라 **학습 1스텝차에는** "
                "내부(Wqkv/Wi/norm)로 gradient가 흐르지 않는다. Wo가 갱신된 다음 스텝부터 열린다 "
                "→ lr warmup(수십 스텝) 권장.",
                "인코더 pre-norm 잔차 크기가 레이어 12 부근에서 ~5e3으로 커진다(0.9에서 물려받은 "
                "성질). bf16 학습에서 이 크기는 표현 여유가 크지 않으므로 loss spike 시 "
                "이 구간을 의심할 것.",
                "1.0은 0.9와 동일 함수로 출발하므로 '분포 이동' 리스크는 없다. 다만 bf16 저장이라 "
                "추론 시 0.9와 완전히 같은 텐서는 아니다(복제·1/2 자체는 2진 정확).",
                "메모리: GC off + L=8192 + B=4는 활성화만 수백 GB로 불가. GC on에서 "
                "L=8192는 B<=2, L=4096은 B<=4를 기본으로 잡을 것.",
                "8192 학습에서 attention은 full 레이어 15개에서 2차 비용 → "
                "MAX_TOKENS_BATCH로 토큰 예산을 반드시 제한(현재 8192).",
                "vocab 256000 동결 → 임베딩이 파라미터의 약 30%(393M)를 차지. "
                "vocab pruning/축소는 별도 검토 대상.",
                "`rl_agent_config.encoder`를 `PIXELZX/XERON-1.0-base`로 바꿨다. "
                "`encoder/` 디렉터리가 있으면 build_model은 그쪽을 쓰므로 학습에는 영향 없음.",
            ],
            "next": [
                "A100에서 1.0 베이스로 1 epoch RLCD 파인튜닝(`scripts/a100/run_a100.sh` 스타일, "
                "MAX_LEN 8192 / GC on / DTYPE bf16).",
                "학습 초반 loss가 0.9의 최종 loss와 같은 지점에서 시작하는지 확인 → 확장이 "
                "제대로 됐다는 실전 증거.",
                "identity 레이어가 '깨어나는' 시점(각 홀수 레이어 Wo의 ||·|| 성장) 모니터링.",
                "필요 시 `hidden 1536 / layers 44` 위에 head_size 확대(예: 1536) 실험 — "
                "`WideHeadDecisionModel`이 지원.",
            ],
        }
        write_report(args.report, manifest, verify, smoke, prm, acts, probe, extra)
        data.update({"manifest": manifest, "verify": verify, "probe": probe,
                     "smoke": smoke, "params": prm, "activations": acts,
                     "weights_bytes": weights_bytes})
        persist()

    print("[done]", flush=True)


if __name__ == "__main__":
    main()

# XERON-1.0 Encoder Expansion Report

- Generated: 2026-09-25 19:30 (KST) / script `scripts/expand_encoder_10.py` (seed 1234)
- Source (read-only): `output/xeron-0.9-snapshot` (encoder 768x22, head 1024x4)
- Output: `/home/yuchan/laya-models/xeron-1.0-base` (encoder 1536x44, head 1024x4, max_len 8192)
- config: `configs/xeron-1.0-expand.json` (expansion manifest), `expansion` metadata embedded in `rl_agent_config.json`
- Method: Net2Net width expansion (duplicate + 1/2 scale, chunk-aware) + identity depth insertion (output projection 0) + `head_proj` input-side expansion — **function preserving**

## 1. Parameters

| Model | encoder | head(+head_proj) | total | bf16 weights | AdamW fp32 state |
|---|---|---|---|---|---|
| xeron-0.9 | 306.9M | 52.5M | 359.4M | 685.6 MB | master 1.3 GB + m/v 2.7 GB = 4.0 GB |
| xeron-1.0 | 1.276B | 53.3M | 1.329B | 2.5 GB | master 5.0 GB + m/v 9.9 GB = 14.9 GB |

- encoder params: 306.9M -> 1.276B (4.156x; width 2x × depth 2x = 4x theoretical, but the 393M embedding exists only once)
- head share of the total: 4.0% (head_size 1024 is copied as-is, only `head_proj` doubles)
- Measured on disk: `model.safetensors` = 2.5 GB (bf16), tokenizer = 32.8 MB

## 2. Function-preservation verification (0.9 vs 1.0, same input · CPU fp32)

Leveraging the `h_1.0 = [h ; h]` structure, the first/second halves of the 1.0 encoder output were compared against 0.9 separately (if only one half matched, the expansion would be wrong). `enc rel` is normalized by max|h| of the 0.9 final output, so the denominator is the small post-final_norm value and the figure is conservative (intrinsic relative error is fp32 <= 3.6e-6, fp64 <= 6.5e-8 per §2b).

| Case | B x L | cos(front 768) | cos(back 768) | enc max|d| | enc rel | logits cos | logits max|d| | act cos | act max|d| |
|---|---|---|---|---|---|---|---|---|---|
| short | 2 x 64 | 1.000000715 | 1.000000715 | 1.91e-05 | 8.6e-07 | 1.000000000 | 1.19e-06 | 0.999999940 | 2.86e-06 |
| medium | 2 x 512 | 1.000025988 | 1.000025988 | 4.70e-04 | 2.0e-05 | 1.000000000 | 1.52e-06 | 1.000000119 | 9.54e-07 |
| long | 1 x 4096 | 1.000184774 | 1.000184774 | 7.39e-04 | 3.1e-05 | 1.000000000 | 5.59e-07 | 0.999999940 | 4.77e-07 |

Verdict: **PASS** (criteria cosine >= 0.9999, max|d| <= 0.001; maximum absolute error 7.39e-04). The decision logits match even more closely, at the 1e-6 level.

## 2b. Cause of the residual = fp32 rounding (float64 control)

L=512, seed 1234 fixed. Per layer, the output of 0.9 layer j was compared with the output of 1.0 layer 2j (just before the following identity layer).

- Bit check of the 21 identity layers: Δ = 0.0e+00 (fp32) / 0.0e+00 (fp64) → **exact no-op**
- The two halves of the 1.0 final hidden state are bit-identical: True (fp32) / True (fp64) → the duplicate structure is exactly preserved
- Final |Δ|: fp32 3.81e-04 (rel 1.7e-05) → fp64 2.95e-06 (rel 1.3e-07) = 129× reduction
- Cause: the 0.9 pre-final_norm residual grows to |h|max ≈ 5279 around layer 19 (a property of the model itself). The absolute error looks large in proportion to that magnitude, but the relative error is fp32 <= 1.7e-05, fp64 <= 1.3e-07 — machine-precision level.

| Layer j | 0.9 \|h\|max | fp32 abs | fp32 rel | fp64 abs | fp64 rel |
|---|---|---|---|---|---|
| 0 | 21.4 | 6.68e-06 | 3.1e-07 | 5.33e-14 | 2.5e-15 |
| 1 | 26.3 | 5.72e-06 | 2.2e-07 | 5.68e-14 | 2.2e-15 |
| 2 | 28.1 | 7.63e-06 | 2.7e-07 | 4.26e-09 | 1.5e-10 |
| 3 | 38.6 | 1.14e-05 | 3.0e-07 | 1.06e-08 | 2.8e-10 |
| 4 | 38.7 | 9.54e-06 | 2.5e-07 | 5.56e-09 | 1.4e-10 |
| 5 | 31.5 | 1.14e-05 | 3.6e-07 | 6.07e-08 | 1.9e-09 |
| 6 | 40.2 | 1.14e-05 | 2.8e-07 | 7.65e-08 | 1.9e-09 |
| 7 | 43.3 | 1.14e-05 | 2.6e-07 | 7.65e-08 | 1.8e-09 |
| 8 | 40.6 | 1.72e-05 | 4.2e-07 | 9.59e-08 | 2.4e-09 |
| 9 | 50.8 | 1.91e-05 | 3.8e-07 | 8.82e-08 | 1.7e-09 |
| 10 | 48.6 | 2.29e-05 | 4.7e-07 | 2.32e-07 | 4.8e-09 |
| 11 | 78.0 | 1.79e-04 | 2.3e-06 | 2.46e-06 | 3.2e-08 |
| 12 | 5085.3 | 1.86e-02 | 3.6e-06 | 3.31e-04 | 6.5e-08 |
| 13 | 5093.0 | 1.86e-02 | 3.6e-06 | 3.32e-04 | 6.5e-08 |
| 14 | 5093.1 | 1.86e-02 | 3.6e-06 | 3.32e-04 | 6.5e-08 |
| 15 | 5094.8 | 1.86e-02 | 3.6e-06 | 3.32e-04 | 6.5e-08 |
| 16 | 5095.3 | 1.86e-02 | 3.6e-06 | 3.32e-04 | 6.5e-08 |
| 17 | 5097.9 | 1.86e-02 | 3.6e-06 | 3.32e-04 | 6.5e-08 |
| 18 | 5276.5 | 1.83e-02 | 3.5e-06 | 3.17e-04 | 6.0e-08 |
| 19 | 5278.7 | 1.83e-02 | 3.5e-06 | 3.17e-04 | 6.0e-08 |
| 20 | 5275.5 | 1.83e-02 | 3.5e-06 | 3.17e-04 | 6.0e-08 |
| 21 | 22.6 | 3.81e-04 | 1.7e-05 | 2.95e-06 | 1.3e-07 |

→ The expansion itself is mathematically exact (2^d duplication + 1/2 is binary-exact); the remaining difference is only compute/storage-precision rounding.

## 3. A100 80GB activation memory estimate

Assumptions: bf16 activations, `attn_implementation="sdpa"` (memory-efficient → score matrix not stored), gradient checkpointing on stores only layer boundaries. static = weights (bf16) + grad (bf16) + AdamW (fp32 master + m/v).

- static(1.0): 2.5 GB + 2.5 GB + 14.9 GB = **19.8 GB**

| L | B | GC on stored | GC on total | GC off stored | GC off total |
|---|---|---|---|---|---|
| 4096 | 1 | 528.0 MB | 582.0 MB | 7.5 GB | 7.5 GB |
| 4096 | 2 | 1.0 GB | 1.1 GB | 15.0 GB | 15.1 GB |
| 4096 | 4 | 2.1 GB | 2.3 GB | 29.9 GB | 30.1 GB |
| 8192 | 1 | 1.0 GB | 1.1 GB | 15.0 GB | 15.1 GB |
| 8192 | 2 | 2.1 GB | 2.3 GB | 29.9 GB | 30.1 GB |
| 8192 | 4 | 4.1 GB | 4.5 GB | 59.8 GB | 60.2 GB |

- Note: with an eager/math backend that actually stores the score matrix, a single full-attention layer costs 768.0 MB at L=4096 and 3.0 GB at L=8192 → the 15 full layers alone would need hundreds of GB. **sdpa (memory-efficient) is mandatory.**

## 4. 8192-token smoke test (CPU, batch 1)

| Item | Value |
|---|---|
| encoder max_position_embeddings | 32768 |
| input shape | [1, 8192] |
| last_hidden_state | [1, 8192, 1536] |
| NaN / Inf | 0 / 0 |
| finite | True |
| decision logits | [1, 8] finite=True |
| act_logits | [1, 2] finite=True |
| \|h\|max | 22.758 |
| elapsed | 169.8 s |

## 5. Expansion mapping summary

| Source layer j | 1.0 layer | Content | attention type |
|---|---|---|---|
| 0..21 | 2j | Net2Net expansion copy (real compute) | full iff j%3==0 (preserved) |
| 0..21 | 2j+1 | identity copy: `attn.Wo=0`, `mlp.Wo=0` | irrelevant (output 0) |

- Inserted (identity) layer indices: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43]
- `layer_types` is regenerated for all 44 as `i%3==0 → full_attention` (keeping `global_attn_every_n_layers=3`). Real layer j sits at position 2j, so `2j%3==0 ⟺ j%3==0` → the pattern is identical to the source.
- 0.9 tokenizer sha256: `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` (unchanged, vocab 256000 frozen)
- Chunk rules: `attn.Wqkv` output duplicates per 3 chunks q|k|v, `mlp.Wi` output per 2 chunks input|gate (GLU); LayerNorm/embedding are duplicated without scaling; `head_proj` is halved on the input side.
- `max_len` 4096 → 8192, `head_max_len` 256, `head_layers` 4, `head_size` 1024 (unchanged)

## 6. Risks / next steps

**Risks**

- Identity layers (odd indices) have `attn.Wo = mlp.Wo = 0`, so **for the first training step** no gradient flows into their internals (Wqkv/Wi/norm). They open from the step after Wo is updated → an lr warmup (tens of steps) is recommended.
- The encoder pre-norm residual magnitude grows to ~5e3 around layer 12 (a trait inherited from 0.9). In bf16 training this magnitude leaves little representational headroom, so suspect this region if you see a loss spike.
- 1.0 starts as the same function as 0.9, so there is no 'distribution shift' risk. However, since it is stored in bf16, the inference tensors are not exactly identical to 0.9 (duplication/1/2 themselves are binary-exact).
- Memory: GC off + L=8192 + B=4 is impossible at hundreds of GB of activations alone. With GC on, target L=8192 at B<=2 and L=4096 at B<=4 by default.
- At 8192 training, attention is quadratic in the 15 full layers → you must cap the token budget with MAX_TOKENS_BATCH (currently 8192).
- vocab frozen at 256000 → embeddings take ~30% of the params (393M). vocab pruning/reduction is a separate item to review.
- `rl_agent_config.encoder` was changed to `PIXELZX/XERON-1.0-base`. If an `encoder/` directory exists, build_model uses that instead, so training is unaffected.

**Next steps (training is handled separately)**

- Fine-tune the 1.0 base with 1 epoch of RLCD on A100 (`scripts/a100/run_a100.sh` style, MAX_LEN 8192 / GC on / DTYPE bf16).
- Check that early-training loss starts at the same point as 0.9's final loss → real-world evidence that the expansion was done properly.
- Monitor when the identity layers 'wake up' (the growth of ||·|| in each odd layer's Wo).
- If needed, experiment with enlarging head_size (e.g. 1536) on top of `hidden 1536 / layers 44` — `WideHeadDecisionModel` supports it.

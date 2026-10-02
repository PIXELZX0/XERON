# XERON-1.0 숏컨텍스트(≤1024 토큰) — Kaggle TPU v5e-8 4세션 파이프라인

Kaggle TPU(v5e-8, 8칩 × 15.75GiB)에서 **XERON-1.0(1.329B)** 을 SPMD/FSDPv2로 파인튜닝한다.
1024 토큰 초과분(18,802 items)은 v5e에서 아예 실행이 안 되므로(seq≥2048 OOM) **A100에서 먼저**
학습하고(`scripts/a100/vast_direct_train_long.sh`), 그 결과를 이 파이프라인의 베이스로 쓴다.

## 왜 SPMD인가 (DDP는 불가)

| | DDP (복제) | SPMD + FSDPv2 |
|---|---|---|
| 칩당 정적 메모리 | 2.6(bf16 p) + 2.6(g) + 5.2~10.5(AdamW) = **15.7GB → OOM** | ÷8 ≈ **1.3GB** |
| AdamW | 어떤 배치에서도 OOM | **사용 가능** |
| 처리량 (8칩) | seq128 28.0 / 256 21.8 seq/s | **68.75 / 33.0 / 14.5 / 8.0 (128/256/512/1024)** |

가중 처리량 **36.1 seq/s** → 1에폭 **25.9h**. Kaggle 제약(주 20h, 1세션 9h, 동시 1세션)에서
**세션 9h ≈ 1,169,000 seq**.

## 세션 계획

- 학습 대상: `train_items_x10_8192.pt` 중 **len ≤ 1024** (3,371,030 items, 96.8% 가 여기)
- 샤딩: 길이 오름차순으로 정렬한 뒤 **등토큰 그리디 분배**(학습시간 균등) → 샤드 4개
- **4샤드 × 약 1.07M seq × 9h = 1에폭**, 주 20h 한도로 **약 1.8주**
- 세션당 버킷 구성(비례): ≤128 = 543K / ≤256 = 422K / ≤512 = 80K / ≤1024 = 28K

## 사전 준비 (1회)

1. **베이스 모델**: A100 롱런 결과를 HF에 올린다 (`PIXELZX/XERON-1.0-long`, public 또는 token 필요)
2. **HF 토큰**: Kaggle 스크립트 커널은 push 시 env 주입이 안 되므로, 비공개 데이터셋
   `pistonx/xeron-creds` 에 `hf_token.txt` 를 넣고 커널에 마운트한다 (DCP 이어받기/업로드용)

   ⚠️ 마운트 경로가 두 가지다 — `/kaggle/input/<slug>/` (구) 와
   **`/kaggle/input/datasets/<owner>/<slug>/`** (TPU VM 런이 실제로 쓰는 경로).
   v8 스모크는 데이터셋을 붙였는데도 구 경로만 보다가 `HF_TOKEN 없음` 을 찍었다.
   `_load_token()` 은 이제 두 경로를 모두 glob 하고, 읽은 토큰으로 `whoami` 까지 확인한다.
3. 데이터셋 `pistonx/xeron-1-0-train-items-8192` 는 이미 업로드되어 있음(커널 metadata에 등록).

## 실행

```bash
./scripts/kaggle/run_spmd_shard.sh 0            # 0번 샤드 (약 9h)
./scripts/kaggle/run_spmd_shard.sh 0 --smoke    # 스모크: 2만 seq, 40분 제한
# 이후 1, 2, 3번 샤드 순차 실행 (동시 1세션 제약)
kaggle kernels logs -f pistonx/xeron-1-0-short-spmd
```

## 이어받기 / 프리엠션 대응

- 8스텝마다가 아니라 **에폭 단위**로 DCP 저장(`SPMDSavePlanner`) + 추론용 HF 스냅샷
- 체크포인트 경로: `/kaggle/working/dcp/shardN/` → 세션 종료 시 HF `CKPT_HF`(기본
  `PIXELZX/XERON-1.0-short-ckpt`)로 업로드, 다음 실행 때 `shardN/` 만 내려받아 복원
- 세션 중단(9h/프리엠션) 시에도 최소 1에폭 전 상태는 복구됨
- `MAX_TRAIN_MIN` 으로 시간 제한을 걸면 DCP 저장 후 정상 종료 → 시간 초과로 잘리는 것 방지

## 디스크 주의 (`/kaggle/working` 20GB)

베이스 2.6GB + DCP 7.9GB(모델 2.66 + 옵티마이저 bf16 5.24) + 스냅샷 2.6GB ≈ **13.1GB**.
여유가 있지만 `checkpoint_epoch*.pt` 같은 대용량 중간산출물은 절대 만들지 않는다
(`train_spmd_tpu.py` 는 DCP + 스냅샷만 쓴다).

## shape 고정 (XLA 재컴파일 방지) — 필수

XLA는 **텐서 shape 마다** 그래프를 새로 컴파일한다. 길이순 정렬 데이터는 micro-batch 마다
패딩 길이(=batch 내 최장)가 달라져 거의 모든 micro-batch 가 새 그래프가 된다.

- v8 스모크 실측: 128 micro / 2154s = **1.9 seq/s (~10-60 s/micro)** — v5e-8 실측 peak
  (8.5K tok/s)의 **1/90**. 학습 연산이 아니라 컴파일이 시간을 먹은 것.
- 실데이터(shard 0, 26,337 batch)의 서로 다른 패딩 shape = **1,447개** (재컴파일 1,447회).

`PAD_L_BUCKET`(기본 16) + `MARKER_PAD`(기본 16) 로 두 축을 올림 패딩하면:

| PAD_L_BUCKET | MARKER_PAD | distinct shape | 패딩 오버헤드 | shard0 예상(peak 기준) |
|---|---|---|---|---|
| 0 (동적) | 0 | 1,447 | +0.0% | — |
| 16 | 16 | **44** | +4.6% | 4.67h |
| 32 | 16 | 27 | +9.7% | 4.89h |
| 64 | 16 | 15 | +18.8% | — |

손실은 바뀌지 않는다: `logp`·`adv`·reward 는 모두 마커 축으로 축약된 뒤 `(G,B)` 로
평균되므로 패딩 열은 정확히 0을 더한다. 토큰 축 패딩은 `attention_mask`/`src_key_padding_mask`
로 마스킹된다. (로컬 CPU 검증: 자연폭 K=5 vs 패딩 K=16 에서 rl/ce 차이 < 1e-6)

## 검증 상태 및 남은 리스크

- ✅ SPMD 처리량/메모리/입력 샤딩/FSDPv2 순서 함정은 **GCP v5litepod-8에서 실측 검증**
- ✅ DCP 저장/복원(SPMDSavePlanner/SPMDLoadPlanner) 실측 성공 (저장 75s / 복원 2.9s)
- ⚠️ Kaggle v5e-8 에서는 DCP 저장이 **694.9s**(모델 2.66GB + 옵티마이저 bf16 5.24GB) —
  GCP 실측 75s 의 9배. 커널 하드 타임아웃은 `MAX_TRAIN_MIN*60 + 1800s`(저장 30분 여유).
- ⚠️ epoch 단위 저장(`EPOCHS % CKPT_EVERY == 0`)일 때 final 저장을 또 하던 버그 수정
  (`train_spmd_tpu.py`) — 중복 저장 700s 가 예산을 넘겨 TRAIN_DONE 직전에 잘렸다.
- ⚠️ **미검증**: `train_spmd_tpu.py` 의 실제 학습 스텝(RLCD 손실·그래디언트 누적·스케줄러)이
  TPU에서 도는지, 9h 세션 안에 들어오는지 → **반드시 `--smoke` 로 먼저 확인**
- ⚠️ 그래디언트 클리핑은 SPMD에서 전역 노름 합산이 불가해 로컬 샤드 노름 × √ndev 로 근사한다
- ⚠️ `torch_xla.utils.checkpoint` 는 `use_reentrant=True` 만 지원(현재는 ckpt 미사용)

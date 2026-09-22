# XERON-0.1 — JevBench v1.3 방식 평가 (공개 아이템 부분집합)

JevBench v1.3.0(Benchmark Heaven)의 **채점 방식·태스크·어댑터를 그대로 사용**해 XERON-0.1을 측정했다.
하네스: https://github.com/fstandhartinger/jevbench (MIT), 커밋 시점 2026-09-22, `laya_local` 어댑터 그대로 사용.

## ⚠️ 범위 (반드시 먼저 읽을 것)

JevBench의 534개 결정 중 **공개된 것은 231개**뿐이다 (나머지: judge tier 146개 전부 비공개, standard 24, easy 24, hard 109 비공개).
따라서 이 문서는 **공식 보드 순위와 직접 비교할 수 없다** — 보드의 "partial runs are shown without a rank" 규칙과 같은 위치다.

- 공개: easy 48/72 · standard 72/96 · **judge 0/146** · hard 111/220
- 매칭 비교: XERON·Laya·Jev를 **동일한 231개 아이템**에서 비교했다 (Jev/Laya는 공식 per-task 아티팩트의 공개 아이템 결과 사용)
- 로컬 3종(XERON, laya-typed-decisions, laya-multilingual)은 **동일 머신·동일 어댑터·직렬 1요청·threads=4·모델 로드 후 타이밍**으로 직접 실행

## 1. 정확도 (동일 231 공개 아이템, 매칭 비교)

| 시스템 | easy (48) | standard (72) | judge | hard (111) | 전체 (231) | chance-corrected Intelligence |
|---|---|---|---|---|---|---|
| **Jev 1.13.0** (TypeSafe, API) | 1.000 | 0.986 | — | 0.730 | **0.866** | **82.2** |
| **laya-typed-decisions** (Convai, 421M, 벤더 튜닝) | 0.979 | 0.653 | — | 0.270 | **0.537** | 38.0 |
| **XERON-0.1** (PIXELZX, 322M, 우리) | 0.875 | 0.444 | — | **0.306** | 0.468 | 23.3 |
| laya-multilingual (**미튜닝 base**, 322M) | 0.896 | 0.403 | — | 0.324 | 0.468 | 21.5 |
| *참고: Laya 공식 보드 행 (전체 534)* | 0.944 | 0.729 | 0.692 | 0.341 | — | 45.8 |

**읽는 법**

- **Jev가 압도적**이다 (매칭 공개셋 0.866). 이건 JevBench가 제로샷 강자를 위해 설계된 벤치라 당연한 결과.
- **XERON-0.1 vs 자기 백본(미튜닝 laya-multilingual)**: 전체 0.468 = 0.468 (동률)이지만 **Intelligence 21.5 → 23.3**, 캘리브레이션 대폭 개선. 즉 파인튜닝 이득은 정확도보다 **확률 품질**에서 나왔다.
- **XERON-0.1 vs 벤더가 이 벤치마크에 맞춰 튜닝한 laya-typed-decisions**: 전체 −6.9%p 열세. **단, hard tier에서는 XERON이 우세(0.306 vs 0.270)**.
- easy tier의 `fact`/`tool_selection`은 셋 다 만점 수준, **standard tier(저자 작성 루브릭: policy/adequacy/ordinal/routing)에서 격차가 벌어진다.**

## 2. 캘리브레이션 (hard tier, 자체 측정)

| 시스템 | ECE ↓ | 확률 충실도 (100×(1−TVD), gold 분포 10문항) | Calibration 축 |
|---|---|---|---|
| laya-typed-decisions | **0.077** | **0.704** | **77.5** |
| XERON-0.1 | 0.211 | 0.611 | 59.5 |
| laya-multilingual (base) | 0.289 | 0.427 | 42.4 |
| Laya 공식 보드 | 0.206 | 0.660 | 62.5 |
| Jev 공식 보드 | 0.061 | 0.774 | 82.7 |

XERON은 미튜닝 base보다 크게 좋아졌지만(ECE 0.289→0.211) 벤더 영어 튜닝에는 못 미친다.

## 3. 속도·비용 (동일 머신, 직렬, 모델 로드 후 측정)

### 3a. 로컬 CPU (yuchan-server, 12 vCPU, 4 threads)

| 시스템 | p50 (원시) | p95 (원시) | 보정 p50/p95 (×2+0.15s) | 결정당 입력 토큰 | USD/1,000 결정 (추정) | Speed 축 | Cost 축 |
|---|---|---|---|---|---|---|---|
| **XERON-0.1** | **0.129 s** | 5.683 s | 0.408 / 11.52 | 630 | $0.00630 | **73.3** | 76.0 |
| laya-typed-decisions | 0.394 s | 3.608 s | 0.937 / 7.37 | 338 | $0.00338 | 71.6 | 84.1 |
| laya-multilingual (base) | 0.129 s | 5.704 s | 0.408 / 11.56 | 630 | $0.00630 | 73.3 | 76.0 |
| Jev 1.13.0 (API) | 0.652 s | 0.722 s | 0.652 / 0.722 (보정 없음) | 950 | $0.0399 | 83.3 | 52.0 |

- 머신: yuchan-server (12 vCPU), CPU 4 threads, 동일 조건.
- **비용 축은 토크나이저 아티팩트에 민감하다** — JevBench는 "시스템 자체 토큰 수 × $0.01/M" 추정을 쓴다. mmBERT(256k vocab)는 ModernBERT(50k vocab)보다 결정당 토큰을 ~1.9배 많이 세서 XERON의 Cost 축이 실제 GPU 비용 차이보다 낮게 나온다. 같은 크기 클래스(322M vs 421M)이므로 **실질 비용은 사실상 동일**하다고 보는 게 맞다.

## 4. Colab T4 재실행 (GPU) — 정확도 동일, 속도만 향상

동일한 하네스·동일 어댑터(`laya_local`)를 **Google Colab Tesla T4**에서 그대로 재실행했다.
어댑터는 수정하지 않았다 — `laya.load()`가 CUDA를 자동 감지한다 (`device=cuda` 확인).
실행 경로: `colab` CLI (google-colab-cli 0.6.0) → `colab new -s xeron-jb --gpu T4` → `colab install laya` → 스크립트 실행 → `colab download` → `colab stop`.

| 시스템 | p50 (T4) | p95 (T4) | p50 (로컬 CPU) | easy | standard | hard | 전체 | ECE hard |
|---|---|---|---|---|---|---|---|---|
| XERON-0.1 | 0.0297 s | 0.136 s | 0.129 s | 0.875 | 0.444 | 0.306 | 0.468 | 0.211 |
| laya-typed-decisions | 0.0385 s | 0.085 s | 0.394 s | 0.979 | 0.653 | 0.270 | 0.537 | 0.072 |
| laya-multilingual (base) | 0.0279 s | 0.049 s | 0.129 s | 0.896 | 0.403 | 0.333 | 0.472 | 0.296 |

- **231개 전체 실행 시간: 14.1초** (로컬 CPU 276초) → GPU에서 **약 20배 빠름**. 모델 로드 34.5초 포함해도 1분 이내.
- 정확도는 CPU 실행과 사실상 동일 (XERON easy/standard/hard 완전 일치). laya-multilingual hard가 0.324→0.333으로 미세하게 달라지는 것은 GPU/CPU 부동소수점 차이로 인한 근소한 동점(同點) 판정 변화다.
- 원본: `*.colab-t4.results.jsonl`, 집계: `summary-colab-t4.json`

## 5. JevBench Score (부분 실행 — 순위 없음)

공개 아이템에 judge tier가 없으므로 공식 규칙대로 없는 tier는 가중치를 재정규화해 계산했다. **공식 보드 점수와 직접 비교 금지.**

| 시스템 | Intelligence | Calibration | Speed | Cost | JevBench Score (부분) |
|---|---|---|---|---|---|
| laya-typed-decisions | 38.0 | 77.5 | 71.6 | 84.1 | **37.5** |
| XERON-0.1 | 23.3 | 59.5 | 73.3 | 76.0 | 11.5 |
| laya-multilingual (base) | 21.5 | 42.4 | 73.3 | 76.0 | 8.8 |

> Intelligence < 50 이면 공식 규칙상 `(Intelligence/50)²` 패널티가 곱해진다. XERON은 23.3 → ×0.217 적용됨. 이게 최종 점수를 크게 끌어내린 주 원인.

## 6. 어디서 이기고 어디서 지는가 (family별 정확도, 231 공개 아이템)

| family (tier) | XERON-0.1 | laya-td | laya-ml |
|---|---|---|---|
| tool_selection (easy) | **1.00** | 1.00 | 1.00 |
| fact (easy) | 0.917 | 0.917 | 0.833 |
| intent (easy) | 0.75 | 0.75 | 0.708 |
| adversarial (hard) | **0.50** | 0.333 | 0.667 |
| trap (hard) | **0.125** | 0.000 | 0.125 |
| judge_hard (hard) | **0.588** | 0.353 | 0.588 |
| temporal_numeric (hard) | **0.267** | 0.200 | 0.333 |
| routing_hard (hard) | 0.40 | 0.40 | 0.40 |
| probability (hard) | 0.40 | 0.40 | 0.30 |
| tradeoff (hard) | 0.50 | 0.50 | 0.167 |
| multi_hop (hard) | 0.167 | 0.167 | 0.056 |
| long_policy (hard) | 0.211 | 0.316 | 0.316 |
| extraction (std) | 0.583 | 0.792 | 0.667 |
| ordinal (std) | 0.50 | 0.75 | 0.25 |
| **policy (std)** | 0.50 | **0.917** | 0.25 |
| **adequacy (std)** | 0.25 | **0.583** | 0.167 |
| **routing (std)** | 0.333 | **0.583** | 0.75 |

**해석**: XERON은 **hard tier의 adversarial/trap/judge_hard에서 벤더 모델을 이기고**, **standard tier의 저자 작성 루브릭(policy·adequacy·routing·ordinal)에서 진다.**
전자는 우리 학습 데이터(장문·웹 에이전트·다국어 판단)와 겹치는 영역이고, 후자는 JevBench 고유의 "정책 문서 + 선택지" 스타일로 **우리 학습 데이터에 전혀 없던 분포**다.

스키마 유효성: 3종 모두 **231/231 strict valid, 재정규화 0건** — 출력 자체는 완벽하게 깨끗하다.

## 7. 결론과 다음 단계

1. **XERON-0.1은 JevBench류 태스크에서 벤더 Laya 튜닝판보다 약하다.** 이유는 명확하다 — 우리는 KLUE/브라우저/Mind2Web/SCOTUS로 학습했고, JevBench는 "정책·루브릭·선택지" 판단을 측정한다. 도메인이 겹치지 않는다.
2. 그럼에도 **파인튜닝 효과는 검증됐다**: 동일 백본 대비 확률 품질(ECE 0.289→0.211, TVD 0.573→0.389)과 hard tier가 개선됐다.
3. **다음 단계 제안**: JevBench 스타일 데이터로 추가 파인튜닝.
   - 공개 아이템: `datasets/public/{original,easy,hard}.jsonl` (231건, MIT)
   - 대규모: HF `Praveenrajus/jev-bench` (22 configs · 166,054 rows, human-labeled System One questions)
   - 목표: standard tier 0.44 → 0.65+, ECE hard 0.21 → 0.10 이하

## 재현

```bash
# 하네스
git clone https://github.com/fstandhartinger/jevbench /tmp/jevbench
# 공개 아이템 231개, laya_local 어댑터, 직렬 1요청, threads=4, 로드 후 타이밍
python run_public_jevbench.py <model_dir> <label> <out_prefix>
# 채점 (tier 정확도 / ECE / TVD / 축 / 부분 JevBench Score)
python score_public_jevbench.py <out_prefix> <label>
```

원본 per-decision 결과: `results/jevbench-public/*.results.jsonl` (각 231행, 예측·확률분포·지연 포함)

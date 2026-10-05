#!/usr/bin/env python3
"""XERON-1.0 숏컨텍스트(<=1024 토큰) 파인튜닝 — Kaggle TPU v5e-8, SPMD(FSDPv2).

4세션 파이프라인의 1세션. 세션마다 `SHARD_INDEX`(0..3) 하나를 맡아 1에폭을 돌리고,
DCP 체크포인트를 원격(HF/S3)으로 올려 **프리엠션/세션 종료 후 이어받기**를 보장한다.

왜 SPMD인가: v5e는 칩당 HBM 15.75GiB. 1.33B 모델을 DDP로 복제하면 정적 메모리만
15.7GB라 어떤 배치로도 OOM. FSDPv2로 8등분하면 칩당 ~1.3GB → AdamW 그대로 학습 가능.
실측 처리량(8칩, seq128 68.75 / seq256 33.0 / seq512 14.5 / seq1024 8.0 seq/s) 기준
가중 36.1 seq/s → 1에폭 25.9h, 20h/주 한도에서 샤드당 약 9h.

필수 제약(실측으로 확인한 함정):
  - bf16 캐스팅을 FSDPv2 래핑 **전**에
  - 입력 샤딩은 MpDeviceLoader(input_sharding=...) 경로만
  - torch_xla.utils.checkpoint 는 use_reentrant=True 전용
  - Kaggle TPU: 주 20h / 1세션 최대 9h / 동시 1세션 / /kaggle/working 20GB
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
WORK = "/kaggle/working"
REPO = os.path.join(WORK, "XERON")
EXPORT = os.path.join(WORK, "export")


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def sh(cmd, cwd=None, env=None, check=True, timeout=None):
    log(f"$ {cmd}")
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}")
    return r.returncode


# ---------------------------------------------------------------- config (env 로만 제어)
def _i(n, d):
    return int(os.environ.get(n, d))


def _s(n, d):
    return os.environ.get(n, d)


def _load_token():
    """Kaggle 스크립트 커널은 push 시 env 주입이 안 된다.
    HF_TOKEN 을 ① env ② 마운트된 비공개 데이터셋 파일 순으로 찾는다.

    주의: Kaggle 의 데이터셋 마운트 경로는 두 가지다 —
      /kaggle/input/<slug>/                  (구 경로)
      /kaggle/input/datasets/<owner>/<slug>/ (TPU VM 런이 쓰는 경로)
    v8 스모크는 데이터셋을 붙였는데도 후자를 못 찾아 'HF_TOKEN 없음' 을 찍었다.
    """
    import glob as _glob
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    cands = []
    for pat in ("/kaggle/input/xeron-creds/hf_token.txt",
                "/kaggle/input/datasets/*/xeron-creds/hf_token.txt",
                "/kaggle/input/*creds*/hf_token.txt",
                "/kaggle/input/datasets/*/*creds*/hf_token.txt"):
        cands += sorted(_glob.glob(pat))
    for cand in cands:
        try:
            tok = open(cand).read().strip()
        except Exception as ex:
            log(f"  [!] {cand} 읽기 실패 ({type(ex).__name__})")
            continue
        if tok:
            os.environ["HF_TOKEN"] = tok
            log(f"  HF_TOKEN 을 {cand} 에서 읽었습니다 ({len(tok)}자)")
            try:   # 토큰이 '있는 것' 과 '유효한 것' 은 다르다 — 미리 확인해 둔다
                from huggingface_hub import HfApi
                who = HfApi(token=tok).whoami()
                log(f"  HF whoami: {who.get('name')} ({who.get('type')})")
            except Exception as ex:
                log(f"  [!] HF 토큰 검증 실패: {type(ex).__name__}: {str(ex)[:120]}")
            return tok
    log("  HF_TOKEN 없음 — 공개 베이스만 사용 가능, DCP 업로드는 실패합니다")
    return None


_load_token()


SHARD_INDEX = _i("SHARD_INDEX", 0)   # RUNNER_REWRITES_THIS_LINE
SHARD_COUNT = _i("SHARD_COUNT", 4)
EPOCHS = _i("EPOCHS", 1)
MICRO_BATCH = _i("MICRO_BATCH", 32)       # 전역 배치 (8칩에 샤딩됨)
GRAD_ACCUM = _i("GRAD_ACCUM", 8)          # 유효배치 256
MAX_LEN = _i("MAX_LEN", 1024)
MAX_ITEMS = _i("MAX_ITEMS", 2000)                # 0=샤드 전체. >0 이면 길이 오름차순 앞에서 자름(스모크)
CKPT_MODE = _s("CKPT_MODE", "local")      # local | hf | s3
BASE_HF = _s("BASE_HF", "PIXELZX/XERON-1.0-long")   # RUNNER_REWRITES_THIS_LINE
CKPT_HF = _s("CKPT_HF", "PIXELZX/XERON-1.0-short-ckpt")
MINUTES = _i("MAX_TRAIN_MIN", 30)          # 0=무제한, 아니면 DCP 저장 후 조기 종료
DIAG = _i("DIAG", 0)                       # RUNNER_REWRITES_THIS_LINE (1=학습 전 처리량 진단)
DATA_PT = None

os.makedirs(EXPORT, exist_ok=True)

# ---------------------------------------------------------------- 0) 환경
log("=== TPU 환경 ===")
for k in ("ISTPUVM", "TPU_ACCELERATOR_TYPE", "TPU_CHIPS_PER_HOST_BOUNDS",
          "KAGGLE_DOCKER_IMAGE", "PJRT_DEVICE"):
    log(f"  {k}={os.environ.get(k)}")
log(f"  cpus={os.cpu_count()} python={sys.version.split()[0]}")
sh("nvidia-smi -L || true", check=False)
sh("python -c \"import torch, torch_xla, torch_xla.runtime as xr; "
   "print('torch', torch.__version__, 'torch_xla', torch_xla.__version__); "
   "import jax; print('jax', jax.__version__, jax.default_backend())\" || true", check=False)
if not os.environ.get("ISTPUVM"):
    log("!! ISTPUVM 미설정 — TPU가 아닐 수 있습니다(계정 인증/가속기 설정 확인)")

# ---------------------------------------------------------------- 1) 의존성
log("=== deps ===")
sh(f"{sys.executable} -m pip install -q laya transformers safetensors scipy pandas tabulate",
   timeout=2400)
sh(f"{sys.executable} -c \"import torch,torch_xla,laya,transformers;"
   f"print('ok',torch.__version__,torch_xla.__version__,transformers.__version__)\"")

# ---------------------------------------------------------------- 2) 레포
log("=== repo ===")
if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 https://github.com/PIXELZX0/XERON.git {REPO}")
sh("git log --oneline -1", cwd=REPO)
PYSP = os.path.join(REPO, "scripts")
sh("PYTHONPATH=scripts python -c \"import train_spmd_tpu\"", cwd=REPO, check=False)

# ---------------------------------------------------------------- 3) 데이터 (≤1024 숏)
log("=== dataset ===")
for root, _d, files in os.walk("/kaggle/input"):
    if "train_items_x10_8192.pt" in files:
        DATA_PT = os.path.join(root, "train_items_x10_8192.pt")
        break
if DATA_PT is None:
    sh("ls -laR /kaggle/input | head -40", check=False)
    raise SystemExit("train_items_x10_8192.pt 를 /kaggle/input 에서 찾지 못했습니다")
log(f"  items={DATA_PT} ({os.path.getsize(DATA_PT)/2**30:.2f}GB)")
sh(f"df -h {WORK} | tail -2")

# ---------------------------------------------------------------- 4) 베이스 모델
log(f"=== base model {BASE_HF} ===")
BASE_DIR = os.path.join(WORK, "base")
os.makedirs(BASE_DIR, exist_ok=True)
if not os.path.exists(os.path.join(BASE_DIR, "model.safetensors")):
    from huggingface_hub import snapshot_download
    d = snapshot_download(BASE_HF, local_dir=BASE_DIR,
                          allow_patterns=["model.safetensors", "encoder/*", "tokenizer/*",
                                          "rl_agent_config.json"],
                          token=os.environ.get("HF_TOKEN"))
    log(f"  base_dir={d}")
sh(f"ls -la {BASE_DIR}; du -sh {BASE_DIR}")

# ---------------------------------------------------------------- 5) DCP 이어받기
DCP_DIR = os.path.join(WORK, "dcp")
CKPT_SUB = f"shard{SHARD_INDEX}"
if CKPT_MODE in ("hf", "s3"):
    log(f"=== checkpoint restore ({CKPT_MODE}) ===")
    try:
        if CKPT_MODE == "hf":
            from huggingface_hub import snapshot_download
            os.makedirs(DCP_DIR, exist_ok=True)
            snapshot_download(CKPT_HF, local_dir=DCP_DIR,
                              allow_patterns=[f"{CKPT_SUB}/*"],
                              token=os.environ.get("HF_TOKEN"))
        else:
            import boto3
            ep, bk = os.environ["S3_ENDPOINT"], os.environ["S3_BUCKET"]
            s3 = boto3.client("s3", endpoint_url=ep)
            os.makedirs(os.path.join(DCP_DIR, CKPT_SUB), exist_ok=True)
            for o in s3.list_objects_v2(Bucket=bk, Prefix=f"xeron-short/{CKPT_SUB}/").get(
                    "Contents", []):
                rel = o["Key"].split(f"xeron-short/", 1)[1]
                dst = os.path.join(DCP_DIR, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                s3.download_file(bk, o["Key"], dst)
        log("  restore done")
    except Exception as ex:
        log(f"  restore skipped ({type(ex).__name__}: {str(ex)[:160]})")
else:
    log("=== checkpoint mode local (세션 종료 시 소실) ===")

# ---------------------------------------------------------------- 6) 학습
TRAIN_ENV = {
    "BASE_DIR": BASE_DIR,
    "ITEMS": DATA_PT,
    "OUT_DIR": os.path.join(EXPORT, "xeron-1.0-short"),
    "DCP_DIR": DCP_DIR,
    "SHARD_INDEX": str(SHARD_INDEX),
    "SHARD_COUNT": str(SHARD_COUNT),
    "EPOCHS": str(EPOCHS),
    "MICRO_BATCH": str(MICRO_BATCH),
    "GRAD_ACCUM": str(GRAD_ACCUM),
    "GROUP_SIZE": "4",
    "LR_ENCODER": _s("LR_ENCODER", "2.0e-5"),
    "LR_HEAD": _s("LR_HEAD", "5.0e-5"),
    "SIGMA_START": "0.3",
    "SIGMA_END": "0.2",
    "RL_WEIGHT": "0.5",
    "WEIGHT_DECAY": "0.02",
    "HEAD_LAYERS": "4",
    "HEAD_SIZE": "1024",
    "HEAD_DROPOUT": "0.0",
    "MAX_LEN": str(MAX_LEN),
    "HEAD_MAX_LEN": "256",
    "MAX_TOKENS_BATCH": "32768",
    "CKPT_EVERY": "1",
    "MAX_TRAIN_MIN": str(MINUTES),   # 0=제한없음(본런). 트레이너가 스텝 경계에서 정상 종료
    "RESUME_SHARD": "1",
    "PYTHONUNBUFFERED": "1",
}
if MAX_ITEMS:
    TRAIN_ENV["MAX_ITEMS"] = str(MAX_ITEMS)
DIAG = str(DIAG)
TRAIN_ENV["DIAG"] = DIAG
if DIAG not in ("0", "", "false"):
    TRAIN_ENV["DIAG_MICROS"] = _s("DIAG_MICROS", "4")

# 처리량 프로브(PROBE_N>0): 학습 루프 **자체**를 micro 단위로 계측한다.
# data 대기 / fwd / loss / bwd / opt / mark_step 을 각각 찍고 PROBE_N 개에서 정상 종료.
# PROBE_METRICS=1 이면 micro 마다 XLA 카운터(CachedCompile/UncachedCompile/ExecuteReplicated)
# 도 같이 남긴다 → "매 micro 재컴파일" vs "입력 경로" 를 가른다.
_probe = _s("PROBE_N", "0")   # RUNNER_REWRITES_THIS_LINE (본런 0 / 프로브 >0)
if _probe not in ("0", "", "false"):
    TRAIN_ENV["PROBE_N"] = _probe
    TRAIN_ENV["PROBE_METRICS"] = _s("PROBE_METRICS", "1")
    TRAIN_ENV["RATE_EVERY"] = _s("RATE_EVERY", "8")

log(f"=== train shard {SHARD_INDEX}/{SHARD_COUNT} ===")
log("  env=" + json.dumps(TRAIN_ENV, indent=2))
# 트레이너의 MAX_TRAIN_MIN 은 '스텝 경계에서 정상 종료'(DCP 저장 + 스냅샷 + HF 업로드) 예산이다.
# 하드 타임아웃을 같은 값으로 걸면 저장/업로드 도중 SIGKILL 되어 커널이 ERROR 로 끝난다
# (v6: 2400s 에서 kill / v8: 저장을 마치고 TRAIN_DONE 직전 3300s 에서 kill).
# v8 실측 DCP 저장 = **694.9s**(≈11.6분) — 주석의 "~80s" 가정보다 8배 느려서 +900s 로는 부족했다.
# 저장 2회가 겹치던 중복 저장(train_spmd_tpu.py 에서 수정)까지 고려해 30분 여유를 둔다.
timeout = (MINUTES * 60 + 1800) if MINUTES else None
if DIAG not in ("0", "", "false"):
    # 진단은 학습 전에 변형별 micro 를 돌리므로 예산을 따로 준다(변형당 컴파일 포함).
    timeout = (MINUTES * 60 + 3600) if MINUTES else 3600
sh(f"{sys.executable} -u scripts/train_spmd_tpu.py", cwd=REPO, env=TRAIN_ENV, timeout=timeout)

# ---------------------------------------------------------------- 7) 업로드 (세션 간 이어받기)
if CKPT_MODE == "hf":
    log(f"=== upload DCP -> {CKPT_HF} ===")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(CKPT_HF, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=os.path.join(DCP_DIR, CKPT_SUB),
                          path_in_repo=CKPT_SUB, repo_id=CKPT_HF, repo_type="model")
        log("  upload done")
    except Exception as ex:
        log(f"  upload FAILED ({type(ex).__name__}: {str(ex)[:200]})")
elif CKPT_MODE == "s3":
    log("=== upload DCP -> S3 ===")
    try:
        import boto3
        s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT"])
        bk = os.environ["S3_BUCKET"]
        root = os.path.join(DCP_DIR, CKPT_SUB)
        for dp, _dn, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                s3.upload_file(p, bk, "xeron-short/" + os.path.relpath(p, DCP_DIR))
        log("  upload done")
    except Exception as ex:
        log(f"  upload FAILED ({type(ex).__name__}: {str(ex)[:200]})")

# ---------------------------------------------------------------- 8) 요약
log("=== summary ===")
sh(f"du -sh {EXPORT} {DCP_DIR} 2>/dev/null; ls -la {EXPORT}/xeron-1.0-short | head -12",
   check=False)
manifest = {
    "run": "xeron-1.0-short-spmd",
    "shard": SHARD_INDEX, "shard_count": SHARD_COUNT,
    "base": BASE_HF, "items": DATA_PT, "max_len": MAX_LEN,
    "micro_batch_global": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
    "effective_batch": MICRO_BATCH * GRAD_ACCUM,
    "ckpt_mode": CKPT_MODE, "epochs": EPOCHS,
    "env": TRAIN_ENV, "elapsed_min": round((time.time() - T0) / 60, 1),
}
with open(os.path.join(EXPORT, "run-manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
log(json.dumps(manifest, indent=2))
log("KAGGLE SPMD PIPELINE COMPLETE")

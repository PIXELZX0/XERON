#!/usr/bin/env python3
"""XERON-0.5 fine-tune on Kaggle (L4 bf16, fallback T4 fp16).

Why 0.4 exists (XERON-0.3 post-mortem):
  0.3 (T4x2 fp16, 2 epochs on the 62k x3 mix) came out OVERCONFIDENT:
  fitted calibration temperatures were [3.14, 5.41, 5.12] vs 0.2's [0.96, 1.10, 0.57],
  epoch-2 avg loss rose (1.075 -> 1.289 = overfitting), and the RL policy-gradient
  term explodes as sigma anneals (logp scales with 1/(2*sigma^2), 50x at sigma=0.1).
  Result: overall 0.528 vs 0.2's 0.541, soft_acc 0.350 vs 0.535.

  0.4 therefore:
    - base = PIXELZX/XERON-0.2 (the best checkpoint we have, temperature ~1.0)
    - 1 epoch only (epoch 2 was where 0.3 degraded)
    - bf16 on L4 (same numeric regime as the A100 run that produced 0.2)
    - rl_weight 0.5 (halve the PG term), sigma 0.3 -> 0.2 (no 1/sigma^2 blowup)
    - lr_head 5e-5 (was 1e-4), weight_decay 0.02 (was 0.01)

Pipeline: deps -> repo -> base -> 1-epoch RLCD train -> snapshot -> JevBench public-231
          + typed-decisions eval, all under /kaggle/working/export.
"""
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "XERON")
JEV = os.path.join(WORK, "jevbench")
EXPORT = os.path.join(WORK, "export")
DATA_NAME = "train_items_x5.pt"


def find_data():
    """Kaggle mount paths vary (/kaggle/input/<slug>/ vs /kaggle/input/datasets/<slug>/)."""
    for root, _dirs, files in os.walk("/kaggle/input"):
        if DATA_NAME in files:
            return os.path.join(root, DATA_NAME)
    return f"/kaggle/input/xeron-0-5-train-items/{DATA_NAME}"


BASE_MODEL = os.environ.get("BASE_MODEL", "PIXELZX/XERON-0.2")
RUN_NAME = "xeron-0.5"
OUT_DIR = f"./output/{RUN_NAME}"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sh(cmd, cwd=None, env=None, check=True, timeout=None):
    log(f"$ {cmd}")
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, shell=True, cwd=cwd, env=e, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}")
    return r.returncode


def py(code, **kw):
    """Run a python snippet from a temp FILE.

    Never pass code through `python -c <json.dumps(code)>`: the shell keeps the escaped
    \\n literal and it dies with "SyntaxError: unexpected character after line
    continuation character" (that is exactly what broke the 0.3 eval stage).
    """
    path = os.path.join(WORK, "_snippet.py")
    with open(path, "w") as fh:
        fh.write(code)
    return sh(f"{sys.executable} -u {path}", **kw)


def out(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw).stdout.strip()


os.makedirs(EXPORT, exist_ok=True)

# ---------------------------------------------------------------- 0) environment
log("=== GPU ===")
gpu_csv = out("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
log(gpu_csv)
NGPU = max(1, len([l for l in gpu_csv.splitlines() if l.strip()]))
GPU_NAME = gpu_csv.splitlines()[0].split(",")[0].strip() if gpu_csv else "?"
log(f"NGPU={NGPU} GPU_NAME={GPU_NAME}")
log(f"python={sys.version.split()[0]}  cwd={os.getcwd()}")

DATA = find_data()
log(f"DATA={DATA} exists={os.path.exists(DATA)}")
if not os.path.exists(DATA):
    sh("ls -laR /kaggle/input/ | head -60 || true", check=False)
    raise SystemExit(f"training data missing: {DATA}")
sh(f"df -h {WORK} | tail -2")

# bf16 only on Ampere+/Ada (L4, A100, H100). T4/V100/P100 are fp16.
DTYPE = "bf16" if any(t in GPU_NAME for t in ("L4", "A100", "H100", "6000")) else "fp16"
MICRO_BATCH = 4 if DTYPE == "bf16" else 2
GRAD_ACCUM = 8
log(f"DTYPE={DTYPE} MICRO_BATCH={MICRO_BATCH} GRAD_ACCUM={GRAD_ACCUM} "
    f"eff_batch={MICRO_BATCH * NGPU * GRAD_ACCUM}")

# ---------------------------------------------------------------- 1) deps
log("=== install deps ===")
sh(f"{sys.executable} -m pip install -q laya tabulate scipy", timeout=1800)
py("import torch, laya; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
   "'ngpu', torch.cuda.device_count(), 'laya', getattr(laya,'__version__','?'))")
rc = py("import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)", check=False)
if rc != 0:
    log("!! CUDA lost after pip install -> reinstalling torch cu128")
    sh(f"{sys.executable} -m pip install -q --force-reinstall torch "
       "--index-url https://download.pytorch.org/whl/cu128", timeout=2400)
    py("import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")

# ---------------------------------------------------------------- 2) repo
log("=== clone XERON ===")
if not os.path.isdir(REPO):
    sh(f"git clone --depth 1 https://github.com/PIXELZX0/XERON.git {REPO}")
sh("git log --oneline -1", cwd=REPO)

# ---------------------------------------------------------------- 3) base model
log(f"=== download base model {BASE_MODEL} ===")
base_dir = subprocess.run(
    [sys.executable, "-c",
     "from huggingface_hub import snapshot_download; "
     f"print(snapshot_download({BASE_MODEL!r}, allow_patterns=['*.json','*.safetensors','tokenizer/*']))"],
    capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
log(f"base_dir={base_dir}")
sh(f"ls -la {base_dir}")

# ---------------------------------------------------------------- 4) train (1 epoch)
log(f"=== train (1 epoch, {NGPU}x{GPU_NAME}, {DTYPE}) ===")
TRAIN_ENV = {
    "EPOCHS": "1",
    "MICRO_BATCH": str(MICRO_BATCH),
    "GRAD_ACCUM": str(GRAD_ACCUM),
    "GROUP_SIZE": "4",
    "LR_ENCODER": "2e-5",
    "LR_HEAD": "5e-5",
    "SIGMA_START": "0.3",
    "SIGMA_END": "0.2",
    "RL_WEIGHT": "0.5",
    "WEIGHT_DECAY": "0.02",
    "DTYPE": DTYPE,
    "MAX_LEN": "4096",
    "HEAD_MAX_LEN": "256",
    "MAX_TOKENS_BATCH": "8192",
    "CHECKPOINT_EVERY": "1",
    "PYTHONUNBUFFERED": "1",
}
sh(f"torchrun --standalone --nproc_per_node={NGPU} scripts/train_ddp.py "
   f'"{base_dir}" {OUT_DIR} "{DATA}"',
   cwd=REPO, env=TRAIN_ENV)

# ---------------------------------------------------------------- 5) snapshot
log("=== pack inference-ready snapshot ===")
snap_dir = os.path.join(EXPORT, f"{RUN_NAME}-snapshot")
os.makedirs(snap_dir, exist_ok=True)
for rel in ("model.safetensors", "rl_agent_config.json", "encoder", "tokenizer",
            "tokenizer_config.json", "config.json"):
    src = os.path.join(REPO, OUT_DIR, rel)
    if os.path.exists(src):
        dst = os.path.join(snap_dir, rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
sh(f"ls -la {snap_dir}")
sh(f"tar cf {os.path.join(EXPORT, RUN_NAME + '-snapshot.tar')} -C {EXPORT} {RUN_NAME}-snapshot")
sh(f"rm -f {REPO}/{OUT_DIR}/checkpoint_epoch*.pt", check=False)
sh(f"du -sh {EXPORT} {REPO}/{OUT_DIR}")

# ---------------------------------------------------------------- 6) evaluation
MODEL_DIR = os.path.join(REPO, OUT_DIR)
log("=== JevBench public-231 eval ===")
if not os.path.isdir(JEV):
    sh(f"git clone --depth 1 https://github.com/fstandhartinger/jevbench.git {JEV}")

EVAL_PY = f'''
import json, sys, time
sys.path.insert(0, {JEV!r})
from jevbench.adapters import LayaLocalAdapter
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.tasks import load_jsonl
PUB = {JEV!r} + "/datasets/public"
tasks = []
for f in ("original", "easy", "hard"):
    tasks.extend(load_jsonl(f"{{PUB}}/{{f}}.jsonl"))
print("public tasks:", len(tasks), flush=True)
ad = LayaLocalAdapter(endpoint={MODEL_DIR!r}, model={RUN_NAME!r}, threads=4)
t0 = time.perf_counter(); ad.load()
print("loaded in %.1fs device=%s" % (time.perf_counter()-t0, getattr(ad._agent, "device", "?")), flush=True)
for t in tasks[:2]:
    ad.run(t)
r = Runner(ad, Ledger({EXPORT!r} + "/jevbench.ledger.jsonl"), {EXPORT!r} + "/jevbench.raw")
t0 = time.perf_counter()
recs = r.run_all(tasks, progress_every=50, results_path={EXPORT!r} + "/{RUN_NAME}.results.jsonl")
print("jevbench done %d/%d in %.1fs" % (sum(1 for x in recs if x["ok"]), len(recs), time.perf_counter()-t0), flush=True)
'''
with open(os.path.join(EXPORT, "jevbench_eval.py"), "w") as fh:
    fh.write(EVAL_PY)
sh(f"{sys.executable} -u {os.path.join(EXPORT, 'jevbench_eval.py')}")

log("=== typed-decisions eval ===")
sh(f"{sys.executable} -u scripts/evaluate.py --model {MODEL_DIR} --split test "
   f"--device cuda --output {os.path.join(EXPORT, 'typed-decisions-eval.json')}", cwd=REPO)

# ---------------------------------------------------------------- 7) summary
log("=== summary ===")
sh(f"find {EXPORT} -maxdepth 2 -type f | head -40")
manifest = {
    "run": RUN_NAME,
    "base_model": BASE_MODEL,
    "data": DATA,
    "epochs": 1,
    "dtype": DTYPE,
    "gpus": f"{NGPU}x {GPU_NAME} (Kaggle)",
    "train_env": TRAIN_ENV,
    "export": sorted(os.path.relpath(os.path.join(d, f), EXPORT)
                     for d, _, fs in os.walk(EXPORT) for f in fs),
}
with open(os.path.join(EXPORT, "run-manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
log(json.dumps(manifest, indent=2))
log("XERON-0.5 PIPELINE COMPLETE")

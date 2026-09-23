#!/usr/bin/env python3
"""XERON-0.3 fine-tune on Kaggle (T4 x2, fp16) + evaluation.

Pipeline: deps -> repo -> base model (PIXELZX/XERON-0.2) -> 2-epoch RLCD train
          -> inference-ready snapshot -> JevBench public-231 eval + typed-decisions eval.

Everything lands in /kaggle/working/export so `kaggle kernels output` retrieves it.
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
DATA = "/kaggle/input/xeron-0-3-train-items/train_items_x3.pt"


def find_data():
    """Kaggle mount paths vary (/kaggle/input/<slug>/ vs /kaggle/input/datasets/<slug>/)."""
    if os.path.exists(DATA):
        return DATA
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "train_items_x3.pt" in files:
            return os.path.join(root, "train_items_x3.pt")
    return DATA

BASE_MODEL = os.environ.get("BASE_MODEL", "PIXELZX/XERON-0.2")
RUN_NAME = "xeron-0.3"
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
    """Run a python snippet. Writes it to a temp file and runs that file.

    NOTE: never pass the code through `python -c <json.dumps(code)>` — the shell keeps
    the escaped \n literal, which dies with "SyntaxError: unexpected character after
    line continuation character" (this broke the XERON-0.3 eval stage).
    """
    path = os.path.join(WORK, "_snippet.py")
    with open(path, "w") as fh:
        fh.write(code)
    return sh(f"{sys.executable} -u {path}", **kw)


os.makedirs(EXPORT, exist_ok=True)

# ---------------------------------------------------------------- 0) environment
log("=== GPU ===")
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
log(f"python={sys.version.split()[0]}  cwd={os.getcwd()}")
DATA = find_data()
log(f"DATA={DATA} exists={os.path.exists(DATA)}")
if not os.path.exists(DATA):
    sh("ls -laR /kaggle/input/ | head -60 || true", check=False)
    raise SystemExit(f"training data missing: {DATA}")
sh(f"df -h {WORK} | tail -2")

# ---------------------------------------------------------------- 1) deps
log("=== install deps ===")
sh(f"{sys.executable} -m pip install -q laya tabulate scipy", timeout=1800)
py("import torch, laya; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
   "'ngpu', torch.cuda.device_count(), 'laya', getattr(laya,'__version__','?'))")
# guard: if pip swapped in a CPU-only torch, repair it
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

# ---------------------------------------------------------------- 4) train
log("=== train (2 epochs, T4 x2, fp16) ===")
TRAIN_ENV = {
    "EPOCHS": "2",
    "MICRO_BATCH": "2",
    "GRAD_ACCUM": "16",          # effective batch = 2 * 2 GPUs * 16 = 64
    "GROUP_SIZE": "4",
    "LR_ENCODER": "2.5e-5",
    "LR_HEAD": "1e-4",
    "SIGMA_START": "0.4",
    "SIGMA_END": "0.1",
    "DTYPE": "fp16",
    "MAX_LEN": "4096",
    "HEAD_MAX_LEN": "256",
    "MAX_TOKENS_BATCH": "8192",
    "CHECKPOINT_EVERY": "1",
    "PYTHONUNBUFFERED": "1",
}
sh(f"torchrun --standalone --nproc_per_node=2 scripts/train_ddp.py "
   f'"{base_dir}" {OUT_DIR} "{DATA}"',
   cwd=REPO, env=TRAIN_ENV)

# ---------------------------------------------------------------- 5) snapshot + cleanup
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
# drop multi-GB optimizer checkpoints from the output set (keep the model)
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
recs = r.run_all(tasks, progress_every=50, results_path={EXPORT!r} + "/xeron-0.3.results.jsonl")
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
    "epochs": 2,
    "effective_batch": 64,
    "dtype": "fp16",
    "gpus": "T4 x2 (Kaggle)",
    "train_env": TRAIN_ENV,
    "export": sorted(os.path.relpath(os.path.join(d, f), EXPORT)
                     for d, _, fs in os.walk(EXPORT) for f in fs),
}
with open(os.path.join(EXPORT, "run-manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
log(json.dumps(manifest, indent=2))
log("XERON-0.3 PIPELINE COMPLETE")

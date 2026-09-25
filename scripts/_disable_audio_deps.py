#!/usr/bin/env python3
"""Make torchaudio/torchvision unimportable in this interpreter.

Why: transformers imports `torchaudio` at import time (processing_utils ->
audio_utils, guarded by `is_torchaudio_available()`). Base images often ship a
torchaudio built for a different torch, so the import raises
`OSError: ... undefined symbol: ...` and ModernBert import dies with it.
XERON training needs neither package.

Strategy (non-destructive):
  1. if the module resolves in the *venv*, uninstall it there (handled by pip);
  2. otherwise rename both the package directory and its *.dist-info / *.egg-info
     to `<name>.disabled` so `importlib.util.find_spec` and
     `importlib.metadata.version` both stop seeing it.

Idempotent: re-running is a no-op once disabled.
"""
import glob
import importlib.util
import os
import sys

for mod in ("torchaudio", "torchvision"):
    try:
        spec = importlib.util.find_spec(mod)
    except Exception as e:  # broken parent packages
        print(f"[deps] find_spec({mod}) raised {type(e).__name__}: {e}")
        spec = None
    if spec is None:
        print(f"[deps] {mod}: not found (ok)")
    else:
        root = (spec.submodule_search_locations or [None])[0]
        if root and os.path.isdir(root) and not root.endswith(".disabled"):
            dst = root + ".disabled"
            try:
                if os.path.exists(dst):
                    os.rename(root, dst + ".old")
                os.rename(root, dst)
                print(f"[deps] {mod}: package disabled ({root} -> {dst})")
            except Exception as e:
                print(f"[deps] {mod}: could not rename {root}: {e}")
        else:
            print(f"[deps] {mod}: spec has no dir ({root})")

    for sp in list(sys.path):
        if not sp or not os.path.isdir(sp):
            continue
        for pat in (f"{mod}-*.dist-info", f"{mod}-*.egg-info", f"{mod}.dist-info"):
            for p in glob.glob(os.path.join(sp, pat)):
                if p.endswith(".disabled"):
                    continue
                try:
                    os.rename(p, p + ".disabled")
                    print(f"[deps] metadata disabled: {p}")
                except Exception as e:
                    print(f"[deps] metadata rename failed {p}: {e}")

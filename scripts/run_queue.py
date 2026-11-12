"""Run the ablation ladder unattended (CLAUDE.md section 19.15), then cache features for each encoder.

    python -m scripts.run_queue            # start, or continue after any stop: finished runs are skipped

Per run: wait until enough commit is free (Windows crashed loader workers at ~63 of 65 GB commit, section 20), train,
resume from checkpoints/<name>/last.pt after a crash (up to MAX_TRIES, 90 s cool-down), write checkpoints/<name>/DONE.
Then features/<name>/ from the final-epoch encoder (no checkpoint picking: every rung is read at epoch 30).
Progress: outputs/ablation/queue.log; each run's console output: outputs/ablation/<name>.log.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil

PLAIN = ["views.policy=plain", "views.n_local=0", "mining.enabled=false", "mining.beta=1.0"]
NO_MINE_B1 = ["mining.enabled=false", "mining.beta=1.0"]
# rung name -> overrides of configs/ablation.yaml (= full TRACE). Order: the table's ends first, so a short night
# still gives plain vs full.
ABLATION = {
    "abl_r0_plain": PLAIN,
    "abl_r5_full": [],
    "abl_r1_views": ["views.artifacts=false", "views.n_local=0"] + NO_MINE_B1,
    "abl_r2_artifacts": ["views.n_local=0"] + NO_MINE_B1,
    "abl_r3_multicrop": NO_MINE_B1,
    "abl_r4_beta": ["mining.enabled=false"],
    "abl_r0_plain_s1": PLAIN + ["seed=1"],
    "abl_r5_full_s1": ["seed=1"],
}
MIN_FREE_GB = 22.0  # commit headroom: training ~16 GB + margin
MAX_TRIES = 20
LOG = "outputs/ablation/queue.log"


def say(msg: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def commit_free_gb() -> float:
    """Free Windows commit (RAM + page file): what ran out when loader workers crashed (section 20). Training ~16 GB."""
    if os.name != "nt":
        return psutil.virtual_memory().available / 1e9
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("tot_phys", "avail_phys", "tot_page", "avail_page", "tot_virt", "avail_virt",
                                              "avail_ext")]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail_page / 1e9


def wait_for_ram():
    warned = False
    while (free := commit_free_gb()) < MIN_FREE_GB:
        if not warned:
            say(f"waiting: {free:.1f} GB commit free, need {MIN_FREE_GB} (close Chrome / other apps)")
            warned = True
        time.sleep(60)


def run(cmd: list[str], log_path: str) -> int:
    with open(log_path, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def train(name: str, overrides: list[str]):
    out = os.path.join("checkpoints", name)
    if os.path.exists(os.path.join(out, "DONE")):
        say(f"{name}: already done, skipped")
        return True
    for t in range(1, MAX_TRIES + 1):
        wait_for_ram()
        resume = os.path.join(out, "last.pt")
        cmd = [sys.executable, "-m", "encoder.train", "--config-name=ablation", f"encoder_name={name}"] + overrides
        if os.path.exists(resume):
            cmd.append(f"ckpt.resume={resume}")
        say(f"{name}: try {t}{' (resume)' if os.path.exists(resume) else ''}: {' '.join(overrides) or 'full TRACE'}")
        t0 = time.time()
        code = run(cmd, f"outputs/ablation/{name}.log")
        if code == 0:
            open(os.path.join(out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
            say(f"{name}: finished in {(time.time() - t0) / 60:.0f} min")
            return True
        say(f"{name}: exit code {code} after {(time.time() - t0) / 60:.0f} min; cool-down 90 s")
        time.sleep(90)
    say(f"{name}: gave up after {MAX_TRIES} tries")
    return False


if __name__ == "__main__":
    os.makedirs("outputs/ablation", exist_ok=True)
    say(f"queue start: {len(ABLATION)} runs")
    ok = [n for n, o in ABLATION.items() if train(n, o)]
    for n in ok:
        if os.path.exists(os.path.join("features", n, "pool_global.npy")):
            continue
        wait_for_ram()
        say(f"{n}: caching features")
        code = run([sys.executable, "-m", "encoder.features", "--name", n, "--weights",
                    os.path.join("checkpoints", n, "encoder.pt")], f"outputs/ablation/{n}.log")
        say(f"{n}: features exit code {code}")
    say("queue end")

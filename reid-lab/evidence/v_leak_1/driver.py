"""v_leak / driver.py -- run one stage script once per fold in separate processes (N at a time), keep each
process's output and exit code.   python -I -B driver.py SCRIPT.py NAME [workers] [clip ...]"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data  # noqa: E402

HERE = "C:/Research/plugins/tools/reid_lab/work/v_leak_1/"
script, name = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
clips = sys.argv[4:] or list(data.DEV_CLIPS)


def run(clip):
    t0 = time.time()
    for attempt in range(4):   # native Windows crashes (exit 0xC000070A, no output) happen at random under load: retry those
        p = subprocess.run([sys.executable, "-I", "-B", HERE + script, clip], capture_output=True, text=True, cwd=HERE)
        if p.returncode == 0 or p.stdout.strip() or "Traceback" in p.stderr:
            break
    open(HERE + f"out/logs/{name}_{clip}.log", "w", encoding="utf-8").write(p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else ""))
    return clip, p.returncode, round(time.time() - t0, 1), p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip()[-600:]


bad = 0
with ThreadPoolExecutor(workers) as ex, open(HERE + f"out/{name}.log", "w", encoding="utf-8") as log:
    for clip, rc, sec, last in ex.map(run, clips):
        line = f"[exit {rc}, {sec} s] {last}"
        print(line, flush=True)
        log.write(line + "\n")
        bad += rc != 0
print("failed processes:", bad)
sys.exit(1 if bad else 0)

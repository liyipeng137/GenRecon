"""Run a command under the shared reconstruction GPU lease and record resources."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("/tmp/scene_reconstruction_gpu_0.lock"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required")
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / "run.json"
    if record.exists():
        parser.error("Choose a new attempt output directory")
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=repo)
    (args.output / "source_changes.patch").write_bytes(patch)
    source_paths = set(subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines())
    source_paths.update(str(p.relative_to(repo)) for p in (repo / "scripts").glob("*.py"))
    source_hashes = {p: hashlib.sha256((repo / p).read_bytes()).hexdigest()
                     for p in sorted(source_paths) if (repo / p).is_file()}
    result = {"status": "waiting_for_gpu", "command": command, "cwd": str(repo),
              "git_head": head, "source_sha256": source_hashes,
              "started_at": datetime.now(timezone.utc).isoformat(), "samples": []}
    memory_limit = Path("/sys/fs/cgroup/memory.max")
    if memory_limit.exists():
        result["container_memory_limit_bytes"] = memory_limit.read_text().strip()
    record.write_text(json.dumps(result, indent=2) + "\n")
    with args.lock.open("a+") as lease, (args.output / "console.log").open("w") as log:
        fcntl.flock(lease, fcntl.LOCK_EX)
        start = time.monotonic()
        child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PYTHONUNBUFFERED": "1"})
        result["status"] = "running"
        result["pid"] = child.pid
        record.write_text(json.dumps(result, indent=2) + "\n")
        while child.poll() is None:
            sample = {"elapsed_seconds": time.monotonic() - start}
            try:
                smi = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                               "--format=csv,noheader,nounits"], text=True, timeout=5)
                sample["device_memory_mib"], sample["device_utilization_percent"] = [int(v.strip()) for v in smi.splitlines()[0].split(",")]
                status = Path(f"/proc/{child.pid}/status").read_text()
                sample["process_rss_kib"] = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError):
                pass
            result["samples"].append(sample)
            time.sleep(1)
        result["exit_code"] = child.wait()
        result["seconds"] = time.monotonic() - start
    changed = [p for p, digest in source_hashes.items()
               if not (repo / p).exists() or hashlib.sha256((repo / p).read_bytes()).hexdigest() != digest]
    result["source_changed_during_run"] = changed
    result["status"] = "complete" if result["exit_code"] == 0 and not changed else "failed"
    result["sampled_peak_device_memory_mib"] = max((s.get("device_memory_mib", 0) for s in result["samples"]), default=0)
    result["sampled_peak_process_rss_kib"] = max((s.get("process_rss_kib", 0) for s in result["samples"]), default=0)
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "source_sha256"}}, indent=2))
    raise SystemExit(result["exit_code"] or bool(changed))


if __name__ == "__main__":
    main()

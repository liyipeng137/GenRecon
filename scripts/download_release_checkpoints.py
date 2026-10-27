"""Download released checkpoints with resumable, validated HTTP byte ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


SIZES = {"sparse_structure": 2862971400, "shape_slat": 5410532174, "texture_slat": 5410728782}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "download_ranges.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    tasks = []
    fds = {}
    for name, size in SIZES.items():
        path = args.output / f"{name}.pt"
        if name not in state:
            prefix = path.stat().st_size if path.exists() else 0
            if prefix > size:
                raise ValueError(f"Unexpected checkpoint size: {path}")
            state[name] = {"bytes": size, "prefix_bytes": prefix, "completed": []}
        rec = state[name]
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        fds[name] = fd
        completed = set(rec["completed"])
        for start in range(rec["prefix_bytes"], size, 64 * 1024**2):
            end = min(start + 64 * 1024**2, size) - 1
            if start not in completed:
                tasks.append((name, start, end))

    def save_state():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    save_state()

    def download(task):
        name, start, end = task
        url = f"https://kaldir.vc.cit.tum.de/genrecon/{name}.pt"
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    expected = f"bytes {start}-{end}/{SIZES[name]}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Server did not honor range: {response.status} {response.headers.get('Content-Range')}")
                    offset = start
                    while block := response.read(1024**2):
                        if offset + len(block) > end + 1:
                            raise ValueError("Oversized range response")
                        written = 0
                        while written < len(block):
                            written += os.pwrite(fds[name], block[written:], offset + written)
                        offset += len(block)
                    if offset != end + 1:
                        raise ValueError(f"Truncated range {offset} != {end + 1}")
                return task
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            name, start, end = future.result()
            state[name]["completed"].append(start)
            save_state()
            print(f"{count}/{len(tasks)} {name} {start}-{end}", flush=True)
    manifest = []
    for name, fd in fds.items():
        os.fsync(fd)
        os.close(fd)
        path = args.output / f"{name}.pt"
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        manifest.append({"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                         "url": f"https://kaldir.vc.cit.tum.de/genrecon/{name}.pt"})
    (args.output / "release_checkpoints.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()

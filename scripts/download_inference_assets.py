"""Download only the pretrained assets used by the released 512 pipeline."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


ASSETS = [
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "config.json"),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", "model.safetensors"),
    ("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.json"),
    ("microsoft/TRELLIS-image-large", "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
    ("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16.json"),
    ("microsoft/TRELLIS.2-4B", "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"),
    ("microsoft/TRELLIS.2-4B", "ckpts/tex_dec_next_dc_f16c32_fp16.json"),
    ("microsoft/TRELLIS.2-4B", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(hf_hub_download, repo, name): (repo, name) for repo, name in ASSETS}
        for future in as_completed(futures):
            repo, name = futures[future]
            path = Path(future.result())
            record = {"repository": repo, "file": name, "path": str(path), "bytes": path.stat().st_size}
            records.append(record)
            print(json.dumps(record), flush=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

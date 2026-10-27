#!/usr/bin/env bash
set -euo pipefail
task_python="${GENRECON_PYTHON:-/kiri/envs/genrecon/bin/python}"
task_source="${GENRECON_FLASH_SOURCE:-/kiri/tools/genrecon_deps/flash-attention}"
task_patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_expected_commit=49774b31c8d1172a5b28f72e9aed0f322bf4ccba
if [[ "$(git -C "$task_source" rev-parse HEAD)" != "$task_expected_commit" ]]; then
    echo "FlashAttention checkout differs from the audited patch base: $task_expected_commit" >&2
    exit 1
fi
if ! git -C "$task_source" apply --reverse --check "$task_patch_dir/flash_inference_only.patch" 2>/dev/null; then
    git -C "$task_source" apply --check "$task_patch_dir/flash_inference_only.patch"
    git -C "$task_source" apply "$task_patch_dir/flash_inference_only.patch"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS=1
export FLASH_ATTN_CUDA_ARCHS=80
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_LOCAL_VERSION=infer89
"$task_python" -m pip install --force-reinstall --no-cache-dir --no-build-isolation --no-deps "$task_source"

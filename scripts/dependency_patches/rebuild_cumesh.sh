#!/usr/bin/env bash
set -euo pipefail
task_python="${GENRECON_PYTHON:-/kiri/envs/genrecon/bin/python}"
task_source="${GENRECON_CUMESH_SOURCE:-/kiri/tools/genrecon_deps/CuMesh}"
task_patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_expected_commit=12289e1062f0603f2f0d0771b02e1395d247f26f
if [[ "$(git -C "$task_source" rev-parse HEAD)" != "$task_expected_commit" ]]; then
    echo "CuMesh checkout differs from the audited patch base: $task_expected_commit" >&2
    exit 1
fi
if ! git -C "$task_source" apply --reverse --check "$task_patch_dir/cumesh_zero_work.patch" 2>/dev/null; then
    git -C "$task_source" apply --check "$task_patch_dir/cumesh_zero_work.patch"
    git -C "$task_source" apply "$task_patch_dir/cumesh_zero_work.patch"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS=2
"$task_python" -m pip install --force-reinstall --no-cache-dir --no-build-isolation --no-deps "$task_source"

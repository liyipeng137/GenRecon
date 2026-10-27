#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GENRECON_PYTHON=${GENRECON_PYTHON:-/kiri/envs/genrecon/bin/python}
GENRECON_RUN_ROOT=${GENRECON_RUN_ROOT:-/kiri/output/scene_reconstruction/genrecon_first60_20261027}
GENRECON_WEIGHTS=${GENRECON_WEIGHTS:-/kiri/models/genrecon}
GENRECON_ATTEMPT=${GENRECON_ATTEMPT:-attempt_01}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
cd "$REPO_ROOT"

"$GENRECON_PYTHON" scripts/run_profiled.py --output "$GENRECON_RUN_ROOT/$GENRECON_ATTEMPT" -- \
  "$GENRECON_PYTHON" reconstruct_scene.py \
  --mode Iphone --path "$GENRECON_RUN_ROOT/data" \
  --output_path "$GENRECON_RUN_ROOT/$GENRECON_ATTEMPT/result" \
  --ss_ckpt "$GENRECON_WEIGHTS/sparse_structure.pt" \
  --shape_ckpt "$GENRECON_WEIGHTS/shape_slat.pt" \
  --tex_ckpt "$GENRECON_WEIGHTS/texture_slat.pt" \
  --num_imgs_per_scene 60 --pipeline 512 --pipeline_config configs/pipelines/texture.json \
  --colmap_subdir colmap --skip_point_cleaning --min_points_per_chunk 100 \
  --chunk_size_factor 1.11 --image_batch_size 2 --proj_batch_voxels 1024 \
  --joint_decode_max_chunks_per_group 1 --joint_decode_max_inflated_voxels 20000 \
  --joint_decode_overlap_r_in 16 --seed 42

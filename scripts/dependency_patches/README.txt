Reproducible GenRecon dependency patches

These are small source patches and build/test entrypoints; no dependency checkout,
model weights or binary wheels are stored here. Exact bases, limits and test
results are recorded in manifest.json. The current environment is Python 3.12.3,
Torch 2.10.0+cu128 and CUDA toolkit 12.8 on RTX 4090 (sm89).

CuMesh zero-work fix:
  Clone https://github.com/JeffreyXiang/CuMesh recursively, then checkout
  12289e1062f0603f2f0d0771b02e1395d247f26f and update submodules.
  Set GENRECON_CUMESH_SOURCE to that checkout if it is not at the recorded
  /kiri/tools/genrecon_deps/CuMesh default. Set GENRECON_PYTHON if needed.
  Run bash rebuild_cumesh.sh. It validates the base, applies the patch exactly
  once and builds with MAX_JOBS=2. CUDA_HOME defaults to /usr/local/cuda-12.8.
  Run the regression with:
    /kiri/envs/genrecon/bin/python test_cumesh_zero_work.py --phase after --output /tmp/cumesh_zero_work.json
  The 15 CUDA cases cover single/disconnected faces, a connected square,
  duplicate faces, a non-manifold edge, empty faces and B>0/M0 boundaries.
  The last fixture has3 isolated boundary components; even permissive hole
  filling preserves all 10 input faces and 9 input vertices. The patch guards
  zero-block CUDA launches while retaining ID initialization/compaction.

FlashAttention inference build:
  Clone https://github.com/Dao-AILab/flash-attention at v2.8.3.post2,
  commit 49774b31c8d1172a5b28f72e9aed0f322bf4ccba, with its submodules.
  Set GENRECON_FLASH_SOURCE if the checkout is not at the recorded default.
  Run bash rebuild_flash.sh. It validates/applies the patch and builds against
  the active environment without downloading a different Torch.
  This build retains all standard FA2 forward head dimensions and FP16/BF16;
  sm80 kernels run on Ada sm89. It disables backward, dropout, ALiBi, local
  windows and softcap. Use it only for matching inference calls. A training
  task or an attention caller requiring disabled options needs a full build.
  The patch propagates feature flags to both C++ and CUDA and guards the
  unused standalone backward dispatcher to avoid unresolved dynamic symbols.

Do not run CUDA regression fixtures concurrently with a formal reconstruction.
The actual container memory limit is about 57.7GiB, even if free reports 503GiB.

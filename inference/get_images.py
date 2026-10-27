from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from genrecon.datasets.components import chunk_visible_in_camera
from genrecon.pipelines.types import SelectedImages
from inference.view_selection import chunk_projection_areas, coverage_report, select_pose_coverage

# ScanNet++ Nerfstudio DSLR poses are stored in a world frame that differs from
# the aligned scan mesh frame. Empirically, camera centres map as:
# (x_ns, y_ns, z_ns) = (y_mesh, x_mesh, -z_mesh)
# so we remap poses back into the mesh-aligned world before chunk normalisation.
_NERFSTUDIO_TO_MESH_WORLD = torch.tensor(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

# Blender camera-axis → OpenCV (y and z flipped).
_FLIP_Y_Z = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))


def _to_chunk0_extrinsics(c2w_world: torch.Tensor, m_o2c_0: torch.Tensor) -> torch.Tensor:
    """OpenCV world-to-camera in chunk-0 frame: flip @ inv(m_o2c_0 @ c2w_world)."""
    return _FLIP_Y_Z @ torch.linalg.inv(m_o2c_0 @ c2w_world)


def _processed_crop_geometry(intrinsics, width, height, crop_box=None):
    """K and crop used by loading, also reused for selection visibility."""
    intr = intrinsics.clone()
    if crop_box is not None or width == height:
        return intr, crop_box
    cx_px, cy_px = float(intr[0, 2]) * width, float(intr[1, 2]) * height
    size = int(2 * min(cx_px, width - cx_px, cy_px, height - cy_px))
    if size < 1:
        raise ValueError("Principal point does not permit a nonempty square crop")
    left = max(0, min(int(round(cx_px - .5 * size)), width - size))
    top = max(0, min(int(round(cy_px - .5 * size)), height - size))
    intr[0, 0] *= width / size
    intr[1, 1] *= height / size
    intr[0, 2] = (intr[0, 2] * width - left) / size
    intr[1, 2] = (intr[1, 2] * height - top) / size
    return intr, (left, top, size)


class BaseImageSelecter:
    def __init__(self, view_selection="pose_coverage", pose_position_threshold=.05,
                 pose_rotation_threshold=10., min_views_per_chunk=3):
        if view_selection not in ("uniform", "pose_coverage"):
            raise ValueError("view_selection must be uniform or pose_coverage")
        if not np.isfinite(pose_position_threshold) or pose_position_threshold <= 0:
            raise ValueError("pose_position_threshold must be finite and positive")
        if not np.isfinite(pose_rotation_threshold) or not 0 < pose_rotation_threshold <= 180:
            raise ValueError("pose_rotation_threshold must be in (0,180]")
        if isinstance(min_views_per_chunk, bool) or not isinstance(min_views_per_chunk, int) or min_views_per_chunk < 1:
            raise ValueError("min_views_per_chunk must be a positive integer")
        self.view_selection = view_selection
        self.pose_position_threshold = pose_position_threshold
        self.pose_rotation_threshold = pose_rotation_threshold
        self.min_views_per_chunk = min_views_per_chunk

    def _camera_chunk_areas(self, cameras, transforms):
        """Max area over a raw image's actual crops; never deduplicate crops."""
        tagged = [{**cam, "_selection_parent": i} for i, cam in enumerate(cameras)]
        views = self._prepare_scene_pool(tagged)
        intrinsics = []
        for view in views:
            with Image.open(view["img_path"]) as image:
                intr, _ = _processed_crop_geometry(view["intrinsics"], *image.size,
                                                   view.get("crop_box"))
            intrinsics.append(intr.cpu().numpy())
        chunks = np.stack([t.cpu().numpy() for t in transforms]) if transforms else np.empty((0, 4, 4))
        projected = chunk_projection_areas(
            np.stack([v["c2w_blender"].cpu().numpy() for v in views]),
            np.stack(intrinsics), chunks)
        areas = np.zeros((len(cameras), len(chunks)), dtype=np.float64)
        np.maximum.at(areas, [v["_selection_parent"] for v in views], projected)
        return areas

    def _prepare_scene_pool(self, scene_picks_raw: list[dict]) -> list[dict]:
        """Hook for subclasses that want to expand the scene pool before image
        loading (e.g. iPhone two-crop). Default is identity — non-iPhone
        selecter classes are byte-identical to the pre-hook behavior."""
        return scene_picks_raw

    def _process_images(
        self,
        img_paths: list[str],
        intrinsics: torch.Tensor,
        crop_boxes: list[tuple[int, int, int] | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load images, square-crop, apply alpha masking, and resize to 512 and 1024.

        Two cropping paths:
        - ``crop_boxes[i] is None`` (or ``crop_boxes`` is ``None``): legacy
          behavior — center-crop around the principal point when non-square,
          and update intrinsics for the crop in-method.
        - ``crop_boxes[i] = (left, top, S)``: apply the pre-computed crop;
          the matching ``intrinsics[i]`` is assumed already adjusted for it.

        Args:
            img_paths:  list of N image path strings.
            intrinsics: (N, 3, 3) normalized intrinsic matrices.
            crop_boxes: optional per-image (left, top, S) crop box; entries may
                be ``None`` to fall through to the legacy center-crop path.

        Returns:
            Tuple of:
                - updated_intrinsics: (N, 3, 3) intrinsics adjusted for any crop
                - images_512:         (N, 3, 512, 512) float32 tensor in [0, 1]
                - images_1024:        (N, 3, 1024, 1024) float32 tensor in [0, 1]
        """
        updated_intrinsics: list[torch.Tensor] = []
        images_512: list[torch.Tensor] = []
        images_1024: list[torch.Tensor] = []

        for i, (img_path, intr) in enumerate(zip(img_paths, intrinsics)):
            image = Image.open(img_path)
            W, H = image.size
            crop_box = crop_boxes[i] if crop_boxes is not None else None
            intr, crop_box = _processed_crop_geometry(intr, W, H, crop_box)
            if crop_box is not None:
                left, top, size = crop_box
                image = image.crop((left, top, left + size, top + size))

            if "A" in image.getbands():
                alpha = torch.tensor(np.array(image.getchannel("A"))).float() / 255.0
            else:
                alpha = None

            updated_intrinsics.append(intr)

            for size, out_list in ((512, images_512), (1024, images_1024)):
                resized = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
                tensor = torch.tensor(np.array(resized)).permute(2, 0, 1).float() / 255.0
                if alpha is not None:
                    tensor = (
                        tensor
                        * torch.nn.functional.interpolate(
                            alpha[None, None], size=(size, size), mode="bilinear", align_corners=False
                        ).squeeze()
                    )
                out_list.append(tensor)

        return torch.stack(updated_intrinsics), torch.stack(images_512), torch.stack(images_1024)

    def _check_acceptance(self, transf_extr: torch.Tensor, intr: torch.Tensor) -> bool:
        """Chunk is visible from the given camera (frustum + projected-area threshold)."""
        return chunk_visible_in_camera(transf_extr, intr)

    def _pick_cond2d_for_chunk(
        self,
        m_o2c_i: torch.Tensor,
        m_o2c_0: torch.Tensor,
        cameras: list[dict],
        rng: random.Random,
        chunk_idx: int | None = None,
    ) -> dict:
        """Pick ONE randomly-shuffled visible camera for chunk i (rejection-sampled).

        Falls back to the camera whose centre is closest to the chunk centre
        (origin in the chunk-local frame) when no camera in the pool passes
        visibility — keeps every chunk in play instead of dropping it.
        """

        def _pack(cam: dict) -> dict:
            return {
                "img_path": cam["img_path"],
                "intrinsics": cam["intrinsics"],
                "extrinsics_c0": _to_chunk0_extrinsics(cam["c2w_blender"], m_o2c_0),
                "crop_box": cam.get("crop_box"),
            }

        camera_indices = list(range(len(cameras)))
        rng.shuffle(camera_indices)
        for idx in camera_indices:
            cam = cameras[idx]
            ext_i = _FLIP_Y_Z @ torch.linalg.inv(m_o2c_i @ cam["c2w_blender"])
            if self._check_acceptance(ext_i, cam["intrinsics"]):
                return _pack(cam)

        # No camera is strictly visible — fall back to the closest one.
        best_idx = min(
            range(len(cameras)),
            key=lambda j: float(torch.linalg.norm((m_o2c_i @ cameras[j]["c2w_blender"])[:3, 3])),
        )
        tag = f"chunk {chunk_idx}" if chunk_idx is not None else "chunk"
        print(
            f"[get_images] {tag}: no visible camera in scene_picks; "
            f"falling back to closest camera ({Path(cameras[best_idx]['img_path']).name})."
        )
        return _pack(cameras[best_idx])

    def _sample_scene_cameras(
        self,
        cameras: list[dict],
        num_imgs_per_scene: int,
        rng: random.Random,
        m_original_to_chunk: list[torch.Tensor] | None = None,
    ) -> list[dict]:
        """Select original frames before crop expansion, retaining the hard cap.

        Pose selection uses actual crop-frustum coverage. The uniform branch
        preserves the historical original-camera filtering and linspace picks.
        Both report coverage in the actual processed crop geometry.
        """
        if isinstance(num_imgs_per_scene, bool) or not isinstance(num_imgs_per_scene, int) or num_imgs_per_scene < 1:
            raise ValueError("num_imgs_per_scene must be a positive integer")
        if not cameras:
            raise ValueError("No cameras remain after loading/exclusion")
        transforms = list(m_original_to_chunk or [])
        areas = self._camera_chunk_areas(cameras, transforms)
        if self.view_selection == "uniform":
            candidate_ids = []
            for i, cam in enumerate(cameras):
                if not transforms or any(self._check_acceptance(
                    _FLIP_Y_Z @ torch.linalg.inv(t @ cam["c2w_blender"]), cam["intrinsics"])
                    for t in transforms):
                    candidate_ids.append(i)
            fallback = not candidate_ids
            if fallback:
                candidate_ids = list(range(len(cameras)))
            if len(candidate_ids) > num_imgs_per_scene:
                offsets = np.linspace(0, len(candidate_ids) - 1, num_imgs_per_scene).round().astype(int)
                indices = [candidate_ids[i] for i in offsets]
            else:
                indices = candidate_ids
            report = coverage_report(areas, indices, min_views_per_chunk=self.min_views_per_chunk)
            report.update(strategy="uniform", input_camera_count=len(cameras),
                          candidate_camera_count=len(candidate_ids),
                          budget_original_images=num_imgs_per_scene,
                          selected_original_images=len(indices), selected_indices=indices,
                          all_invisible_fallback=fallback)
        else:
            poses = np.stack([c["c2w_blender"].cpu().numpy() for c in cameras])
            intrinsics = np.stack([c["intrinsics"].cpu().numpy() for c in cameras])
            if transforms:
                scales = [abs(np.linalg.det(t[:3, :3].double().cpu().numpy())) ** (-1. / 3.)
                          for t in transforms]
                scale = float(np.median(scales))
            else:
                scale = max(float(np.linalg.norm(np.ptp(poses[:, :3, 3], axis=0))), 1e-8)
            indices, report = select_pose_coverage(
                poses, intrinsics, areas, num_imgs_per_scene, position_scale=scale,
                position_threshold=self.pose_position_threshold,
                rotation_threshold_degrees=self.pose_rotation_threshold,
                min_views_per_chunk=self.min_views_per_chunk)
            report["position_scale_source"] = "median_chunk_side" if transforms else "camera_span"
        report["selected_image_paths"] = [cameras[i]["img_path"] for i in indices]
        self._selection_report = report
        print(f"[view_selection] {self.view_selection}: {len(cameras)} originals -> {len(indices)}; "
              f"uncovered chunks={report['uncovered_chunk_indices']}", flush=True)
        return [cameras[i] for i in indices]

    def get_images(
        self,
        m_original_to_chunk: list[torch.Tensor],
        camera_params_path: Path | str,
        num_imgs_per_scene: int,
        out_path: Path | str,
        seed: int = 0,
        exclude_images: list[str] | set[str] | None = None,
    ) -> SelectedImages:
        """Select the scene-wide image set + one 2D-cond view per chunk.

        Every chunk gets a pick: a randomly-shuffled visible camera when one
        exists in the scene-wide pool, otherwise a fallback to the camera
        whose centre sits closest to the chunk centre. `chunk_indices` maps
        the returned per-chunk entries back to positions in
        `m_original_to_chunk`.

        Extrinsics (both scene set and per-chunk 2D-cond set) are OpenCV w2c
        in the chunk-0 local frame. Passing `m_o2c[0]` as the anchor keeps this
        frame consistent even when chunk 0 is dropped downstream.

        `exclude_images`: basenames (e.g. "IMG_9869.png") to drop before
        selection. Use this to skip frames flagged as misregistered by
        inference/eval_reprojection.py.
        """
        cameras = self._get_cameras(camera_params_path)
        if exclude_images:
            exclude_set = {Path(n).name for n in exclude_images}
            kept = [c for c in cameras if Path(c["img_path"]).name not in exclude_set]
            dropped = [c for c in cameras if Path(c["img_path"]).name in exclude_set]
            if dropped:
                print(
                    f"[get_images] excluding {len(dropped)} image(s): "
                    f"{sorted(Path(c['img_path']).name for c in dropped)}"
                )
            cameras = kept
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        m_o2c_0 = m_original_to_chunk[0]

        # ── Scene-wide set ───────────────────────────────────────────────
        # The per-chunk 2D-cond pick is restricted to this same pool so that
        # the model only ever sees `num_imgs_per_scene` distinct images total.
        # Pass the chunk transforms so the linspace pick stays within cameras
        # visible from at least one chunk (fixes the N=1 single-chunk case).
        scene_picks_raw = self._sample_scene_cameras(
            cameras,
            num_imgs_per_scene,
            rng,
            m_original_to_chunk=m_original_to_chunk,
        )
        # iPhone two-crop expansion happens here: identity for non-iPhone
        # classes, 1→2 entries per non-square frame for iPhone with
        # center_crop=False. Cond2d picking then runs against this expanded
        # pool so visibility chooses the better-fitting crop per chunk.
        scene_picks_raw = self._prepare_scene_pool(scene_picks_raw)
        # Make legacy principal-point centre crops explicit too, so the 2D
        # anchor visibility check sees exactly the K used by image loading.
        processed_pool = []
        for cam in scene_picks_raw:
            with Image.open(cam["img_path"]) as image:
                intr, box = _processed_crop_geometry(cam["intrinsics"], *image.size,
                                                     cam.get("crop_box"))
            processed_pool.append({**cam, "intrinsics": intr, "crop_box": box})
        scene_picks_raw = processed_pool
        scene_picks = [
            {
                "img_path": cam["img_path"],
                "intrinsics": cam["intrinsics"],
                "extrinsics_c0": _to_chunk0_extrinsics(cam["c2w_blender"], m_o2c_0),
                "crop_box": cam.get("crop_box"),
            }
            for cam in scene_picks_raw
        ]
        scene_intr_pre = torch.stack([p["intrinsics"] for p in scene_picks])
        scene_ext_c0 = torch.stack([p["extrinsics_c0"] for p in scene_picks])
        scene_intr, scene_imgs_512, scene_imgs_1024 = self._process_images(
            [p["img_path"] for p in scene_picks],
            scene_intr_pre,
            crop_boxes=[p["crop_box"] for p in scene_picks],
        )

        # ── Per-chunk 2D cond (drawn from the same expanded scene-wide pool) ─
        cond2d_picks: list[dict] = []
        chunk_indices: list[int] = []
        for chunk_idx, m_o2c_i in enumerate(m_original_to_chunk):
            pick = self._pick_cond2d_for_chunk(
                m_o2c_i,
                m_o2c_0,
                scene_picks_raw,
                rng,
                chunk_idx=chunk_idx,
            )
            cond2d_picks.append(pick)
            chunk_indices.append(chunk_idx)

        cond2d_intr_pre = torch.stack([p["intrinsics"] for p in cond2d_picks])
        cond2d_ext_c0_list = [p["extrinsics_c0"] for p in cond2d_picks]
        cond2d_intr_stack, cond2d_imgs_512_stack, cond2d_imgs_1024_stack = self._process_images(
            [p["img_path"] for p in cond2d_picks],
            cond2d_intr_pre,
            crop_boxes=[p.get("crop_box") for p in cond2d_picks],
        )

        report = dict(self._selection_report)
        report["selected_crop_views"] = len(scene_picks)
        report["cond2d_selection"] = "seeded_random_actual_crop_visible_then_closest"
        report["cond2d_views"] = []
        crop_cameras = {(c["img_path"], c.get("crop_box")): c for c in scene_picks_raw}
        for j, (idx, pick) in enumerate(zip(chunk_indices, cond2d_picks)):
            camera = crop_cameras[(pick["img_path"], pick.get("crop_box"))]
            extrinsics = _to_chunk0_extrinsics(camera["c2w_blender"], m_original_to_chunk[idx])
            report["cond2d_views"].append({
                "chunk_index": idx, "image_path": pick["img_path"], "crop_box": pick.get("crop_box"),
                "frustum_visible": self._check_acceptance(extrinsics, cond2d_intr_stack[j]),
            })
        (out_path / "view_selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

        # ── Debug dump ───────────────────────────────────────────────────
        with (out_path / "cameras.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene": [
                        {
                            "img_path": p["img_path"],
                            "extrinsics_c0": p["extrinsics_c0"].tolist(),
                            "intrinsics": scene_intr[i].tolist(),
                        }
                        for i, p in enumerate(scene_picks)
                    ],
                    "chunks": [
                        {
                            "chunk_index": chunk_indices[i],
                            "cond2d_view": {
                                "img_path": cond2d_picks[i]["img_path"],
                                "extrinsics_c0": cond2d_ext_c0_list[i].tolist(),
                                "intrinsics": cond2d_intr_stack[i].tolist(),
                            },
                        }
                        for i in range(len(cond2d_picks))
                    ],
                },
                f,
                indent=2,
            )

        return SelectedImages(
            scene_images_1024=scene_imgs_1024,
            scene_images_512=scene_imgs_512,
            scene_intrinsics=scene_intr,
            scene_extrinsics_c0=scene_ext_c0,
            cond2d_images_1024=list(cond2d_imgs_1024_stack),
            cond2d_images_512=list(cond2d_imgs_512_stack),
            cond2d_intrinsics=list(cond2d_intr_stack),
            cond2d_extrinsics_c0=cond2d_ext_c0_list,
            chunk_indices=chunk_indices,
        )


class SageMixin:
    def _get_cameras(self, camera_params_path: Path | str) -> list[dict]:
        """
        Load camera parameters from a Blender-style transforms.json file.

        Args:
            camera_params_path: path to transforms.json, e.g.
                renders_room/0a1be5f3/transforms.json

        Returns:
            List of dicts, one per frame, each with:
                - "img_path"      – absolute path to the render image (str)
                - "c2w_blender"   – (4, 4) camera-to-world in Blender space (torch.Tensor)
                - "intrinsics"    – (3, 3) normalized intrinsic matrix (torch.Tensor)
        """
        camera_params_path = Path(camera_params_path)
        with camera_params_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_dir = camera_params_path.parent
        default_fov_x = data.get("camera_angle_x")

        cameras = []
        for frame in data["frames"]:
            fov_x_val = frame.get("camera_angle_x", default_fov_x)
            if fov_x_val is None:
                raise KeyError(
                    f"camera_angle_x missing on frame {frame.get('file_path')!r} and at top level of {camera_params_path}"
                )
            fov_x = torch.tensor(fov_x_val, dtype=torch.float32)
            f_xy = 0.5 / torch.tan(fov_x / 2)
            intrinsics = torch.tensor([[f_xy, 0.0, 0.5], [0.0, f_xy, 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            cameras.append(
                {
                    "img_path": str(image_dir / frame["file_path"]),
                    "c2w_blender": c2w,
                    "intrinsics": intrinsics,
                }
            )

        return cameras


class ScannetMixin:
    def _get_cameras(self, camera_params_path: Path | str) -> list[dict]:
        """
        Load camera parameters from a ScanNet++ Nerfstudio transforms JSON.

        Args:
            camera_params_path: path to transforms_undistorted.json, e.g.
                <scene>/dslr/nerfstudio/transforms_undistorted.json
            Images are expected at <scene>/dslr/resized_undistorted_images/.

        Returns:
            List of dicts, one per frame, each with:
                - "img_path"    – absolute path to the DSLR image (str)
                - "c2w_blender" – (4, 4) camera-to-world remapped into the mesh-aligned world frame (torch.Tensor)
                - "intrinsics"  – (3, 3) normalized intrinsic matrix (torch.Tensor)
        """
        camera_params_path = Path(camera_params_path)
        image_root = camera_params_path.parent.parent / "resized_undistorted_images"

        with camera_params_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cameras = []
        for frame in data["frames"]:
            width = float(frame.get("w", data["w"]))
            height = float(frame.get("h", data["h"]))
            fl_x = float(frame.get("fl_x", data["fl_x"]))
            fl_y = float(frame.get("fl_y", data["fl_y"]))
            cx = float(frame.get("cx", data["cx"]))
            cy = float(frame.get("cy", data["cy"]))
            intrinsics = torch.tensor(
                [[fl_x / width, 0.0, cx / width], [0.0, fl_y / height, cy / height], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )

            t_c2w_nerfstudio = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w_blender = _NERFSTUDIO_TO_MESH_WORLD @ t_c2w_nerfstudio

            cameras.append(
                {
                    "img_path": str(image_root / frame["file_path"]),
                    "c2w_blender": c2w_blender,
                    "intrinsics": intrinsics,
                }
            )
        return cameras


class SageImageSelecter(BaseImageSelecter, SageMixin):
    pass


class ScannetIphoneMixin:
    def _get_cameras(self, camera_params_path: Path | str) -> list[dict]:
        """
        Load camera parameters from ScanNet++ iPhone COLMAP files.

        Args:
            camera_params_path: path to <scene>/iphone/colmap/cameras.txt
                (any file in <scene>/iphone/colmap/ works — the directory is used
                to locate cameras.txt + images.txt; image files come from
                <scene>/iphone/rgb/).

        ScanNet++ iPhone COLMAP runs use the OPENCV (radial+tangential) camera
        model with non-zero distortion. The downstream pipeline assumes pure
        pinhole projection, so we undistort each frame with cv2.undistort using
        the COLMAP intrinsics + distortion, derive a new pinhole K via
        cv2.getOptimalNewCameraMatrix(alpha=0), and cache the undistorted JPEGs
        into <scene>/iphone/rgb_undistorted/. Subsequent loads are cache hits.
        Cameras already in PINHOLE / SIMPLE_PINHOLE form pass through unchanged.

        COLMAP poses are world-to-camera in OpenCV camera convention. ScanNet++
        registers the iPhone COLMAP world frame to the mesh-aligned scene frame,
        so no extra world-axis remap is needed (unlike the DSLR nerfstudio path).
        We convert each pose to c2w in Blender camera convention via the
        right-multiply with ``_FLIP_Y_Z`` so downstream `_to_chunk0_extrinsics`
        produces the same OpenCV w2c output as the DSLR path.

        Returns:
            List of dicts, one per frame, each with:
                - "img_path"    – absolute path to the (undistorted) iPhone RGB frame (str)
                - "c2w_blender" – (4, 4) camera-to-world in mesh-aligned world,
                                  Blender camera axes (torch.Tensor)
                - "intrinsics"  – (3, 3) normalized pinhole intrinsics, post-undistort (torch.Tensor)
        """
        camera_params_path = Path(camera_params_path)
        colmap_dir = camera_params_path.parent if camera_params_path.is_file() else camera_params_path
        cameras_txt = colmap_dir / "cameras.txt"
        images_txt = colmap_dir / "images.txt"
        image_root = colmap_dir.parent / "rgb"
        # Cache for undistorted frames. Defaults to <scene>/iphone/rgb_undistorted/
        # but can be redirected (e.g. when the dataset dir is read-only) by setting
        # `self.undistort_cache_dir` on the selecter — frames are namespaced by
        # scene id under that root to avoid collisions across scenes.
        cache_override = getattr(self, "undistort_cache_dir", None)
        if cache_override is not None:
            scene_id = colmap_dir.parent.parent.name
            image_root_undist = Path(cache_override) / scene_id
        else:
            image_root_undist = colmap_dir.parent / "rgb_undistorted"

        if not image_root.is_dir():
            raise FileNotFoundError(
                f"iPhone RGB directory not found: {image_root}. "
                f"ScanNet++ ships iPhone footage as {colmap_dir.parent / 'rgb.mkv'} — "
                f"extract frames into {image_root}/ before running this mode."
            )

        # ── Parse cameras.txt → per-camera pinhole K (post-undistort) + remap maps ─
        cam_table: dict[int, dict] = {}
        with cameras_txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model = parts[1]
                W = int(float(parts[2]))
                H = int(float(parts[3]))
                params = [float(x) for x in parts[4:]]

                if model == "PINHOLE":
                    fx, fy, cx, cy = params[:4]
                    dist = None
                elif model == "SIMPLE_PINHOLE":
                    f_focal, cx, cy = params[:3]
                    fx = fy = f_focal
                    dist = None
                elif model == "OPENCV":
                    fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
                    dist = np.array([k1, k2, p1, p2], dtype=np.float64)
                elif model == "RADIAL":
                    f_focal, cx, cy, k1, k2 = params[:5]
                    fx = fy = f_focal
                    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
                elif model == "SIMPLE_RADIAL":
                    f_focal, cx, cy, k1 = params[:4]
                    fx = fy = f_focal
                    dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
                elif model == "OPENCV_FISHEYE":
                    raise NotImplementedError(
                        f"Camera {cam_id} uses OPENCV_FISHEYE; this loader handles only "
                        f"the OPENCV (radial+tangential) model. Fisheye undistortion needs "
                        f"cv2.fisheye.undistortImage and is not currently wired up."
                    )
                else:
                    raise ValueError(f"Unsupported COLMAP camera model: {model}")

                K_pix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

                if dist is None or not np.any(dist):
                    cam_table[cam_id] = {
                        "needs_undistort": False,
                        "W": W,
                        "H": H,
                        "K": K_pix,
                    }
                    continue

                # Pinhole K that maximises valid-pixel coverage of the undistorted output.
                new_K, roi = cv2.getOptimalNewCameraMatrix(K_pix, dist, (W, H), alpha=0, newImgSize=(W, H))
                x, y, w, h = roi
                if w <= 0 or h <= 0:
                    raise RuntimeError(
                        f"Empty undistortion ROI for camera {cam_id} " f"(W={W}, H={H}, dist={dist.tolist()})."
                    )
                # Precompute the per-pixel remap once; reuse for every frame from this camera.
                mapx, mapy = cv2.initUndistortRectifyMap(K_pix, dist, None, new_K, (W, H), cv2.CV_16SC2)
                # Shift principal point so the cropped output is consistent with `new_K`.
                K_cropped = new_K.copy()
                K_cropped[0, 2] -= x
                K_cropped[1, 2] -= y
                cam_table[cam_id] = {
                    "needs_undistort": True,
                    "W": int(w),
                    "H": int(h),
                    "K": K_cropped,
                    "roi": (int(x), int(y), int(w), int(h)),
                    "mapx": mapx,
                    "mapy": mapy,
                }

        if any(c["needs_undistort"] for c in cam_table.values()):
            image_root_undist.mkdir(exist_ok=True)

        # ── Parse images.txt: 2 lines per image (header + POINTS2D); we only need the header ─
        cameras: list[dict] = []
        with images_txt.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        i = 0
        n_undistorted_now = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            # NAME may contain spaces (e.g. iCloud "IMG_1907 2.png"), so cap split
            # at 9 so parts[9] keeps the full filename.
            parts = line.split(maxsplit=9)
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = (float(x) for x in parts[1:5])
            tx, ty, tz = (float(x) for x in parts[5:8])
            cam_id = int(parts[8])
            # Some ScanNet++ scenes encode NAME as "video/frame_xxx.jpg" while the
            # actual extracted RGB lives flat in iphone/rgb/; take the basename
            # so both layouts resolve correctly.
            name = Path(parts[9]).name

            cam = cam_table[cam_id]
            if cam["needs_undistort"]:
                dst = image_root_undist / name
                if not dst.is_file():
                    src = image_root / name
                    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                    if img is None:
                        raise FileNotFoundError(f"iPhone RGB frame not readable: {src}")
                    undist = cv2.remap(img, cam["mapx"], cam["mapy"], cv2.INTER_LINEAR)
                    x, y, w, h = cam["roi"]
                    undist = undist[y : y + h, x : x + w]
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(dst), undist):
                        raise RuntimeError(f"Failed to write undistorted frame: {dst}")
                    n_undistorted_now += 1
                img_path = dst
            else:
                img_path = image_root / name

            W, H, K = cam["W"], cam["H"], cam["K"]
            intrinsics = torch.tensor(
                [
                    [K[0, 0] / W, 0.0, K[0, 2] / W],
                    [0.0, K[1, 1] / H, K[1, 2] / H],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )

            # COLMAP quaternion (qw, qx, qy, qz) → 3x3 rotation (column-vector convention).
            R_w2c = torch.tensor(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ],
                dtype=torch.float32,
            )
            w2c = torch.eye(4, dtype=torch.float32)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)
            c2w_opencv = torch.linalg.inv(w2c)
            c2w_blender = c2w_opencv @ _FLIP_Y_Z

            cameras.append(
                {
                    "img_path": str(img_path),
                    "c2w_blender": c2w_blender,
                    "intrinsics": intrinsics,
                }
            )

            # Skip the POINTS2D line that follows each image header.
            i += 2

        n_total = len(cameras)
        n_cached = n_total - n_undistorted_now
        if any(c["needs_undistort"] for c in cam_table.values()):
            print(
                f"[get_images] {n_total} iPhone frame(s) available "
                f"({n_undistorted_now} newly undistorted, {n_cached} from cache) "
                f"→ {image_root_undist}/"
            )

        return cameras


class ScannetImageSelecter(BaseImageSelecter, ScannetMixin):
    pass


_DEFAULT_IPHONE_UNDISTORT_CACHE = Path("tmp_images")


def _expand_camera_to_crops(cam: dict, center_crop: bool) -> list[dict]:
    """Return 1 or 2 crop-camera entries for ``cam``.

    With ``center_crop=False`` and a non-square image (w != h), produces two
    square crops along the longer axis at the two extremes, leaving the
    shorter axis untouched. Both share the parent's ``c2w_blender``; only the
    in-image intrinsics (fx_norm, fy_norm, cx_norm, cy_norm) differ.

    With ``center_crop=True`` or a square image, returns a single entry with
    ``crop_box=None`` so ``_process_images`` falls through to the legacy
    center-around-principal-point path.
    """
    with Image.open(cam["img_path"]) as image:
        W, H = image.size
    if center_crop or W == H:
        return [{**cam, "crop_box": None}]

    intr = cam["intrinsics"]
    if W > H:
        S = H
        offsets = [(0, 0), (W - S, 0)]
    else:
        S = W
        offsets = [(0, 0), (0, H - S)]

    crops: list[dict] = []
    for left, top in offsets:
        new_intr = intr.clone()
        new_intr[0, 0] = intr[0, 0] * (W / S)
        new_intr[1, 1] = intr[1, 1] * (H / S)
        new_intr[0, 2] = (intr[0, 2] * W - left) / S
        new_intr[1, 2] = (intr[1, 2] * H - top) / S
        crops.append({**cam, "intrinsics": new_intr, "crop_box": (left, top, S)})
    return crops


class _IphoneSelecterMixin:
    """Adds two-crop expansion to the ScanNet++ / MASt3R iPhone selecters.

    ``center_crop=False`` (default): non-square frames produce two square crops
    along the longer axis, doubling the scene-wide input pool. ``center_crop=
    True`` reverts to the legacy single-square-around-principal-point crop.
    """

    center_crop: bool = False

    def _prepare_scene_pool(self, scene_picks_raw: list[dict]) -> list[dict]:
        expanded: list[dict] = []
        for cam in scene_picks_raw:
            expanded.extend(_expand_camera_to_crops(cam, self.center_crop))
        return expanded


class ScannetIphoneImageSelecter(_IphoneSelecterMixin, BaseImageSelecter, ScannetIphoneMixin):
    def __init__(
        self,
        undistort_cache_dir: Path | str | None = None,
        center_crop: bool = False,
        **selection_kwargs,
    ):
        """
        Args:
            undistort_cache_dir: writable directory to cache undistorted iPhone
                JPEGs. Frames are saved under <cache_dir>/<scene_id>/. If None,
                defaults to ``_DEFAULT_IPHONE_UNDISTORT_CACHE`` — the ScanNet++
                data dir is read-only on this cluster, so we never write next
                to the source frames.
            center_crop: if True, fall back to the legacy single-square crop
                around the principal point. Default False → produce two square
                crops along the longer axis for non-square frames.
        """
        super().__init__(**selection_kwargs)
        self.undistort_cache_dir = (
            Path(undistort_cache_dir) if undistort_cache_dir is not None else _DEFAULT_IPHONE_UNDISTORT_CACHE
        )
        self.center_crop = center_crop


class IphoneMixin(ScannetIphoneMixin):
    """MASt3R-SfM iPhone capture laid out as <scene>/rgb/ +
    <scene>/colmap_mast3r/{cameras,images,points3D}.txt.

    Inherits all COLMAP parsing from ScannetIphoneMixin: _get_cameras derives
    image_root = colmap_dir.parent / "rgb", which for cameras.txt at
    <scene>/colmap_mast3r/cameras.txt resolves to <scene>/rgb/ — exactly the
    layout produced by inference/run_mast3r_sfm.py. MASt3R writes PINHOLE with
    no distortion, so the cv2-undistort branch is never entered.
    """


class IphoneImageSelecter(_IphoneSelecterMixin, BaseImageSelecter, IphoneMixin):
    def __init__(self, center_crop: bool = False, **selection_kwargs):
        """
        Args:
            center_crop: if True, fall back to the legacy single-square crop
                around the principal point. Default False → produce two square
                crops along the longer axis for non-square frames.
        """
        super().__init__(**selection_kwargs)
        self.center_crop = center_crop

#!/usr/bin/env python3

"""
python3 data_gen_pipe/render.py \
  --input-root /work/berke_gokmen/data-1/4d_scenes_synthetic \
  --output-root /work/berke_gokmen/data-1/4d_scenes_synthetic_render
"""

from __future__ import annotations

import argparse
import atexit
import ctypes.util
import hashlib
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _has_display_server() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_accessible_dri_render_node() -> bool:
    dri_root = Path("/dev/dri")
    if not dri_root.exists():
        return False
    for path in sorted(dri_root.glob("renderD*")):
        if os.access(path, os.R_OK | os.W_OK):
            return True
    return False


def _has_osmesa() -> bool:
    return ctypes.util.find_library("OSMesa") is not None


def _configure_pyopengl_platform() -> None:
    chosen = os.environ.get("PYOPENGL_PLATFORM")
    if chosen:
        return

    forced = os.environ.get("PARTFRAMECRAFTER_PYOPENGL_PLATFORM")
    if forced:
        chosen = forced
    elif _has_display_server():
        chosen = None
    elif _has_accessible_dri_render_node():
        chosen = "egl"
    elif _has_osmesa():
        chosen = "osmesa"
    else:
        chosen = "egl"

    if chosen:
        os.environ["PYOPENGL_PLATFORM"] = chosen
        if chosen.lower() == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            if not _has_accessible_dri_render_node():
                os.environ.setdefault("EGL_LOG_LEVEL", "fatal")


_configure_pyopengl_platform()

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageFilter
from pyrender.constants import RenderFlags

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    class _TqdmFallback:
        def __init__(self, iterable=None, **_: object):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

        def update(self, _: int = 1) -> None:
            return None

        def set_description(self, *_: object, **__: object) -> None:
            return None

        def set_postfix_str(self, *_: object, **__: object) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        return _TqdmFallback(iterable=iterable, **kwargs)

    tqdm.write = print  # type: ignore[attr-defined]


IMAGE_SIZE = 518
CAMERA_DISTANCE = 3.0
CAMERA_HEIGHT = 0.5
CAMERA_TARGET_Y = 0.0
ORTHO_XMAG = 1.0
ORTHO_YMAG = 1.0
ZNEAR = 0.01
ZFAR = 10.0
MASK_DEPTH_ATOL = 1e-3
MASK_DEPTH_RTOL = 1e-3
HDRI_ROOT = Path("/work/berke_gokmen/datasets/hdr/hdris")
HDRI_FOV_DEG = 62.0
HDRI_BLUR_RADIUS = 2.5
WHITE_BG = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
BLACK_BG = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
AMBIENT_LIGHT = np.array([0.22, 0.22, 0.22], dtype=np.float32)
KEY_LIGHT_INTENSITY = 1.35
FILL_LIGHT_INTENSITIES = (18.0, 9.0)
SHADOW_AMBIENT_BOOST = np.array([0.04, 0.04, 0.04], dtype=np.float32)
DYNAMIC_PATTERN = re.compile(r"dynamic(\d+)_")
STATIC_PATTERN = re.compile(r"static(\d+)_")
ENVIRONMENT_STYLES = ("plain", "studio", "outdoor-hdri")
ENVIRONMENT_VARIATIONS = ("fixed", "per-scene")
INDOOR_HDRI_TOKENS = (
    "studio",
    "photostudio",
    "bathroom",
    "hall",
    "room",
    "warehouse",
    "workshop",
    "cathedral",
    "chapel",
    "church",
    "depot",
    "service",
)
STUDIO_BG = np.array([0.86, 0.89, 0.93, 1.0], dtype=np.float32)
STUDIO_AMBIENT = np.array([0.28, 0.28, 0.28], dtype=np.float32)
STUDIO_FLOOR_RGBA = np.array([214, 206, 191, 255], dtype=np.uint8)
STUDIO_WALL_RGBA = np.array([196, 205, 218, 255], dtype=np.uint8)
STUDIO_FLOOR_THICKNESS = 0.04
STUDIO_WALL_THICKNESS = 0.04
STUDIO_FLOOR_DROP = 0.03
STUDIO_FLOOR_PADDING_X = 0.80
STUDIO_FLOOR_PADDING_Z = 1.10
STUDIO_WALL_HEIGHT_PADDING = 0.90
STUDIO_WALL_WIDTH_PADDING = 0.90
STUDIO_BACKDROP_MARGIN = 0.20
FLOOR_GRID_RESOLUTION = 96
FLOOR_SURFACE_STYLES = ("concrete", "tile", "wood")
FLOOR_VIEW_MARGIN_X = 0.18
FLOOR_VIEW_MARGIN_Z = 0.28
FLOOR_APRON_DROP = 0.06
FLOOR_APRON_CLEARANCE = 0.04


@dataclass(frozen=True)
class StudioEnvironmentPreset:
    name: str
    bg_rgb: Tuple[float, float, float]
    ambient_rgb: Tuple[float, float, float]
    floor_rgba: Tuple[int, int, int, int]
    wall_rgba: Tuple[int, int, int, int]
    floor_drop: float
    floor_padding_x: float
    floor_padding_z: float
    wall_height_padding: float
    wall_width_padding: float
    backdrop_margin: float
    floor_z_shift: float
    wall_x_shift: float


@dataclass(frozen=True)
class StudioEnvironmentConfig:
    name: str
    bg_color: np.ndarray
    ambient_light: np.ndarray
    floor_rgba: np.ndarray
    wall_rgba: np.ndarray
    floor_surface_name: str
    floor_surface_seed: int
    floor_drop: float
    floor_padding_x: float
    floor_padding_z: float
    wall_height_padding: float
    wall_width_padding: float
    backdrop_margin: float
    floor_z_shift: float
    wall_x_shift: float


@dataclass(frozen=True)
class IndoorHDRIEnvironmentConfig:
    name: str
    hdr_path: Path
    rotation_frac: float
    vertical_bias: float
    exposure: float
    floor_environment: StudioEnvironmentConfig


@dataclass(frozen=True)
class LightRig:
    key_light_pose: np.ndarray
    fill_light_poses: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FloorApronConfig:
    min_x: float
    max_x: float
    max_z: float
    bottom_y: float


@dataclass(frozen=True)
class RenderWorkerConfig:
    output_root: Path
    image_size: int
    camera_distance: float
    camera_height: float
    camera_target_y: float
    ortho_xmag: float
    ortho_ymag: float
    znear: float
    zfar: float
    environment_style: str
    environment_variation: str
    environment_seed: int
    hdri_root: Path
    hdri_fov_deg: float
    hdri_blur_radius: float
    mask_depth_atol: float
    shadows: bool
    overwrite: bool


@dataclass(frozen=True)
class SceneRenderResult:
    scene_name: str
    frame_count: int
    error: Optional[str] = None


WORKER_RENDER_CONFIG: Optional[RenderWorkerConfig] = None
WORKER_RENDERER: Optional[pyrender.OffscreenRenderer] = None
WORKER_CAMERA: Optional[pyrender.Camera] = None
WORKER_CAMERA_POSE: Optional[np.ndarray] = None
WORKER_LIGHT_RIG: Optional[LightRig] = None


STUDIO_ENVIRONMENT_PRESETS = (
    StudioEnvironmentPreset(
        name="cool_studio",
        bg_rgb=(0.86, 0.89, 0.93),
        ambient_rgb=(0.28, 0.28, 0.28),
        floor_rgba=(214, 206, 191, 255),
        wall_rgba=(196, 205, 218, 255),
        floor_drop=0.03,
        floor_padding_x=0.80,
        floor_padding_z=1.10,
        wall_height_padding=0.90,
        wall_width_padding=0.90,
        backdrop_margin=0.20,
        floor_z_shift=-0.10,
        wall_x_shift=0.00,
    ),
    StudioEnvironmentPreset(
        name="warm_loft",
        bg_rgb=(0.93, 0.90, 0.84),
        ambient_rgb=(0.30, 0.29, 0.27),
        floor_rgba=(191, 171, 145, 255),
        wall_rgba=(223, 208, 189, 255),
        floor_drop=0.035,
        floor_padding_x=0.92,
        floor_padding_z=1.18,
        wall_height_padding=0.84,
        wall_width_padding=0.96,
        backdrop_margin=0.22,
        floor_z_shift=-0.06,
        wall_x_shift=-0.05,
    ),
    StudioEnvironmentPreset(
        name="gallery_gray",
        bg_rgb=(0.88, 0.89, 0.87),
        ambient_rgb=(0.26, 0.26, 0.26),
        floor_rgba=(184, 185, 179, 255),
        wall_rgba=(211, 214, 209, 255),
        floor_drop=0.025,
        floor_padding_x=0.88,
        floor_padding_z=1.05,
        wall_height_padding=0.98,
        wall_width_padding=0.92,
        backdrop_margin=0.18,
        floor_z_shift=-0.02,
        wall_x_shift=0.06,
    ),
    StudioEnvironmentPreset(
        name="sage_room",
        bg_rgb=(0.86, 0.90, 0.86),
        ambient_rgb=(0.27, 0.28, 0.27),
        floor_rgba=(176, 170, 154, 255),
        wall_rgba=(199, 208, 194, 255),
        floor_drop=0.03,
        floor_padding_x=0.84,
        floor_padding_z=1.14,
        wall_height_padding=0.92,
        wall_width_padding=0.98,
        backdrop_margin=0.24,
        floor_z_shift=-0.12,
        wall_x_shift=0.03,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render generated synthetic 4D scenes into RGB frames and per-dynamic-object masks."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/work/berke_gokmen/data-1/4d_scenes_synthetic"),
        help="Input directory containing per-scene GLB folders from gen.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/work/berke_gokmen/data-1/4d_scenes_synthetic_render"),
        help="Output directory where rendered frames and masks will be written.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="Square render size in pixels. Default: 518.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=CAMERA_DISTANCE,
        help="Distance of the fixed orthographic camera from the origin along +Z.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=CAMERA_HEIGHT,
        help="Camera Y position. Default: 0.5.",
    )
    parser.add_argument(
        "--camera-target-y",
        type=float,
        default=CAMERA_TARGET_Y,
        help="Y coordinate of the look-at target. Default: 0.0 (scene center).",
    )
    parser.add_argument(
        "--ortho-xmag",
        type=float,
        default=ORTHO_XMAG,
        help="Orthographic half-width. Default: 1.0 to cover the normalized world box.",
    )
    parser.add_argument(
        "--ortho-ymag",
        type=float,
        default=ORTHO_YMAG,
        help="Orthographic half-height. Default: 1.0 to cover the normalized world box.",
    )
    parser.add_argument(
        "--znear",
        type=float,
        default=ZNEAR,
        help="Near plane for rendering.",
    )
    parser.add_argument(
        "--zfar",
        type=float,
        default=ZFAR,
        help="Far plane for rendering.",
    )
    parser.add_argument(
        "--mask-depth-atol",
        type=float,
        default=MASK_DEPTH_ATOL,
        help="Absolute tolerance used when comparing object depth to full-scene depth for visible masks.",
    )
    parser.add_argument(
        "--environment-style",
        type=str,
        choices=ENVIRONMENT_STYLES,
        default="outdoor-hdri",
        help="Background/environment style. 'studio' adds a floor and back wall, 'outdoor-hdri' uses an outdoor HDR backdrop plus a floor, and 'plain' keeps the old flat background.",
    )
    parser.add_argument(
        "--environment-variation",
        type=str,
        choices=ENVIRONMENT_VARIATIONS,
        default="per-scene",
        help="How studio environments vary. 'per-scene' deterministically changes the look per scene; 'fixed' keeps one look for all scenes.",
    )
    parser.add_argument(
        "--environment-seed",
        type=int,
        default=0,
        help="Seed used for deterministic studio-environment variation.",
    )
    parser.add_argument(
        "--hdri-root",
        type=Path,
        default=HDRI_ROOT,
        help="Root directory containing HDRI files used by --environment-style outdoor-hdri.",
    )
    parser.add_argument(
        "--hdri-fov-deg",
        type=float,
        default=HDRI_FOV_DEG,
        help="Virtual perspective FOV used when sampling outdoor HDR backdrops. Default: 62 degrees.",
    )
    parser.add_argument(
        "--hdri-blur-radius",
        type=float,
        default=HDRI_BLUR_RADIUS,
        help="Gaussian blur radius applied to the sampled HDR backdrop. Default: 1.4.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of renderer worker processes to run in parallel. Default: 8.",
    )
    parser.add_argument(
        "--shadows",
        action="store_true",
        help="Enable subtle directional-light shadows in the RGB render pass.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rendered files.",
    )
    args = parser.parse_args()

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive.")
    if args.camera_distance <= 0.0:
        raise ValueError("--camera-distance must be positive.")
    if args.ortho_xmag <= 0.0 or args.ortho_ymag <= 0.0:
        raise ValueError("--ortho-xmag and --ortho-ymag must be positive.")
    if args.znear <= 0.0 or args.zfar <= args.znear:
        raise ValueError("--zfar must be greater than --znear, and both must be positive.")
    if args.mask_depth_atol <= 0.0:
        raise ValueError("--mask-depth-atol must be positive.")
    if args.hdri_fov_deg <= 1.0 or args.hdri_fov_deg >= 179.0:
        raise ValueError("--hdri-fov-deg must be in (1, 179).")
    if args.hdri_blur_radius < 0.0:
        raise ValueError("--hdri-blur-radius must be non-negative.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.environment_style == "outdoor-hdri" and not args.hdri_root.is_dir():
        raise FileNotFoundError(f"HDRI root not found: {args.hdri_root}")
    return args


def natural_key(value: str) -> List[object]:
    parts = re.split(r"(\d+)", value)
    key: List[object] = []
    for token in parts:
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(token.lower())
    return key


def load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        return loaded
    if isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded, node_name=path.stem)
        return scene
    raise TypeError(f"Unsupported geometry type in {path}: {type(loaded)}")


def scene_bounds(scene: trimesh.Scene) -> np.ndarray:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Scene bounds are invalid or non-finite.")
    return bounds


def ensure_trimesh_geometry(geometry: object, source: Path) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if hasattr(geometry, "to_trimesh"):
        converted = geometry.to_trimesh()
        if isinstance(converted, trimesh.Trimesh):
            return converted
    raise TypeError(f"Geometry in {source} cannot be converted to trimesh.Trimesh: {type(geometry)}")


def sanitize_texture_image(texture: object, target_channels: str) -> Optional[Image.Image]:
    if texture is None:
        return None
    if target_channels not in {"RGB", "RGBA"}:
        return texture if isinstance(texture, Image.Image) else None

    if isinstance(texture, Image.Image):
        if texture.mode == target_channels:
            return texture
        return texture.convert(target_channels)

    array = np.asarray(texture)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported texture dtype: {array.dtype}")

    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[2] < 1:
        raise ValueError(f"Expected texture with shape (H, W, C), got {array.shape}.")

    if target_channels == "RGB":
        if array.shape[2] == 1:
            rgb = np.repeat(array, 3, axis=2)
        elif array.shape[2] == 2:
            rgb = np.repeat(array[..., :1], 3, axis=2)
        else:
            rgb = array[..., :3]
        return Image.fromarray(rgb, mode="RGB")

    if array.shape[2] == 1:
        rgb = np.repeat(array, 3, axis=2)
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
    elif array.shape[2] == 2:
        rgb = np.repeat(array[..., :1], 3, axis=2)
        alpha = array[..., 1:2]
    elif array.shape[2] == 3:
        rgb = array
        alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
    else:
        rgba = array[..., :4]
        return Image.fromarray(rgba, mode="RGBA")

    return Image.fromarray(np.concatenate([rgb, alpha], axis=2), mode="RGBA")


def sanitize_trimesh_geometry(geometry: trimesh.Trimesh, source: Path) -> trimesh.Trimesh:
    visual = getattr(geometry, "visual", None)
    if visual is None or not getattr(visual, "defined", False):
        return geometry

    material = getattr(visual, "material", None)
    if material is None:
        return geometry

    def assign_texture(attr_name: str, target_channels: str) -> None:
        if not hasattr(material, attr_name):
            return
        texture = getattr(material, attr_name)
        if texture is None:
            return
        try:
            setattr(material, attr_name, sanitize_texture_image(texture, target_channels))
        except Exception:
            setattr(material, attr_name, None)

    assign_texture("image", "RGBA")
    assign_texture("baseColorTexture", "RGBA")
    assign_texture("normalTexture", "RGB")
    assign_texture("emissiveTexture", "RGB")

    if hasattr(material, "baseColorTexture") and getattr(material, "baseColorTexture", None) is None:
        if hasattr(material, "baseColorFactor") and getattr(material, "baseColorFactor", None) is None:
            material.baseColorFactor = getattr(material, "main_color", np.array([255, 255, 255, 255], dtype=np.uint8))

    return geometry


def load_metadata(scene_dir: Path) -> Optional[dict]:
    metadata_path = scene_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict metadata in {metadata_path}, got {type(data).__name__}.")
    return data


def sequence_center_from_metadata(metadata: Optional[dict]) -> Optional[np.ndarray]:
    if metadata is None:
        return None
    frame_entries = metadata.get("frames")
    if not isinstance(frame_entries, list) or not frame_entries:
        return None

    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for entry in frame_entries:
        if not isinstance(entry, dict):
            continue
        bounds = entry.get("bounds")
        bounds_arr = np.asarray(bounds, dtype=np.float64)
        if bounds_arr.shape != (2, 3) or not np.isfinite(bounds_arr).all():
            continue
        mins.append(bounds_arr[0])
        maxs.append(bounds_arr[1])

    if not mins:
        return None
    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    return 0.5 * (global_min + global_max)


def compute_sequence_center(frame_paths: Sequence[Path]) -> np.ndarray:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for frame_path in frame_paths:
        scene = load_scene(frame_path)
        bounds = scene_bounds(scene)
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        raise ValueError("Cannot compute sequence center from an empty frame list.")
    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    return 0.5 * (global_min + global_max)


def sequence_bounds_from_metadata(metadata: Optional[dict]) -> Optional[np.ndarray]:
    if metadata is None:
        return None
    frame_entries = metadata.get("frames")
    if not isinstance(frame_entries, list) or not frame_entries:
        return None

    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for entry in frame_entries:
        if not isinstance(entry, dict):
            continue
        bounds = np.asarray(entry.get("bounds"), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        return None
    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    return np.stack([global_min, global_max], axis=0)


def compute_centered_sequence_bounds(frame_paths: Sequence[Path], center: np.ndarray) -> np.ndarray:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for frame_path in frame_paths:
        scene = center_scene(load_scene(frame_path), center)
        bounds = scene_bounds(scene)
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        raise ValueError("Cannot compute centered sequence bounds from an empty frame list.")
    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    return np.stack([global_min, global_max], axis=0)


def center_scene(scene: trimesh.Scene, center: np.ndarray) -> trimesh.Scene:
    centered = scene.copy()
    centered.apply_translation(-np.asarray(center, dtype=np.float64))
    return centered


def collect_scene_dirs(input_root: Path) -> List[Path]:
    scene_dirs: List[Path] = []
    for path in sorted(input_root.iterdir(), key=lambda p: natural_key(p.name)):
        if not path.is_dir() or path.name.startswith("."):
            continue
        has_metadata = (path / "metadata.json").is_file()
        has_frames = any(path.glob("frame_*_combined.glb"))
        if has_metadata or has_frames:
            scene_dirs.append(path)
    return scene_dirs


def collect_frame_paths(scene_dir: Path, metadata: Optional[dict]) -> List[Path]:
    frame_paths: List[Path] = []
    if metadata is not None:
        frame_entries = metadata.get("frames")
        if isinstance(frame_entries, list):
            for entry in frame_entries:
                if not isinstance(entry, dict):
                    continue
                rel_path = entry.get("path")
                if isinstance(rel_path, str):
                    frame_path = scene_dir / rel_path
                    if frame_path.is_file():
                        frame_paths.append(frame_path)
    if frame_paths:
        return frame_paths
    return sorted(scene_dir.glob("frame_*_combined.glb"), key=lambda p: natural_key(p.name))


def expected_dynamic_ids(metadata: Optional[dict]) -> Optional[List[int]]:
    if metadata is None:
        return None
    dynamic_entries = metadata.get("dynamic_objects")
    if not isinstance(dynamic_entries, list):
        return None
    ids: List[int] = []
    for index, _entry in enumerate(dynamic_entries):
        ids.append(index)
    return ids


def expected_static_ids(metadata: Optional[dict]) -> Optional[List[int]]:
    if metadata is None:
        return None
    static_entries = metadata.get("static_objects")
    if not isinstance(static_entries, list):
        return None
    ids: List[int] = []
    for index, _entry in enumerate(static_entries):
        ids.append(index)
    return ids


def dynamic_index_from_labels(node_name: Optional[str], geom_name: Optional[str]) -> Optional[int]:
    for label in (node_name or "", geom_name or ""):
        match = DYNAMIC_PATTERN.search(label)
        if match:
            return int(match.group(1))
    return None


def static_index_from_labels(node_name: Optional[str], geom_name: Optional[str]) -> Optional[int]:
    for label in (node_name or "", geom_name or ""):
        match = STATIC_PATTERN.search(label)
        if match:
            return int(match.group(1))
    return None


def extract_dynamic_subscenes(scene: trimesh.Scene, source: Path) -> Dict[int, trimesh.Scene]:
    dynamic_scenes: Dict[int, trimesh.Scene] = {}
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        dynamic_index = dynamic_index_from_labels(node_name, geom_name)
        if dynamic_index is None:
            continue
        geometry = sanitize_trimesh_geometry(ensure_trimesh_geometry(scene.geometry[geom_name], source).copy(), source)
        if dynamic_index not in dynamic_scenes:
            dynamic_scenes[dynamic_index] = trimesh.Scene()
        dynamic_scenes[dynamic_index].add_geometry(
            geometry,
            node_name=node_name,
            geom_name=geom_name,
            transform=transform,
        )
    return dynamic_scenes


def extract_static_ids(scene: trimesh.Scene) -> List[int]:
    static_ids = set()
    for node_name in scene.graph.nodes_geometry:
        _transform, geom_name = scene.graph[node_name]
        static_index = static_index_from_labels(node_name, geom_name)
        if static_index is not None:
            static_ids.add(static_index)
    return sorted(static_ids)


def build_render_metadata(
    source_scene_dir: Path,
    metadata: Optional[dict],
    frame_paths: Sequence[Path],
    first_scene: trimesh.Scene,
    dynamic_ids: Sequence[int],
    sequence_center: np.ndarray,
) -> dict:
    if metadata is not None:
        scene_name = metadata.get("scene_name", source_scene_dir.name)
        num_frames = metadata.get("num_frames", len(frame_paths))
        frame_stride = metadata.get("frame_stride")
        scene_num_parts = len(metadata.get("static_objects", [])) if isinstance(metadata.get("static_objects"), list) else None
        dynamic_num_parts = len(metadata.get("dynamic_objects", [])) if isinstance(metadata.get("dynamic_objects"), list) else None
    else:
        scene_name = source_scene_dir.name
        num_frames = len(frame_paths)
        frame_stride = None
        scene_num_parts = None
        dynamic_num_parts = None

    if scene_num_parts is None:
        scene_num_parts = len(extract_static_ids(first_scene))
    if dynamic_num_parts is None:
        dynamic_num_parts = len(dynamic_ids)

    render_metadata = {
        "scene_name": scene_name,
        "source_scene_dir": str(source_scene_dir),
        "num_frames": int(num_frames),
        "dynamic_num_parts": int(dynamic_num_parts),
        "scene_num_parts": int(scene_num_parts),
        "sequence_center": [round(float(value), 6) for value in np.asarray(sequence_center, dtype=np.float64)],
    }
    if frame_stride is not None:
        render_metadata["frame_stride"] = int(frame_stride)
    return render_metadata


def save_render_metadata(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def look_at(
    camera_position: np.ndarray,
    target: np.ndarray,
    world_up: Optional[np.ndarray] = None,
) -> np.ndarray:
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    world_up = np.asarray(world_up, dtype=np.float64).reshape(3)

    forward = target - camera_position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target cannot coincide.")
    forward /= forward_norm

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        for fallback_up in (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ):
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
            if right_norm >= 1e-8:
                world_up = fallback_up
                break
        else:
            raise ValueError("Unable to construct a stable camera basis.")
    right /= right_norm
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = camera_position
    return pose


def create_camera_pose(camera_distance: float, camera_height: float, target_y: float) -> np.ndarray:
    position = np.array([0.0, camera_height, camera_distance], dtype=np.float64)
    target = np.array([0.0, target_y, 0.0], dtype=np.float64)
    return look_at(position, target)


def create_light_rig(camera_distance: float, target_y: float) -> LightRig:
    target = np.array([0.0, target_y, 0.0], dtype=np.float64)
    key_light_pose = look_at(np.array([1.75, 1.65, 2.55], dtype=np.float64), target)
    fill_light_poses = (
        look_at(np.array([0.0, target_y + 0.75, camera_distance + 0.55], dtype=np.float64), target),
        look_at(np.array([-1.65, 1.15, 2.55], dtype=np.float64), target),
    )
    return LightRig(key_light_pose=key_light_pose, fill_light_poses=fill_light_poses)


def create_camera(image_size: int, xmag: float, ymag: float, znear: float, zfar: float) -> pyrender.OrthographicCamera:
    return pyrender.OrthographicCamera(
        xmag=float(xmag),
        ymag=float(ymag),
        znear=float(znear),
        zfar=float(zfar),
    )


def stable_environment_seed(scene_name: str, environment_seed: int, variation_mode: str) -> int:
    key = f"{environment_seed}" if variation_mode == "fixed" else f"{environment_seed}:{scene_name}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def normalized_hdri_label(path: Path) -> str:
    return path.parent.name.lower().replace("-", "_")


def is_indoor_hdri_path(path: Path) -> bool:
    label = normalized_hdri_label(path)
    return any(token in label for token in INDOOR_HDRI_TOKENS)


@lru_cache(maxsize=None)
def discover_outdoor_hdri_paths(hdri_root: Path) -> Tuple[Path, ...]:
    hdr_paths = tuple(sorted(hdri_root.glob("*/*.hdr"), key=lambda p: normalized_hdri_label(p)))
    outdoor_paths = tuple(path for path in hdr_paths if not is_indoor_hdri_path(path))
    if not outdoor_paths:
        raise FileNotFoundError(f"No outdoor HDRIs were found under {hdri_root}.")
    return outdoor_paths


@lru_cache(maxsize=64)
def load_hdr_image(path: Path) -> np.ndarray:
    try:
        import imageio.v2 as imageio

        image = imageio.imread(path).astype(np.float32)
    except Exception:
        try:
            import cv2
        except Exception as exc:
            raise ImportError(
                "Reading HDRIs requires either imageio or cv2 to be installed in the render environment."
            ) from exc
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load HDR image from {path}")
        image = image.astype(np.float32)
        if image.ndim == 3 and image.shape[2] >= 3:
            image = image[..., :3][:, :, ::-1]

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an HDR image with at least 3 channels at {path}, got shape {image.shape}.")
    return np.nan_to_num(image[..., :3], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def tonemap_hdr_to_uint8(hdr_rgb: np.ndarray, exposure: float) -> np.ndarray:
    linear = np.clip(np.asarray(hdr_rgb, dtype=np.float32), 0.0, None)
    luminance = (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    )
    white_point = max(float(np.percentile(luminance, 99.5)), 1e-4)
    mapped = 1.0 - np.exp(-(linear * float(exposure)) / white_point)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.round(mapped * 255.0), 0.0, 255.0).astype(np.uint8)


def resize_rgb_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil_image.resize(size, resampling), dtype=np.uint8)


def compose_background(color: np.ndarray, depth: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    composed = np.asarray(color, dtype=np.uint8).copy()
    background = np.asarray(background_rgb, dtype=np.uint8)
    if background.shape[:2] != composed.shape[:2]:
        raise ValueError(
            f"Background shape {background.shape[:2]} does not match rendered image shape {composed.shape[:2]}."
        )
    empty_mask = np.asarray(depth) <= 0.0
    if composed.ndim != 3 or composed.shape[2] not in (3, 4):
        raise ValueError(f"Expected rendered color image with 3 or 4 channels, got shape {composed.shape}.")
    composed[..., :3][empty_mask] = background[..., :3][empty_mask]
    if composed.shape[2] == 4:
        composed[..., 3][empty_mask] = 255
    return composed


def jitter_rgb(base: Sequence[float], rng: np.random.Generator, amount: float) -> np.ndarray:
    rgb = np.asarray(base, dtype=np.float32) + rng.uniform(-amount, amount, size=3).astype(np.float32)
    return np.clip(rgb, 0.0, 1.0)


def jitter_rgba(base: Sequence[int], rng: np.random.Generator, amount: int) -> np.ndarray:
    rgba = np.asarray(base, dtype=np.int16)
    rgba[:3] += rng.integers(-amount, amount + 1, size=3, dtype=np.int16)
    rgba[:3] = np.clip(rgba[:3], 0, 255)
    rgba[3] = 255
    return rgba.astype(np.uint8)


def choose_floor_surface(
    scene_name: str,
    environment_seed: int,
    variation_mode: str,
) -> Tuple[str, int]:
    seed = stable_environment_seed(f"floor-surface:{scene_name}", environment_seed, variation_mode)
    rng = np.random.default_rng(seed)
    surface_name = str(rng.choice(np.asarray(FLOOR_SURFACE_STYLES, dtype=object)))
    return surface_name, int(seed % (2**31 - 1))


def choose_studio_environment(
    scene_name: str,
    environment_seed: int,
    variation_mode: str,
) -> StudioEnvironmentConfig:
    rng = np.random.default_rng(stable_environment_seed(scene_name, environment_seed, variation_mode))
    preset = STUDIO_ENVIRONMENT_PRESETS[int(rng.integers(0, len(STUDIO_ENVIRONMENT_PRESETS)))]
    floor_surface_name, floor_surface_seed = choose_floor_surface(
        scene_name=scene_name,
        environment_seed=environment_seed + 503,
        variation_mode=variation_mode,
    )
    return StudioEnvironmentConfig(
        name=preset.name,
        bg_color=np.concatenate([jitter_rgb(preset.bg_rgb, rng, 0.02), np.array([1.0], dtype=np.float32)]).astype(np.float32),
        ambient_light=jitter_rgb(preset.ambient_rgb, rng, 0.015).astype(np.float32),
        floor_rgba=jitter_rgba(preset.floor_rgba, rng, 8),
        wall_rgba=jitter_rgba(preset.wall_rgba, rng, 8),
        floor_surface_name=floor_surface_name,
        floor_surface_seed=floor_surface_seed,
        floor_drop=max(0.015, preset.floor_drop + float(rng.uniform(-0.008, 0.008))),
        floor_padding_x=max(0.55, preset.floor_padding_x + float(rng.uniform(-0.12, 0.12))),
        floor_padding_z=max(0.75, preset.floor_padding_z + float(rng.uniform(-0.12, 0.12))),
        wall_height_padding=max(0.55, preset.wall_height_padding + float(rng.uniform(-0.10, 0.10))),
        wall_width_padding=max(0.55, preset.wall_width_padding + float(rng.uniform(-0.10, 0.10))),
        backdrop_margin=max(0.10, preset.backdrop_margin + float(rng.uniform(-0.05, 0.05))),
        floor_z_shift=preset.floor_z_shift + float(rng.uniform(-0.06, 0.06)),
        wall_x_shift=preset.wall_x_shift + float(rng.uniform(-0.08, 0.08)),
    )


def choose_outdoor_hdri_environment(
    scene_name: str,
    hdri_root: Path,
    environment_seed: int,
    variation_mode: str,
) -> IndoorHDRIEnvironmentConfig:
    outdoor_hdri_paths = discover_outdoor_hdri_paths(hdri_root)
    rng = np.random.default_rng(
        stable_environment_seed(f"outdoor-hdri:{scene_name}", environment_seed, variation_mode)
    )
    hdr_path = outdoor_hdri_paths[int(rng.integers(0, len(outdoor_hdri_paths)))]
    floor_environment = choose_studio_environment(
        scene_name=f"{scene_name}:floor:{hdr_path.parent.name}",
        environment_seed=environment_seed + 1009,
        variation_mode=variation_mode,
    )
    return IndoorHDRIEnvironmentConfig(
        name=hdr_path.parent.name,
        hdr_path=hdr_path,
        rotation_frac=float(rng.uniform(0.0, 1.0)),
        vertical_bias=float(rng.uniform(-0.10, 0.08)),
        exposure=float(rng.uniform(0.95, 1.35)),
        floor_environment=floor_environment,
    )


def bilinear_sample_equirect(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.mod(u, 1.0) * float(width - 1)
    y = np.clip(v, 0.0, 1.0) * float(height - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0)[..., None].astype(np.float32)
    wy = (y - y0)[..., None].astype(np.float32)

    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def prepare_hdri_background(
    config: IndoorHDRIEnvironmentConfig,
    image_size: int,
    camera_pose: np.ndarray,
    hdri_fov_deg: float,
    hdri_blur_radius: float,
) -> np.ndarray:
    hdr_image = load_hdr_image(config.hdr_path)
    right = np.asarray(camera_pose[:3, 0], dtype=np.float32)
    up = np.asarray(camera_pose[:3, 1], dtype=np.float32)
    forward = -np.asarray(camera_pose[:3, 2], dtype=np.float32)

    half_tan = float(np.tan(np.deg2rad(hdri_fov_deg) * 0.5))
    xs = np.linspace(-half_tan, half_tan, image_size, dtype=np.float32)
    ys = np.linspace(half_tan, -half_tan, image_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_y = grid_y + np.float32(config.vertical_bias * half_tan)

    directions = (
        forward.reshape(1, 1, 3)
        + grid_x[..., None] * right.reshape(1, 1, 3)
        + grid_y[..., None] * up.reshape(1, 1, 3)
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(min=1e-6)

    azimuth = np.arctan2(directions[..., 0], directions[..., 2]) / (2.0 * np.pi)
    u = np.mod(azimuth + 0.5 + config.rotation_frac, 1.0)
    v = np.arccos(np.clip(directions[..., 1], -1.0, 1.0)) / np.pi
    sampled_hdr = bilinear_sample_equirect(hdr_image, u, v)
    background = tonemap_hdr_to_uint8(sampled_hdr, config.exposure)
    if hdri_blur_radius > 0.0:
        background = np.asarray(
            Image.fromarray(background, mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(hdri_blur_radius))),
            dtype=np.uint8,
        )
    return background


def make_colored_box(extents: Sequence[float], translation: Sequence[float], rgba: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=np.asarray(extents, dtype=np.float64))
    mesh.apply_translation(np.asarray(translation, dtype=np.float64))
    mesh.visual.face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def blend_rgba(lhs: np.ndarray, rhs: np.ndarray, amount: float) -> np.ndarray:
    mixed = (1.0 - amount) * np.asarray(lhs[:3], dtype=np.float32) + amount * np.asarray(rhs[:3], dtype=np.float32)
    return np.array(
        [
            int(np.clip(round(float(mixed[0])), 0, 255)),
            int(np.clip(round(float(mixed[1])), 0, 255)),
            int(np.clip(round(float(mixed[2])), 0, 255)),
            255,
        ],
        dtype=np.uint8,
    )


def make_floor_surface_colors(
    local_x: np.ndarray,
    local_z: np.ndarray,
    surface_name: str,
    base_rgba: np.ndarray,
    accent_rgba: np.ndarray,
    seed: int,
) -> np.ndarray:
    x = np.asarray(local_x, dtype=np.float32)
    z = np.asarray(local_z, dtype=np.float32)
    base = np.asarray(base_rgba[:3], dtype=np.float32)
    accent = np.asarray(accent_rgba[:3], dtype=np.float32)
    dark = 0.55 * base

    if surface_name == "tile":
        tiles_x = 8.0
        tiles_z = 8.0
        u = (x + 0.5) * tiles_x
        v = (z + 0.5) * tiles_z
        fu = u - np.floor(u)
        fv = v - np.floor(v)
        grout = np.minimum.reduce([fu, 1.0 - fu, fv, 1.0 - fv]) < 0.055
        ix = np.floor(u).astype(np.float32)
        iz = np.floor(v).astype(np.float32)
        tile_noise = np.mod(np.sin(ix * 12.9898 + iz * 78.233 + float(seed) * 0.013) * 43758.5453, 1.0)
        tint = 0.90 + 0.18 * tile_noise[..., None]
        color = base * tint + 0.12 * accent
        color = np.where(grout[..., None], dark, color)
    elif surface_name == "wood":
        boards = 12.0
        board_coord = (x + 0.5) * boards
        frac = board_coord - np.floor(board_coord)
        seams = np.minimum(frac, 1.0 - frac) < 0.035
        grain = (
            0.55
            + 0.20 * np.sin((z + 0.5) * 34.0 + float(seed) * 0.0017)
            + 0.10 * np.sin((z + 0.5) * 83.0 + (x + 0.5) * 7.0)
        )
        board_idx = np.floor(board_coord).astype(np.float32)
        board_tint = 0.86 + 0.20 * np.mod(np.sin(board_idx * 19.13 + float(seed) * 0.0031) * 15731.743, 1.0)
        color = (0.72 * base + 0.28 * accent) * board_tint[..., None]
        color = color * (0.88 + 0.24 * grain[..., None])
        color = np.where(seams[..., None], dark * 0.9, color)
    else:
        noise = (
            0.50
            + 0.18 * np.sin((x + 0.5) * 19.0 + float(seed) * 0.0011)
            + 0.14 * np.sin((z + 0.5) * 27.0 + float(seed) * 0.0017)
            + 0.08 * np.sin((x + z + 1.0) * 41.0)
        )
        flecks = np.mod(np.sin((x * 173.0 + z * 241.0 + float(seed) * 0.0023)) * 43758.5453, 1.0)
        color = base * (0.88 + 0.20 * noise[..., None]) + 0.08 * accent
        color = np.where((flecks > 0.955)[..., None], dark * 0.85, color)

    alpha = np.full((*x.shape, 1), 255.0, dtype=np.float32)
    return np.clip(np.concatenate([color, alpha], axis=-1), 0.0, 255.0).astype(np.uint8)


def floor_bottom_view_bounds(
    camera_pose: Optional[np.ndarray],
    floor_y: float,
    xmag: Optional[float],
    ymag: Optional[float],
) -> Optional[np.ndarray]:
    if camera_pose is None or xmag is None or ymag is None:
        return None

    pose = np.asarray(camera_pose, dtype=np.float64)
    forward = -pose[:3, 2]
    if not np.isfinite(forward).all() or forward[1] >= -1e-6:
        return None

    right = pose[:3, 0]
    up = pose[:3, 1]
    origin = pose[:3, 3]
    hits: List[np.ndarray] = []
    for sx in (-1.0, 1.0):
        ray_origin = origin + (float(sx) * float(xmag)) * right - float(ymag) * up
        travel = (float(floor_y) - float(ray_origin[1])) / float(forward[1])
        if not np.isfinite(travel) or travel <= 0.0:
            continue
        hits.append(ray_origin + travel * forward)

    if len(hits) != 2:
        return None

    points = np.stack(hits, axis=0)
    return np.array(
        [
            float(points[:, 0].min() - FLOOR_VIEW_MARGIN_X),
            float(points[:, 0].max() + FLOOR_VIEW_MARGIN_X),
            float(points[:, 2].max() + FLOOR_VIEW_MARGIN_Z),
        ],
        dtype=np.float64,
    )


def floor_camera_apron_config(
    camera_pose: Optional[np.ndarray],
    floor_y: float,
    xmag: Optional[float],
    ymag: Optional[float],
) -> Optional[FloorApronConfig]:
    if camera_pose is None or xmag is None or ymag is None:
        return None

    pose = np.asarray(camera_pose, dtype=np.float64)
    forward = -pose[:3, 2]
    if not np.isfinite(forward).all() or forward[1] >= -1e-6:
        return None

    right = pose[:3, 0]
    up = pose[:3, 1]
    origin = pose[:3, 3]
    bottom_origins = []
    for sx in (-1.0, 1.0):
        bottom_origins.append(origin + (float(sx) * float(xmag)) * right - float(ymag) * up)

    bottom_points = np.stack(bottom_origins, axis=0)
    min_bottom_y = float(bottom_points[:, 1].min())
    apron_bottom_y = min(float(floor_y) - FLOOR_APRON_DROP, min_bottom_y - FLOOR_APRON_CLEARANCE)

    hits: List[np.ndarray] = []
    for ray_origin in bottom_points:
        travel = (apron_bottom_y - float(ray_origin[1])) / float(forward[1])
        if not np.isfinite(travel) or travel <= 0.0:
            return None
        hits.append(ray_origin + travel * forward)

    points = np.stack(hits, axis=0)
    return FloorApronConfig(
        min_x=float(points[:, 0].min() - FLOOR_VIEW_MARGIN_X),
        max_x=float(points[:, 0].max() + FLOOR_VIEW_MARGIN_X),
        max_z=float(points[:, 2].max() + FLOOR_VIEW_MARGIN_Z),
        bottom_y=apron_bottom_y,
    )


def create_textured_floor_mesh(
    centered_bounds: np.ndarray,
    floor_rgba: np.ndarray,
    accent_rgba: np.ndarray,
    floor_drop: float,
    floor_padding_x: float,
    floor_padding_z: float,
    floor_z_shift: float,
    surface_name: str,
    surface_seed: int,
    camera_pose: Optional[np.ndarray] = None,
    floor_view_xmag: Optional[float] = None,
    floor_view_ymag: Optional[float] = None,
) -> trimesh.Trimesh:
    min_corner = np.asarray(centered_bounds[0], dtype=np.float64)
    max_corner = np.asarray(centered_bounds[1], dtype=np.float64)
    span = np.maximum(max_corner - min_corner, 1e-6)
    center_x = 0.5 * float(min_corner[0] + max_corner[0])
    center_z = 0.5 * float(min_corner[2] + max_corner[2])

    floor_width = max(2.6, float(span[0]) + floor_padding_x)
    floor_depth = max(3.0, float(span[2]) + floor_padding_z)
    floor_y = float(min_corner[1] - floor_drop)
    floor_z = center_z + floor_z_shift
    min_x = center_x - 0.5 * floor_width
    max_x = center_x + 0.5 * floor_width
    min_z = floor_z - 0.5 * floor_depth
    max_z = floor_z + 0.5 * floor_depth
    flat_floor_max_z = max_z
    apron_config: Optional[FloorApronConfig] = None

    view_bounds = floor_bottom_view_bounds(
        camera_pose=camera_pose,
        floor_y=floor_y,
        xmag=floor_view_xmag,
        ymag=floor_view_ymag,
    )
    if view_bounds is not None:
        min_x = min(min_x, float(view_bounds[0]))
        max_x = max(max_x, float(view_bounds[1]))
        max_z = max(max_z, float(view_bounds[2]))
    else:
        apron_config = floor_camera_apron_config(
            camera_pose=camera_pose,
            floor_y=floor_y,
            xmag=floor_view_xmag,
            ymag=floor_view_ymag,
        )
        if apron_config is not None:
            min_x = min(min_x, apron_config.min_x)
            max_x = max(max_x, apron_config.max_x)
            max_z = max(max_z, apron_config.max_z)

    center_x = 0.5 * (min_x + max_x)
    floor_z = 0.5 * (min_z + max_z)
    floor_width = max(max_x - min_x, 1e-6)
    floor_depth = max(max_z - min_z, 1e-6)

    res_x = FLOOR_GRID_RESOLUTION
    res_z = FLOOR_GRID_RESOLUTION
    xs = np.linspace(min_x, max_x, res_x, dtype=np.float32)
    zs = np.linspace(min_z, max_z, res_z, dtype=np.float32)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_y = np.full_like(grid_x, floor_y, dtype=np.float32)
    if apron_config is not None and apron_config.max_z > flat_floor_max_z + 1e-6:
        apron_depth = max(apron_config.max_z - flat_floor_max_z, 1e-6)
        apron_mix = np.clip((grid_z - flat_floor_max_z) / apron_depth, 0.0, 1.0)
        grid_y = floor_y + apron_mix * (apron_config.bottom_y - floor_y)
    local_x = (grid_x - center_x) / max(floor_width, 1e-6)
    local_z = (grid_z - floor_z) / max(floor_depth, 1e-6)

    vertices = np.stack(
        [grid_x, grid_y, grid_z],
        axis=-1,
    ).reshape(-1, 3)
    vertex_colors = make_floor_surface_colors(
        local_x=local_x,
        local_z=local_z,
        surface_name=surface_name,
        base_rgba=floor_rgba,
        accent_rgba=accent_rgba,
        seed=surface_seed,
    ).reshape(-1, 4)

    faces: List[List[int]] = []
    for row in range(res_z - 1):
        for col in range(res_x - 1):
            idx0 = row * res_x + col
            idx1 = idx0 + 1
            idx2 = idx0 + res_x
            idx3 = idx2 + 1
            faces.append([idx0, idx2, idx1])
            faces.append([idx1, idx2, idx3])

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.visual.vertex_colors = vertex_colors
    return mesh


def add_floor_plane(
    scene: pyrender.Scene,
    centered_bounds: np.ndarray,
    floor_rgba: np.ndarray,
    accent_rgba: np.ndarray,
    floor_drop: float,
    floor_padding_x: float,
    floor_padding_z: float,
    floor_z_shift: float,
    surface_name: str,
    surface_seed: int,
    camera_pose: Optional[np.ndarray] = None,
    floor_view_xmag: Optional[float] = None,
    floor_view_ymag: Optional[float] = None,
) -> None:
    floor_mesh = create_textured_floor_mesh(
        centered_bounds=centered_bounds,
        floor_rgba=floor_rgba,
        accent_rgba=accent_rgba,
        floor_drop=floor_drop,
        floor_padding_x=floor_padding_x,
        floor_padding_z=floor_padding_z,
        floor_z_shift=floor_z_shift,
        surface_name=surface_name,
        surface_seed=surface_seed,
        camera_pose=camera_pose,
        floor_view_xmag=floor_view_xmag,
        floor_view_ymag=floor_view_ymag,
    )
    scene.add(pyrender.Mesh.from_trimesh(floor_mesh, smooth=False), name="environment_floor")


def add_studio_environment(
    scene: pyrender.Scene,
    centered_bounds: np.ndarray,
    environment: StudioEnvironmentConfig,
    camera_pose: Optional[np.ndarray] = None,
    floor_view_xmag: Optional[float] = None,
    floor_view_ymag: Optional[float] = None,
) -> None:
    add_floor_plane(
        scene=scene,
        centered_bounds=centered_bounds,
        floor_rgba=environment.floor_rgba,
        accent_rgba=blend_rgba(environment.floor_rgba, environment.wall_rgba, 0.32),
        floor_drop=environment.floor_drop,
        floor_padding_x=environment.floor_padding_x,
        floor_padding_z=environment.floor_padding_z,
        floor_z_shift=environment.floor_z_shift,
        surface_name=environment.floor_surface_name,
        surface_seed=environment.floor_surface_seed,
        camera_pose=camera_pose,
        floor_view_xmag=floor_view_xmag,
        floor_view_ymag=floor_view_ymag,
    )

    min_corner = np.asarray(centered_bounds[0], dtype=np.float64)
    max_corner = np.asarray(centered_bounds[1], dtype=np.float64)
    span = np.maximum(max_corner - min_corner, 1e-3)
    center_x = 0.5 * float(min_corner[0] + max_corner[0])

    wall_width = max(2.8, float(span[0]) + environment.wall_width_padding)
    wall_height = max(2.2, float(span[1]) + environment.wall_height_padding)
    wall_z = float(min_corner[2] - environment.backdrop_margin - 0.5 * STUDIO_WALL_THICKNESS)
    wall_y = float(min_corner[1] - environment.floor_drop + 0.5 * wall_height)
    wall_mesh = make_colored_box(
        extents=(wall_width, wall_height, STUDIO_WALL_THICKNESS),
        translation=(center_x + environment.wall_x_shift, wall_y, wall_z),
        rgba=environment.wall_rgba,
    )
    scene.add(pyrender.Mesh.from_trimesh(wall_mesh, smooth=False), name="environment_backdrop")


def create_pyrender_scene(
    mesh_scene: trimesh.Scene,
    source: Path,
    bg_color: np.ndarray,
    ambient_light: Optional[np.ndarray],
    environment_style: str = "plain",
    centered_bounds: Optional[np.ndarray] = None,
    studio_environment: Optional[StudioEnvironmentConfig] = None,
    hdri_environment: Optional[IndoorHDRIEnvironmentConfig] = None,
    camera_pose: Optional[np.ndarray] = None,
    floor_view_xmag: Optional[float] = None,
    floor_view_ymag: Optional[float] = None,
) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    if environment_style == "studio":
        if centered_bounds is None:
            centered_bounds = scene_bounds(mesh_scene)
        if studio_environment is None:
            studio_environment = choose_studio_environment(
                scene_name="default",
                environment_seed=0,
                variation_mode="fixed",
            )
        add_studio_environment(
            scene,
            centered_bounds,
            studio_environment,
            camera_pose=camera_pose,
            floor_view_xmag=floor_view_xmag,
            floor_view_ymag=floor_view_ymag,
        )
    elif environment_style == "outdoor-hdri":
        if centered_bounds is None:
            centered_bounds = scene_bounds(mesh_scene)
        if hdri_environment is None:
            raise ValueError("outdoor-hdri rendering requires an HDRI environment config.")
        floor_environment = hdri_environment.floor_environment
        add_floor_plane(
            scene=scene,
            centered_bounds=centered_bounds,
            floor_rgba=floor_environment.floor_rgba,
            accent_rgba=blend_rgba(floor_environment.floor_rgba, floor_environment.wall_rgba, 0.28),
            floor_drop=floor_environment.floor_drop,
            floor_padding_x=floor_environment.floor_padding_x,
            floor_padding_z=floor_environment.floor_padding_z,
            floor_z_shift=floor_environment.floor_z_shift,
            surface_name=floor_environment.floor_surface_name,
            surface_seed=floor_environment.floor_surface_seed,
            camera_pose=camera_pose,
            floor_view_xmag=floor_view_xmag,
            floor_view_ymag=floor_view_ymag,
        )
    for node_name in mesh_scene.graph.nodes_geometry:
        transform, geom_name = mesh_scene.graph[node_name]
        geometry = sanitize_trimesh_geometry(ensure_trimesh_geometry(mesh_scene.geometry[geom_name], source).copy(), source)
        mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
        scene.add(mesh, pose=transform, name=node_name)
    return scene


def render_scene_color_depth(
    renderer: pyrender.OffscreenRenderer,
    mesh_scene: trimesh.Scene,
    source: Path,
    camera: pyrender.Camera,
    camera_pose: np.ndarray,
    light_rig: LightRig,
    environment_style: str,
    centered_bounds: np.ndarray,
    studio_environment: Optional[StudioEnvironmentConfig],
    hdri_environment: Optional[IndoorHDRIEnvironmentConfig],
    hdri_background: Optional[np.ndarray],
    shadows: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if environment_style == "studio" and studio_environment is not None:
        bg_color = studio_environment.bg_color
        ambient_light = studio_environment.ambient_light
    elif environment_style == "outdoor-hdri" and hdri_environment is not None:
        bg_color = BLACK_BG
        ambient_light = hdri_environment.floor_environment.ambient_light
    else:
        bg_color = WHITE_BG
        ambient_light = AMBIENT_LIGHT
    if shadows:
        ambient_light = np.clip(np.asarray(ambient_light, dtype=np.float32) + SHADOW_AMBIENT_BOOST, 0.0, 1.0)
    scene = create_pyrender_scene(
        mesh_scene=mesh_scene,
        source=source,
        bg_color=bg_color,
        ambient_light=ambient_light,
        environment_style=environment_style,
        centered_bounds=centered_bounds,
        studio_environment=studio_environment,
        hdri_environment=hdri_environment,
        camera_pose=camera_pose,
        floor_view_xmag=float(getattr(camera, "xmag", 0.0)),
        floor_view_ymag=float(getattr(camera, "ymag", 0.0)),
    )
    scene.add(camera, pose=camera_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=KEY_LIGHT_INTENSITY),
        pose=light_rig.key_light_pose,
    )
    for fill_pose, fill_intensity in zip(light_rig.fill_light_poses, FILL_LIGHT_INTENSITIES):
        scene.add(
            pyrender.PointLight(color=np.ones(3), intensity=float(fill_intensity)),
            pose=fill_pose,
        )
    render_flags = RenderFlags.RGBA
    if shadows:
        render_flags |= RenderFlags.SHADOWS_DIRECTIONAL
    color, depth = renderer.render(scene, flags=render_flags)
    if hdri_background is not None:
        color = compose_background(color, depth, hdri_background)
    return color, depth


def render_scene_depth(
    renderer: pyrender.OffscreenRenderer,
    mesh_scene: trimesh.Scene,
    source: Path,
    camera: pyrender.Camera,
    camera_pose: np.ndarray,
) -> np.ndarray:
    scene = create_pyrender_scene(
        mesh_scene=mesh_scene,
        source=source,
        bg_color=BLACK_BG,
        ambient_light=np.zeros(3, dtype=np.float32),
    )
    scene.add(camera, pose=camera_pose)
    _color, depth = renderer.render(scene)
    return depth


def visible_mask_from_depths(
    full_depth: np.ndarray,
    object_depth: np.ndarray,
    atol: float,
) -> np.ndarray:
    object_present = object_depth > 0.0
    full_present = full_depth > 0.0
    visible = object_present & full_present & np.isclose(
        object_depth,
        full_depth,
        atol=atol,
        rtol=MASK_DEPTH_RTOL,
    )
    return (visible.astype(np.uint8) * 255)


def save_rgb(path: Path, image_array: np.ndarray) -> None:
    rgb = image_array[..., :3] if image_array.ndim == 3 and image_array.shape[-1] > 3 else image_array
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, mask_array: np.ndarray) -> None:
    Image.fromarray(mask_array, mode="L").save(path)


def format_frame_name(frame_index: int) -> str:
    return f"{frame_index:06d}"


def render_scene_sequence(
    scene_dir: Path,
    output_root: Path,
    renderer: pyrender.OffscreenRenderer,
    camera: pyrender.Camera,
    camera_pose: np.ndarray,
    light_rig: LightRig,
    environment_style: str,
    environment_variation: str,
    environment_seed: int,
    hdri_root: Path,
    hdri_fov_deg: float,
    hdri_blur_radius: float,
    image_size: int,
    mask_depth_atol: float,
    shadows: bool,
    overwrite: bool,
    scene_bar: Optional[object] = None,
) -> None:
    metadata = load_metadata(scene_dir)
    frame_paths = collect_frame_paths(scene_dir, metadata)
    if not frame_paths:
        raise FileNotFoundError(f"No frame GLBs found in {scene_dir}")

    scene_output_dir = output_root / scene_dir.name
    frames_dir = scene_output_dir / "frames"
    masks_dir = scene_output_dir / "masks"
    render_metadata_path = scene_output_dir / "metadata.json"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = expected_dynamic_ids(metadata)
    expected_static = expected_static_ids(metadata)
    sequence_center = sequence_center_from_metadata(metadata)
    if sequence_center is None:
        sequence_center = compute_sequence_center(frame_paths)
    sequence_bounds = sequence_bounds_from_metadata(metadata)
    if sequence_bounds is None:
        centered_sequence_bounds = compute_centered_sequence_bounds(frame_paths, sequence_center)
    else:
        centered_sequence_bounds = sequence_bounds - np.asarray(sequence_center, dtype=np.float64).reshape(1, 3)
    studio_environment = (
        choose_studio_environment(
            scene_name=scene_dir.name,
            environment_seed=environment_seed,
            variation_mode=environment_variation,
        )
        if environment_style == "studio"
        else None
    )
    hdri_environment = (
        choose_outdoor_hdri_environment(
            scene_name=scene_dir.name,
            hdri_root=hdri_root,
            environment_seed=environment_seed,
            variation_mode=environment_variation,
        )
        if environment_style == "outdoor-hdri"
        else None
    )
    hdri_background = (
        prepare_hdri_background(
            hdri_environment,
            image_size=image_size,
            camera_pose=camera_pose,
            hdri_fov_deg=hdri_fov_deg,
            hdri_blur_radius=hdri_blur_radius,
        )
        if hdri_environment is not None
        else None
    )
    first_scene: Optional[trimesh.Scene] = None
    render_metadata_written = False

    for frame_index, frame_path in enumerate(frame_paths):
        frame_stem = format_frame_name(frame_index)
        frame_png_path = frames_dir / f"{frame_stem}.png"

        scene = center_scene(load_scene(frame_path), sequence_center)
        if first_scene is None:
            first_scene = scene.copy()
        dynamic_subscenes = extract_dynamic_subscenes(scene, frame_path)
        if expected_ids is not None and not dynamic_subscenes:
            raise RuntimeError(
                f"Could not recover dynamic-object nodes from {frame_path}. "
                "Current render.py expects gen.py-exported node names like dynamic0_*."
            )
        if expected_ids is None:
            dynamic_ids = sorted(dynamic_subscenes.keys())
        else:
            dynamic_ids = expected_ids

        if not dynamic_ids:
            raise RuntimeError(
                f"No dynamic objects were found in {frame_path}. "
                "Current render.py expects gen.py-exported dynamic node prefixes like dynamic0_*."
            )

        if expected_static is None and first_scene is not None:
            expected_static = extract_static_ids(first_scene)

        if not render_metadata_written:
            render_metadata = build_render_metadata(
                source_scene_dir=scene_dir,
                metadata=metadata,
                frame_paths=frame_paths,
                first_scene=first_scene,
                dynamic_ids=dynamic_ids,
                sequence_center=sequence_center,
            )
            if expected_static is not None:
                render_metadata["scene_num_parts"] = int(len(expected_static))
            render_metadata["environment_style"] = environment_style
            render_metadata["shadows"] = bool(shadows)
            render_metadata["hdri_blur_radius"] = float(hdri_blur_radius)
            if studio_environment is not None:
                render_metadata["environment_name"] = studio_environment.name
                render_metadata["floor_surface"] = studio_environment.floor_surface_name
            if hdri_environment is not None:
                render_metadata["environment_name"] = hdri_environment.name
                render_metadata["hdri_path"] = str(hdri_environment.hdr_path)
                render_metadata["floor_surface"] = hdri_environment.floor_environment.floor_surface_name
            save_render_metadata(render_metadata_path, render_metadata, overwrite=overwrite)
            render_metadata_written = True

        mask_paths = [masks_dir / f"{frame_stem}_object_{object_id + 1:03d}.png" for object_id in dynamic_ids]
        if (
            not overwrite
            and frame_png_path.is_file()
            and all(mask_path.is_file() for mask_path in mask_paths)
        ):
            if scene_bar is not None:
                scene_bar.update(1)
            continue

        color, full_depth = render_scene_color_depth(
            renderer=renderer,
            mesh_scene=scene,
            source=frame_path,
            camera=camera,
            camera_pose=camera_pose,
            light_rig=light_rig,
            environment_style=environment_style,
            centered_bounds=centered_sequence_bounds,
            studio_environment=studio_environment,
            hdri_environment=hdri_environment,
            hdri_background=hdri_background,
            shadows=shadows,
        )
        save_rgb(frame_png_path, color)

        for object_id, mask_path in zip(dynamic_ids, mask_paths):
            object_scene = dynamic_subscenes.get(object_id)
            if object_scene is None:
                mask_array = np.zeros((color.shape[0], color.shape[1]), dtype=np.uint8)
            else:
                object_depth = render_scene_depth(
                    renderer=renderer,
                    mesh_scene=object_scene,
                    source=frame_path,
                    camera=camera,
                    camera_pose=camera_pose,
                )
                mask_array = visible_mask_from_depths(
                    full_depth=full_depth,
                    object_depth=object_depth,
                    atol=mask_depth_atol,
                )
            save_mask(mask_path, mask_array)

        if scene_bar is not None:
            scene_bar.update(1)


def build_render_worker_config(args: argparse.Namespace) -> RenderWorkerConfig:
    return RenderWorkerConfig(
        output_root=args.output_root,
        image_size=args.image_size,
        camera_distance=args.camera_distance,
        camera_height=args.camera_height,
        camera_target_y=args.camera_target_y,
        ortho_xmag=args.ortho_xmag,
        ortho_ymag=args.ortho_ymag,
        znear=args.znear,
        zfar=args.zfar,
        environment_style=args.environment_style,
        environment_variation=args.environment_variation,
        environment_seed=args.environment_seed,
        hdri_root=args.hdri_root,
        hdri_fov_deg=args.hdri_fov_deg,
        hdri_blur_radius=args.hdri_blur_radius,
        mask_depth_atol=args.mask_depth_atol,
        shadows=bool(args.shadows),
        overwrite=bool(args.overwrite),
    )


def shutdown_render_worker() -> None:
    global WORKER_RENDERER

    if WORKER_RENDERER is not None:
        WORKER_RENDERER.delete()
        WORKER_RENDERER = None


def initialize_render_worker(config: RenderWorkerConfig) -> None:
    global WORKER_RENDER_CONFIG, WORKER_RENDERER, WORKER_CAMERA, WORKER_CAMERA_POSE, WORKER_LIGHT_RIG

    WORKER_RENDER_CONFIG = config
    WORKER_CAMERA_POSE = create_camera_pose(
        camera_distance=config.camera_distance,
        camera_height=config.camera_height,
        target_y=config.camera_target_y,
    )
    WORKER_LIGHT_RIG = create_light_rig(config.camera_distance, config.camera_target_y)
    WORKER_CAMERA = create_camera(
        image_size=config.image_size,
        xmag=config.ortho_xmag,
        ymag=config.ortho_ymag,
        znear=config.znear,
        zfar=config.zfar,
    )
    WORKER_RENDERER = pyrender.OffscreenRenderer(config.image_size, config.image_size)
    atexit.register(shutdown_render_worker)


def run_render_scene_worker(scene_dir: Path) -> SceneRenderResult:
    if (
        WORKER_RENDER_CONFIG is None
        or WORKER_RENDERER is None
        or WORKER_CAMERA is None
        or WORKER_CAMERA_POSE is None
        or WORKER_LIGHT_RIG is None
    ):
        raise RuntimeError("Render worker was not initialized correctly.")

    metadata = load_metadata(scene_dir)
    frame_count = len(collect_frame_paths(scene_dir, metadata))
    try:
        render_scene_sequence(
            scene_dir=scene_dir,
            output_root=WORKER_RENDER_CONFIG.output_root,
            renderer=WORKER_RENDERER,
            camera=WORKER_CAMERA,
            camera_pose=WORKER_CAMERA_POSE,
            light_rig=WORKER_LIGHT_RIG,
            environment_style=WORKER_RENDER_CONFIG.environment_style,
            environment_variation=WORKER_RENDER_CONFIG.environment_variation,
            environment_seed=WORKER_RENDER_CONFIG.environment_seed,
            hdri_root=WORKER_RENDER_CONFIG.hdri_root,
            hdri_fov_deg=WORKER_RENDER_CONFIG.hdri_fov_deg,
            hdri_blur_radius=WORKER_RENDER_CONFIG.hdri_blur_radius,
            image_size=WORKER_RENDER_CONFIG.image_size,
            mask_depth_atol=WORKER_RENDER_CONFIG.mask_depth_atol,
            shadows=WORKER_RENDER_CONFIG.shadows,
            overwrite=WORKER_RENDER_CONFIG.overwrite,
            scene_bar=None,
        )
    except Exception as exc:
        return SceneRenderResult(scene_name=scene_dir.name, frame_count=frame_count, error=str(exc))
    return SceneRenderResult(scene_name=scene_dir.name, frame_count=frame_count)


def run_serial_render(scene_dirs: Sequence[Path], args: argparse.Namespace) -> None:
    camera_pose = create_camera_pose(
        camera_distance=args.camera_distance,
        camera_height=args.camera_height,
        target_y=args.camera_target_y,
    )
    light_rig = create_light_rig(args.camera_distance, args.camera_target_y)
    camera = create_camera(
        image_size=args.image_size,
        xmag=args.ortho_xmag,
        ymag=args.ortho_ymag,
        znear=args.znear,
        zfar=args.zfar,
    )
    renderer = pyrender.OffscreenRenderer(args.image_size, args.image_size)

    try:
        with tqdm(total=len(scene_dirs), desc="Scenes", unit="scene", position=0) as all_scenes_bar:
            for scene_dir in scene_dirs:
                metadata = load_metadata(scene_dir)
                frame_count = len(collect_frame_paths(scene_dir, metadata))
                rendered_ok = False
                with tqdm(
                    total=frame_count,
                    desc=f"Scene {scene_dir.name}",
                    unit="frame",
                    leave=False,
                    position=1,
                ) as scene_bar:
                    try:
                        render_scene_sequence(
                            scene_dir=scene_dir,
                            output_root=args.output_root,
                            renderer=renderer,
                            camera=camera,
                            camera_pose=camera_pose,
                            light_rig=light_rig,
                            environment_style=args.environment_style,
                            environment_variation=args.environment_variation,
                            environment_seed=args.environment_seed,
                            hdri_root=args.hdri_root,
                            hdri_fov_deg=args.hdri_fov_deg,
                            hdri_blur_radius=args.hdri_blur_radius,
                            image_size=args.image_size,
                            mask_depth_atol=args.mask_depth_atol,
                            shadows=args.shadows,
                            overwrite=args.overwrite,
                            scene_bar=scene_bar,
                        )
                        rendered_ok = True
                    except Exception as exc:
                        tqdm.write(f"[warn] Skipping scene '{scene_dir.name}': {exc}")
                all_scenes_bar.update(1)
                all_scenes_bar.set_postfix_str(scene_dir.name)
                if rendered_ok:
                    tqdm.write(f"[ok] Rendered {scene_dir.name}")
    finally:
        renderer.delete()


def run_parallel_render(scene_dirs: Sequence[Path], args: argparse.Namespace) -> None:
    worker_config = build_render_worker_config(args)
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        mp_context=mp_context,
        initializer=initialize_render_worker,
        initargs=(worker_config,),
    ) as executor:
        future_to_scene = {executor.submit(run_render_scene_worker, scene_dir): scene_dir for scene_dir in scene_dirs}
        with tqdm(total=len(scene_dirs), desc="Scenes", unit="scene", position=0) as all_scenes_bar:
            for future in as_completed(future_to_scene):
                scene_dir = future_to_scene[future]
                try:
                    result = future.result()
                except Exception as exc:
                    tqdm.write(f"[warn] Skipping scene '{scene_dir.name}': worker failed: {exc}")
                else:
                    if result.error is not None:
                        tqdm.write(f"[warn] Skipping scene '{result.scene_name}': {result.error}")
                    else:
                        tqdm.write(f"[ok] Rendered {result.scene_name}")
                    all_scenes_bar.set_postfix_str(result.scene_name)
                all_scenes_bar.update(1)


def main() -> None:
    args = parse_args()
    scene_dirs = collect_scene_dirs(args.input_root)
    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders found in {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[info] Rendering {len(scene_dirs)} scenes with {args.num_workers} worker(s) "
        f"at {args.image_size}x{args.image_size}."
    )

    if args.num_workers == 1:
        run_serial_render(scene_dirs, args)
        return

    run_parallel_render(scene_dirs, args)


if __name__ == "__main__":
    main()

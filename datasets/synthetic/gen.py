#!/usr/bin/env python3

"""
python3 data_gen_pipe/gen.py \
  --models-path /work/berke_gokmen/data-1/3dfront/3D-FUTURE-model \
  --dynamic-objects-path /data/animesh/humanoids_scaled \
  --output-path /work/berke_gokmen/data-1/4d_scenes/glb \
  --total-scenes 250 \
  --num-frames 8 \
  --frame-stride 2 \
  --min-dynamic-objects 1 \
  --max-dynamic-objects 2 \
  --min-static-objects 2 \
  --max-static-objects 4 \
  --num-workers 8 \
  --seed 42
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix

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


WORLD_LIMIT = 0.95
WORLD_Y_MAX = 0.95
FOOTPRINT_PADDING = 0.04
DYNAMIC_OBJECT_CLEARANCE = 0.20
STATIC_OBJECT_CLEARANCE = 0.10
DYNAMIC_STATIC_CLEARANCE = 0.20
STATIC_PLACEMENT_ATTEMPTS = 32
STATIC_PLACEMENT_PASSES = 4
SCENE_SAMPLE_ATTEMPTS = 256
EPS = 1e-6
NUMPY_SEED_MODULUS = 2**32
STATIC_CENTER_CLEARANCE_X = 0.34
STATIC_CENTER_CLEARANCE_Z = 0.20
STATIC_SCALE_BOOST = 1.30
STATIC_MAX_DIMENSION_RATIO = 3.25
STATIC_MAX_HEIGHT_TO_WIDTH_RATIO = 2.25
STATIC_OBLIQUE_YAWS = (30.0, 45.0, 60.0, 120.0, 135.0, 150.0, 210.0, 225.0, 240.0, 300.0, 315.0, 330.0)
STATIC_STRONG_OBLIQUE_YAWS = (35.0, 45.0, 55.0, 125.0, 135.0, 145.0, 215.0, 225.0, 235.0, 305.0, 315.0, 325.0)
EXCLUDED_DYNAMIC_SEQUENCE_NAMES = {
    "crunch",
    "situp",
    "vampire_standingdodgebackward",
    "eve",
    "pearl",
    "mutant_",
    "swim",
    "regina",
    "remy",
    "joyfuljump",
    "headspin",
    "fallflat",
    "sitting",
    "medea",
    "doosy",
    "ganfaul",
    "drake",
    "malcolm",
    "douglas",
    "ironhead"
}

DYNAMIC_LAYOUT = {
    1: {
        "height_range": (0.68, 0.90),
        "footprint_range": (0.50, 0.74),
        "band_width": 0.82,
        "band_depth": 0.68,
        "gap": 0.10,
        "z_offsets": [0.00],
    },
    2: {
        "height_range": (0.54, 0.76),
        "footprint_range": (0.36, 0.54),
        "band_width": 1.24,
        "band_depth": 0.70,
        "gap": 0.22,
        "z_offsets": [0.00, 0.00],
    },
    3: {
        "height_range": (0.42, 0.60),
        "footprint_range": (0.24, 0.36),
        "band_width": 1.38,
        "band_depth": 0.78,
        "gap": 0.24,
        "z_offsets": [0.00, 0.00, 0.00],
    },
}


@dataclass(frozen=True)
class StaticAsset:
    name: str
    obj_path: Path
    super_category: str
    category: Optional[str] = None


@dataclass(frozen=True)
class DynamicAsset:
    name: str
    frame_paths: Tuple[Path, ...]


@dataclass(frozen=True)
class GenerationConfig:
    num_frames: int
    frame_stride: int
    min_static_objects: int
    max_static_objects: int
    output_path: Path
    overwrite: bool


@dataclass(frozen=True)
class SceneJob:
    scene_name: str
    dynamic_assets: Tuple[DynamicAsset, ...]
    seed: int


@dataclass(frozen=True)
class SceneJobResult:
    scene_name: str
    dynamic_count: int
    static_count: int
    error: Optional[str] = None


@dataclass
class PreparedDynamicSequence:
    name: str
    frames: List[trimesh.Scene]
    frame_paths: List[Path]
    start_index: int
    bounds: np.ndarray
    extents: np.ndarray


@dataclass
class PlacedDynamicSequence:
    name: str
    frames: List[trimesh.Scene]
    frame_paths: List[Path]
    start_index: int
    scale: float
    translation: np.ndarray
    bounds: np.ndarray
    footprint: Tuple[float, float, float, float]


@dataclass
class PlacedStaticObject:
    name: str
    scene: trimesh.Scene
    obj_path: Path
    super_category: str
    category: Optional[str]
    scale: float
    yaw_deg: float
    translation: np.ndarray
    bounds: np.ndarray
    footprint: Tuple[float, float, float, float]


WORKER_STATIC_ASSETS: Tuple[StaticAsset, ...] = ()
WORKER_CONFIG: Optional[GenerationConfig] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 4D scenes with 1-3 dynamic objects and collision-free static props."
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        default=Path("/work/berke_gokmen/data-1/3dfront/3D-FUTURE-model"),
        help="Directory containing 3D-FUTURE subfolders with normalized_model.obj files.",
    )
    parser.add_argument(
        "--model-info-path",
        type=Path,
        default=None,
        help="Path to the 3D-FUTURE model_info.json file. Defaults to <models-path>/model_info.json.",
    )
    parser.add_argument(
        "--dynamic-objects-path",
        type=Path,
        default=Path("/data/animesh/humanoids_scaled"),
        help="Directory containing one subfolder per dynamic object sequence.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("/work/berke_gokmen/data-1/4d_scenes_synthetic"),
        help="Where generated scene folders are written.",
    )
    parser.add_argument(
        "--num-scenes",
        "--total-scenes",
        type=int,
        dest="num_scenes",
        default=10,
        help="How many synthetic scenes to generate.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of consecutive frames to export per scene.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Stride used when sampling dynamic source frames. Default: 2.",
    )
    parser.add_argument(
        "--dynamic-pattern",
        type=str,
        default="*.glb",
        help="Glob pattern used to collect dynamic frames inside each sequence directory.",
    )
    parser.add_argument(
        "--min-dynamic-objects",
        type=int,
        default=1,
        help="Minimum number of dynamic objects per scene.",
    )
    parser.add_argument(
        "--max-dynamic-objects",
        type=int,
        default=3,
        help="Maximum number of dynamic objects per scene.",
    )
    parser.add_argument(
        "--min-static-objects",
        type=int,
        default=2,
        help="Minimum number of static objects to try to place per scene.",
    )
    parser.add_argument(
        "--max-static-objects",
        type=int,
        default=6,
        help="Maximum number of static objects to try to place per scene.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing scene directory with the same name.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of scene-generation worker processes. Default: 1.",
    )
    args = parser.parse_args()

    if args.model_info_path is None:
        args.model_info_path = args.models_path / "model_info.json"

    if args.num_scenes <= 0:
        raise ValueError("--num-scenes/--total-scenes must be positive.")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.min_dynamic_objects < 1 or args.max_dynamic_objects > 3:
        raise ValueError("Dynamic object count must stay in [1, 3].")
    if args.min_dynamic_objects > args.max_dynamic_objects:
        raise ValueError("--min-dynamic-objects cannot exceed --max-dynamic-objects.")
    if args.min_static_objects < 0:
        raise ValueError("--min-static-objects cannot be negative.")
    if args.min_static_objects > args.max_static_objects:
        raise ValueError("--min-static-objects cannot exceed --max-static-objects.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
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


def set_seed(seed: int) -> random.Random:
    normalized_seed = int(seed) % NUMPY_SEED_MODULUS
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    return random.Random(normalized_seed)


def load_allowed_model_infos(model_info_path: Path) -> dict[str, dict[str, object]]:
    if not model_info_path.is_file():
        raise FileNotFoundError(f"model_info.json not found: {model_info_path}")

    with model_info_path.open("r", encoding="utf-8") as handle:
        model_infos = json.load(handle)
    if not isinstance(model_infos, list):
        raise ValueError(f"Expected a list in {model_info_path}, got {type(model_infos).__name__}.")

    allowed_infos: dict[str, dict[str, object]] = {}
    for entry in model_infos:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("model_id")
        super_category = entry.get("super-category")
        if not isinstance(model_id, str) or not model_id:
            continue
        if super_category == "Lighting":
            continue
        allowed_infos[model_id] = entry

    if not allowed_infos:
        raise ValueError(f"No allowed model ids were found in {model_info_path}.")
    return allowed_infos


def discover_static_assets(models_path: Path, model_info_path: Path) -> List[StaticAsset]:
    if not models_path.is_dir():
        raise FileNotFoundError(f"Static model directory not found: {models_path}")
    allowed_infos = load_allowed_model_infos(model_info_path)

    assets: List[StaticAsset] = []
    for model_dir in sorted(models_path.iterdir(), key=lambda p: natural_key(p.name)):
        if not model_dir.is_dir():
            continue
        info = allowed_infos.get(model_dir.name)
        if info is None:
            continue
        obj_path = model_dir / "normalized_model.obj"
        if obj_path.exists():
            super_category = info.get("super-category")
            category = info.get("category")
            assets.append(
                StaticAsset(
                    name=model_dir.name,
                    obj_path=obj_path,
                    super_category=super_category if isinstance(super_category, str) else "",
                    category=category if isinstance(category, str) else None,
                )
            )

    if not assets:
        raise FileNotFoundError(
            "No non-Lighting static assets with normalized_model.obj were found in "
            f"{models_path} using {model_info_path}"
        )
    return assets


def discover_dynamic_assets(
    dynamic_root: Path,
    pattern: str,
    min_frames: int,
) -> List[DynamicAsset]:
    if not dynamic_root.is_dir():
        raise FileNotFoundError(f"Dynamic object directory not found: {dynamic_root}")

    assets: List[DynamicAsset] = []
    for sequence_dir in sorted(dynamic_root.iterdir(), key=lambda p: natural_key(p.name)):
        if not sequence_dir.is_dir():
            continue
        if sequence_dir.name.lower() in EXCLUDED_DYNAMIC_SEQUENCE_NAMES:
            continue
        if any(excluded in sequence_dir.name.lower() for excluded in EXCLUDED_DYNAMIC_SEQUENCE_NAMES):
            continue
        frame_paths = sorted(sequence_dir.glob(pattern), key=lambda p: natural_key(p.name))
        if len(frame_paths) < min_frames:
            continue
        try:
            load_scene(frame_paths[0])
        except Exception:
            continue
        assets.append(DynamicAsset(name=sequence_dir.name, frame_paths=tuple(frame_paths)))

    if not assets:
        raise FileNotFoundError(
            f"No dynamic sequences with at least {min_frames} frames were found in {dynamic_root}"
        )
    return assets


def material_has_loaded_texture(material: object) -> bool:
    for attr in (
        "image",
        "baseColorTexture",
        "emissiveTexture",
        "normalTexture",
        "metallicRoughnessTexture",
        "occlusionTexture",
    ):
        value = getattr(material, attr, None)
        if value is None:
            continue
        nested_image = getattr(value, "image", None)
        nested_source = getattr(value, "source", None)
        if nested_image is not None or nested_source is not None:
            return True
        return True
    return False


def scene_has_loaded_texture(scene: trimesh.Scene, source: Path) -> bool:
    for node_name in scene.graph.nodes_geometry:
        _transform, geom_name = scene.graph[node_name]
        geometry = ensure_trimesh_geometry(scene.geometry[geom_name], source)
        visual = getattr(geometry, "visual", None)
        if visual is None:
            continue
        material = getattr(visual, "material", None)
        if material is not None and material_has_loaded_texture(material):
            return True
    return False


def load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        scene = loaded
    elif isinstance(loaded, trimesh.Trimesh):
        scene = trimesh.Scene()
        scene.add_geometry(loaded, node_name=path.stem)
    else:
        raise TypeError(f"Unsupported geometry type at {path}: {type(loaded)}")
    if not scene_has_loaded_texture(scene, path):
        raise ValueError(f"Skipping {path}: no loadable texture was found.")
    return scene


def ensure_trimesh_geometry(geometry: object, source: Path) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if hasattr(geometry, "to_trimesh"):
        converted = geometry.to_trimesh()
        if isinstance(converted, trimesh.Trimesh):
            return converted
    raise TypeError(f"Geometry in {source} cannot be converted to trimesh.Trimesh: {type(geometry)}")


def append_scene_geometry(
    target_scene: trimesh.Scene,
    source_scene: trimesh.Scene,
    prefix: str,
    source: Path,
) -> None:
    base_count = len(list(target_scene.graph.nodes_geometry))
    for local_idx, node_name in enumerate(source_scene.graph.nodes_geometry):
        transform, geom_name = source_scene.graph[node_name]
        geometry = ensure_trimesh_geometry(source_scene.geometry[geom_name], source).copy()
        new_node = f"{prefix}_{base_count + local_idx:04d}"
        new_geom = f"{new_node}_geom"
        target_scene.add_geometry(
            geometry,
            node_name=new_node,
            geom_name=new_geom,
            transform=transform,
        )


def collapse_scene_to_single_mesh(
    scene: trimesh.Scene,
    source: Path,
    node_name: str,
) -> trimesh.Scene:
    meshes: List[trimesh.Trimesh] = []
    for source_node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[source_node]
        geometry = ensure_trimesh_geometry(scene.geometry[geom_name], source).copy()
        geometry.apply_transform(transform)
        meshes.append(geometry)
    if not meshes:
        raise ValueError(f"Scene from {source} has no geometry to collapse.")
    if len(meshes) == 1:
        merged = meshes[0]
    else:
        merged = trimesh.util.concatenate(meshes)
    merged_scene = trimesh.Scene()
    merged_scene.add_geometry(
        merged,
        node_name=node_name,
        geom_name=f"{node_name}_geom",
    )
    return merged_scene


def scene_bounds(scene: trimesh.Scene) -> np.ndarray:
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("Scene bounds are invalid or non-finite.")
    return bounds


def compute_global_bounds(scenes: Sequence[trimesh.Scene]) -> np.ndarray:
    mins: List[np.ndarray] = []
    maxs: List[np.ndarray] = []
    for scene in scenes:
        bounds = scene_bounds(scene)
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        raise ValueError("No scenes available to compute global bounds.")
    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    return np.stack([global_min, global_max], axis=0)


def center_translation_from_bounds(bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=np.float64)
    return -0.5 * (bounds[0] + bounds[1])


def translate_bounds(bounds: np.ndarray, translation: np.ndarray) -> np.ndarray:
    translated = np.asarray(bounds, dtype=np.float64).copy()
    translated += np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return translated


def is_cube_like_static_extents(extents: np.ndarray) -> bool:
    dims = np.asarray(extents, dtype=np.float64)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= EPS):
        return False

    max_extent = float(np.max(dims))
    min_extent = float(np.min(dims))
    max_horizontal = max(float(dims[0]), float(dims[2]), EPS)
    height = float(dims[1])
    return (
        max_extent / min_extent <= STATIC_MAX_DIMENSION_RATIO
        and height / max_horizontal <= STATIC_MAX_HEIGHT_TO_WIDTH_RATIO
    )


def choose_static_yaw(extents: np.ndarray, rng: random.Random) -> float:
    dims = np.asarray(extents, dtype=np.float64)
    horizontal_ratio = max(float(dims[0]), float(dims[2]), EPS) / max(min(float(dims[0]), float(dims[2])), EPS)
    yaw_choices = STATIC_STRONG_OBLIQUE_YAWS if horizontal_ratio >= 1.6 else STATIC_OBLIQUE_YAWS
    return float(rng.choice(yaw_choices))


def center_scene_xz_and_floor(scene: trimesh.Scene) -> trimesh.Scene:
    centered = scene.copy()
    bounds = scene_bounds(centered)
    translation = np.array(
        [
            -0.5 * (bounds[0, 0] + bounds[1, 0]),
            -bounds[0, 1],
            -0.5 * (bounds[0, 2] + bounds[1, 2]),
        ],
        dtype=np.float64,
    )
    centered.apply_translation(translation)
    return centered


def normalize_dynamic_sequence(scenes: Sequence[trimesh.Scene]) -> Tuple[List[trimesh.Scene], np.ndarray, np.ndarray]:
    global_bounds = compute_global_bounds(scenes)
    translation = np.array(
        [
            -0.5 * (global_bounds[0, 0] + global_bounds[1, 0]),
            -global_bounds[0, 1],
            -0.5 * (global_bounds[0, 2] + global_bounds[1, 2]),
        ],
        dtype=np.float64,
    )
    normalized: List[trimesh.Scene] = []
    for scene in scenes:
        scene_copy = scene.copy()
        scene_copy.apply_translation(translation)
        normalized.append(scene_copy)
    bounds = compute_global_bounds(normalized)
    extents = bounds[1] - bounds[0]
    return normalized, bounds, extents


def transform_scene(
    scene: trimesh.Scene,
    scale: float = 1.0,
    translation: Optional[np.ndarray] = None,
    yaw_deg: float = 0.0,
) -> trimesh.Scene:
    transformed = scene.copy()
    if abs(yaw_deg) > EPS:
        transformed.apply_transform(rotation_matrix(np.deg2rad(yaw_deg), [0.0, 1.0, 0.0]))
    if abs(scale - 1.0) > EPS:
        transformed.apply_scale(scale)
    if translation is not None:
        transformed.apply_translation(np.asarray(translation, dtype=np.float64))
    return transformed


def bounds_fit_world(bounds: np.ndarray) -> bool:
    return (
        bounds[0, 0] >= -WORLD_LIMIT - 1e-4
        and bounds[1, 0] <= WORLD_LIMIT + 1e-4
        and bounds[0, 1] >= -1e-4
        and bounds[1, 1] <= WORLD_Y_MAX + 1e-4
        and bounds[0, 2] >= -WORLD_LIMIT - 1e-4
        and bounds[1, 2] <= WORLD_LIMIT + 1e-4
    )


def bounds_fit_centered_world(bounds: np.ndarray) -> bool:
    return (
        bounds[0, 0] >= -WORLD_LIMIT - 1e-4
        and bounds[1, 0] <= WORLD_LIMIT + 1e-4
        and bounds[0, 1] >= -WORLD_LIMIT - 1e-4
        and bounds[1, 1] <= WORLD_LIMIT + 1e-4
        and bounds[0, 2] >= -WORLD_LIMIT - 1e-4
        and bounds[1, 2] <= WORLD_LIMIT + 1e-4
    )


def footprint_from_bounds(bounds: np.ndarray, padding: float = 0.0) -> Tuple[float, float, float, float]:
    return (
        float(bounds[0, 0] - padding),
        float(bounds[1, 0] + padding),
        float(bounds[0, 2] - padding),
        float(bounds[1, 2] + padding),
    )


def footprints_overlap(
    lhs: Tuple[float, float, float, float],
    rhs: Tuple[float, float, float, float],
) -> bool:
    return not (
        lhs[1] <= rhs[0]
        or rhs[1] <= lhs[0]
        or lhs[3] <= rhs[2]
        or rhs[3] <= lhs[2]
    )


def interval_gap(lhs_min: float, lhs_max: float, rhs_min: float, rhs_max: float) -> float:
    return max(0.0, max(lhs_min - rhs_max, rhs_min - lhs_max))


def footprints_have_clearance(
    lhs: Tuple[float, float, float, float],
    rhs: Tuple[float, float, float, float],
    min_gap: float,
) -> bool:
    x_gap = interval_gap(lhs[0], lhs[1], rhs[0], rhs[1])
    z_gap = interval_gap(lhs[2], lhs[3], rhs[2], rhs[3])

    x_overlaps = x_gap <= EPS
    z_overlaps = z_gap <= EPS
    if x_overlaps and z_overlaps:
        return False
    if x_overlaps and z_gap < min_gap:
        return False
    if z_overlaps and x_gap < min_gap:
        return False
    if x_gap > 0.0 and z_gap > 0.0 and math.hypot(x_gap, z_gap) < min_gap:
        return False
    return True


def serialize_array(array: np.ndarray) -> List[object]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return [round(float(value), 6) for value in arr.tolist()]
    return [serialize_array(value) for value in arr]


def serialize_footprint(footprint: Tuple[float, float, float, float]) -> List[float]:
    return [round(float(value), 6) for value in footprint]


def translate_footprint(
    footprint: Tuple[float, float, float, float],
    translation: np.ndarray,
) -> Tuple[float, float, float, float]:
    offset = np.asarray(translation, dtype=np.float64)
    return (
        float(footprint[0] + offset[0]),
        float(footprint[1] + offset[0]),
        float(footprint[2] + offset[2]),
        float(footprint[3] + offset[2]),
    )


def scene_signature_for_assets(dynamic_assets: Sequence[DynamicAsset]) -> Tuple[str, ...]:
    return tuple(sorted(asset.name for asset in dynamic_assets))


def scene_name_for_signature(scene_signature: Sequence[str]) -> str:
    return "__".join(scene_signature)


def scene_name_for_assets(dynamic_assets: Sequence[DynamicAsset]) -> str:
    return scene_name_for_signature(scene_signature_for_assets(dynamic_assets))


def unique_scene_name(scene_base_name: str, rng: random.Random) -> str:
    return f"{scene_base_name}_{rng.randint(0, 9999):04d}"


def shuffle_dynamic_assets(
    dynamic_assets: Sequence[DynamicAsset],
    rng: random.Random,
) -> List[DynamicAsset]:
    shuffled = list(dynamic_assets)
    rng.shuffle(shuffled)
    return shuffled


def choose_dynamic_assets(
    dynamic_assets: Sequence[DynamicAsset],
    min_dynamic_objects: int,
    max_dynamic_objects: int,
    used_scene_signatures: Sequence[Tuple[str, ...]],
    rng: random.Random,
) -> Sequence[DynamicAsset]:
    max_count = min(max_dynamic_objects, len(dynamic_assets))
    if max_count < min_dynamic_objects:
        raise ValueError("Not enough dynamic sequences to satisfy the requested dynamic object count.")

    used = set(used_scene_signatures)
    for _ in range(SCENE_SAMPLE_ATTEMPTS):
        count = rng.randint(min_dynamic_objects, max_count)
        chosen = rng.sample(list(dynamic_assets), count)
        rng.shuffle(chosen)
        if scene_signature_for_assets(chosen) not in used:
            return chosen
    raise RuntimeError("Unable to find a new dynamic-object combination for a scene.")


def build_generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        min_static_objects=args.min_static_objects,
        max_static_objects=args.max_static_objects,
        output_path=args.output_path,
        overwrite=bool(args.overwrite),
    )


def existing_scene_signatures(output_path: Path) -> set[Tuple[str, ...]]:
    if not output_path.exists():
        return set()
    signatures: set[Tuple[str, ...]] = set()
    for path in output_path.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue

        metadata_path = path / "metadata.json"
        if metadata_path.is_file():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = None
            if isinstance(metadata, dict):
                dynamic_objects = metadata.get("dynamic_objects")
                if isinstance(dynamic_objects, list):
                    names = [
                        item.get("name")
                        for item in dynamic_objects
                        if isinstance(item, dict) and isinstance(item.get("name"), str)
                    ]
                    if names:
                        signatures.add(tuple(sorted(names)))
                        continue

        fallback_name = path.name.rsplit("_", 1)[0] if re.search(r"_\d{4}$", path.name) else path.name
        signatures.add(tuple(sorted(token for token in fallback_name.split("__") if token)))
    return signatures


def plan_scene_job(
    dynamic_assets: Sequence[DynamicAsset],
    min_dynamic_objects: int,
    max_dynamic_objects: int,
    blocked_scene_signatures: Sequence[Tuple[str, ...]],
    rng: random.Random,
) -> SceneJob:
    chosen_dynamic_assets = choose_dynamic_assets(
        dynamic_assets,
        min_dynamic_objects,
        max_dynamic_objects,
        tuple(blocked_scene_signatures),
        rng,
    )
    scene_base_name = scene_name_for_assets(chosen_dynamic_assets)
    return SceneJob(
        scene_name=unique_scene_name(scene_base_name, rng),
        dynamic_assets=tuple(chosen_dynamic_assets),
        seed=rng.getrandbits(32),
    )


@lru_cache(maxsize=None)
def load_prepared_static_asset(obj_path: Path) -> Optional[Tuple[trimesh.Scene, np.ndarray]]:
    try:
        base_scene = center_scene_xz_and_floor(load_scene(obj_path))
    except Exception:
        return None

    extents = scene_bounds(base_scene)[1] - scene_bounds(base_scene)[0]
    if not is_cube_like_static_extents(extents):
        return None
    return base_scene, extents


def prepare_dynamic_sequence(
    asset: DynamicAsset,
    num_frames: int,
    frame_stride: int,
    rng: random.Random,
) -> PreparedDynamicSequence:
    required_span = 1 + (num_frames - 1) * frame_stride
    max_start = len(asset.frame_paths) - required_span
    if max_start < 0:
        raise ValueError(
            f"Dynamic asset {asset.name} has fewer than {required_span} frames required "
            f"for num_frames={num_frames} with frame_stride={frame_stride}."
        )
    start_index = rng.randint(0, max_start)
    selected_frame_paths = list(
        asset.frame_paths[start_index : start_index + required_span : frame_stride]
    )
    if len(selected_frame_paths) != num_frames:
        raise RuntimeError(
            f"Dynamic asset {asset.name} produced {len(selected_frame_paths)} frames, expected {num_frames}."
        )
    frames = [load_scene(frame_path) for frame_path in selected_frame_paths]
    normalized_frames, bounds, extents = normalize_dynamic_sequence(frames)
    merged_frames = [
        collapse_scene_to_single_mesh(
            scene=frame_scene,
            source=frame_path,
            node_name=f"{asset.name}_merged",
        )
        for frame_scene, frame_path in zip(normalized_frames, selected_frame_paths)
    ]
    if np.any(extents <= EPS):
        raise ValueError(f"Dynamic asset {asset.name} has degenerate bounds.")
    return PreparedDynamicSequence(
        name=asset.name,
        frames=merged_frames,
        frame_paths=selected_frame_paths,
        start_index=start_index,
        bounds=bounds,
        extents=extents,
    )


def choose_dynamic_scale(extents: np.ndarray, dynamic_count: int, rng: random.Random) -> float:
    layout = DYNAMIC_LAYOUT[dynamic_count]
    target_height = rng.uniform(*layout["height_range"])
    target_footprint = rng.uniform(*layout["footprint_range"])
    max_horizontal = max(float(extents[0]), float(extents[2]), EPS)
    max_extent = max(float(np.max(extents)), EPS)
    height_scale = target_height / max(float(extents[1]), EPS)
    footprint_scale = target_footprint / max_horizontal
    extent_scale = 0.94 / max_extent
    return min(height_scale, footprint_scale, extent_scale)


def place_dynamic_sequences(
    prepared_sequences: Sequence[PreparedDynamicSequence],
    rng: random.Random,
) -> List[PlacedDynamicSequence]:
    dynamic_count = len(prepared_sequences)
    layout = DYNAMIC_LAYOUT[dynamic_count]
    required_edge_gap = DYNAMIC_OBJECT_CLEARANCE + (2.0 * FOOTPRINT_PADDING)
    effective_gap = max(layout["gap"], required_edge_gap + 0.02)
    scales: List[float] = []
    for sequence in prepared_sequences:
        base_scale = choose_dynamic_scale(sequence.extents, dynamic_count, rng)
        scales.append(base_scale * rng.uniform(0.95, 1.05))

    def scaled_extents(axis: int) -> List[float]:
        return [float(sequence.extents[axis]) * scale for sequence, scale in zip(prepared_sequences, scales)]

    widths = scaled_extents(0)
    depths = scaled_extents(2)
    heights = scaled_extents(1)

    total_width = sum(widths) + effective_gap * max(0, dynamic_count - 1)
    shrink = 1.0
    if total_width > layout["band_width"]:
        shrink = min(shrink, layout["band_width"] / max(total_width, EPS))
    if depths and max(depths) > layout["band_depth"]:
        shrink = min(shrink, layout["band_depth"] / max(max(depths), EPS))
    if heights and max(heights) > WORLD_Y_MAX:
        shrink = min(shrink, WORLD_Y_MAX / max(max(heights), EPS))
    if shrink < 1.0:
        scales = [scale * shrink for scale in scales]
        widths = scaled_extents(0)
        depths = scaled_extents(2)
        heights = scaled_extents(1)
        total_width = sum(widths) + effective_gap * max(0, dynamic_count - 1)

    current_x = -0.5 * total_width
    x_positions: List[float] = []
    for width in widths:
        x_positions.append(current_x + 0.5 * width)
        current_x += width + effective_gap

    remaining_width = max(0.0, layout["band_width"] - total_width)
    group_shift = rng.uniform(-0.25 * remaining_width, 0.25 * remaining_width)
    z_offsets = list(layout["z_offsets"])
    if dynamic_count > 1:
        rng.shuffle(z_offsets)

    placed: List[PlacedDynamicSequence] = []
    footprints: List[Tuple[float, float, float, float]] = []
    for index, sequence in enumerate(prepared_sequences):
        if dynamic_count >= 3:
            z_jitter = 0.0
        elif dynamic_count == 2:
            z_jitter = rng.uniform(-0.01, 0.01)
        else:
            z_jitter = rng.uniform(-0.03, 0.03)
        translation = np.array(
            [x_positions[index] + group_shift, 0.0, z_offsets[index] + z_jitter],
            dtype=np.float64,
        )
        placed_frames = [
            transform_scene(frame, scale=scales[index], translation=translation)
            for frame in sequence.frames
        ]
        bounds = compute_global_bounds(placed_frames)
        if not bounds_fit_world(bounds):
            raise RuntimeError(f"Dynamic placement for {sequence.name} exceeded the [-1, 1] world bounds.")
        footprint = footprint_from_bounds(bounds, padding=FOOTPRINT_PADDING)
        if any(not footprints_have_clearance(footprint, other, DYNAMIC_OBJECT_CLEARANCE) for other in footprints):
            raise RuntimeError(
                f"Dynamic placement for {sequence.name} was too close to another dynamic object."
            )
        footprints.append(footprint)
        placed.append(
            PlacedDynamicSequence(
                name=sequence.name,
                frames=placed_frames,
                frame_paths=sequence.frame_paths,
                start_index=sequence.start_index,
                scale=float(scales[index]),
                translation=translation,
                bounds=bounds,
                footprint=footprint,
            )
        )

    return placed


def choose_static_target_count(
    min_static_objects: int,
    max_static_objects: int,
    dynamic_count: int,
    rng: random.Random,
) -> int:
    adjusted_max = max_static_objects - max(0, dynamic_count - 1)
    if dynamic_count >= 3:
        adjusted_max = min(adjusted_max, 2)
    adjusted_max = max(0, adjusted_max)
    if adjusted_max == 0:
        return 0
    adjusted_min = min(min_static_objects, adjusted_max)
    return rng.randint(adjusted_min, adjusted_max)


def is_sofa_asset(asset: StaticAsset) -> bool:
    super_category = asset.super_category.strip().lower()
    category = (asset.category or "").strip().lower()
    return super_category == "sofa" or any(token in category for token in ("sofa", "couch", "sectional", "loveseat"))


def choose_static_scale(
    asset: StaticAsset,
    extents: np.ndarray,
    dynamic_bounds: np.ndarray,
    dynamic_count: int,
    rng: random.Random,
) -> float:
    max_horizontal = max(float(extents[0]), float(extents[2]), EPS)
    height = max(float(extents[1]), EPS)
    max_extent = max(float(np.max(extents)), EPS)
    dynamic_height = max(float(dynamic_bounds[1, 1] - dynamic_bounds[0, 1]), EPS)
    dynamic_horizontal = max(
        float(dynamic_bounds[1, 0] - dynamic_bounds[0, 0]),
        float(dynamic_bounds[1, 2] - dynamic_bounds[0, 2]),
        EPS,
    )
    dynamic_extent = max(float(np.max(dynamic_bounds[1] - dynamic_bounds[0])), EPS)

    if is_sofa_asset(asset):
        target_height = rng.uniform(0.62, 0.86) * dynamic_height
        target_footprint = rng.uniform(0.98, 1.24) * dynamic_horizontal
        target_extent = rng.uniform(0.98, 1.22) * dynamic_extent
    elif height > 1.35 * max_horizontal:
        target_height = rng.uniform(0.74, 1.02) * dynamic_height
        target_footprint = rng.uniform(0.48, 0.72) * dynamic_horizontal
        target_extent = rng.uniform(0.60, 0.88) * dynamic_extent
    elif height < 0.55 * max_horizontal:
        target_height = rng.uniform(0.48, 0.72) * dynamic_height
        target_footprint = rng.uniform(0.68, 0.98) * dynamic_horizontal
        target_extent = rng.uniform(0.66, 0.96) * dynamic_extent
    else:
        target_height = rng.uniform(0.58, 0.84) * dynamic_height
        target_footprint = rng.uniform(0.62, 0.92) * dynamic_horizontal
        target_extent = rng.uniform(0.66, 0.94) * dynamic_extent

    crowd_scale = max(0.95, 1.0 - 0.02 * max(0, dynamic_count - 1))
    height_scale = (target_height * crowd_scale) / height
    footprint_scale = (target_footprint * crowd_scale) / max_horizontal
    extent_scale = (target_extent * crowd_scale) / max_extent
    ceiling_scale = WORLD_Y_MAX / height
    base_scale = min(height_scale, footprint_scale, extent_scale, ceiling_scale)
    return base_scale * STATIC_SCALE_BOOST


def compute_union_bounds(bounds_list: Sequence[np.ndarray]) -> np.ndarray:
    if not bounds_list:
        return np.array([[-0.2, 0.0, -0.2], [0.2, 0.2, 0.2]], dtype=np.float64)
    mins = np.min(np.stack([bounds[0] for bounds in bounds_list], axis=0), axis=0)
    maxs = np.max(np.stack([bounds[1] for bounds in bounds_list], axis=0), axis=0)
    return np.stack([mins, maxs], axis=0)


def sample_side_position(
    dynamic_bounds: np.ndarray,
    object_bounds: np.ndarray,
    asset: StaticAsset,
    preferred_side: Optional[str],
    rng: random.Random,
) -> Optional[Tuple[float, float]]:
    dynamic_center_x = 0.5 * float(dynamic_bounds[0, 0] + dynamic_bounds[1, 0])
    dynamic_center_z = 0.5 * float(dynamic_bounds[0, 2] + dynamic_bounds[1, 2])
    dynamic_half_x = 0.5 * float(dynamic_bounds[1, 0] - dynamic_bounds[0, 0])
    dynamic_half_z = 0.5 * float(dynamic_bounds[1, 2] - dynamic_bounds[0, 2])
    object_half_x = 0.5 * float(object_bounds[1, 0] - object_bounds[0, 0])
    object_half_z = 0.5 * float(object_bounds[1, 2] - object_bounds[0, 2])

    side_clearance = STATIC_CENTER_CLEARANCE_X + (0.08 if is_sofa_asset(asset) else 0.0)
    min_offset_x = dynamic_half_x + object_half_x + side_clearance
    world_min_center_x = -WORLD_LIMIT + object_half_x + 0.04
    world_max_center_x = WORLD_LIMIT - object_half_x - 0.04
    left_range = (world_min_center_x, dynamic_center_x - min_offset_x)
    right_range = (dynamic_center_x + min_offset_x, world_max_center_x)
    valid_ranges: List[Tuple[str, Tuple[float, float]]] = []
    if left_range[0] <= left_range[1]:
        valid_ranges.append(("left", left_range))
    if right_range[0] <= right_range[1]:
        valid_ranges.append(("right", right_range))
    if not valid_ranges:
        return None
    if preferred_side is not None:
        preferred_ranges = [entry for entry in valid_ranges if entry[0] == preferred_side]
        if preferred_ranges:
            chosen_side, chosen_range = preferred_ranges[0]
        else:
            chosen_side, chosen_range = rng.choice(valid_ranges)
    else:
        chosen_side, chosen_range = rng.choice(valid_ranges)
    if chosen_side == "left":
        edge_max = chosen_range[0] + 0.45 * (chosen_range[1] - chosen_range[0])
        center_x = rng.uniform(chosen_range[0], max(chosen_range[0], edge_max))
    else:
        edge_min = chosen_range[1] - 0.45 * (chosen_range[1] - chosen_range[0])
        center_x = rng.uniform(min(chosen_range[1], edge_min), chosen_range[1])

    z_band = min(0.30, max(0.18, dynamic_half_z + object_half_z + STATIC_CENTER_CLEARANCE_Z))
    min_center_z = -WORLD_LIMIT + object_half_z + 0.04
    max_center_z = WORLD_LIMIT - object_half_z - 0.04
    preferred_min_z = max(min_center_z, dynamic_center_z - z_band)
    preferred_max_z = min(max_center_z, dynamic_center_z + z_band)
    if preferred_min_z > preferred_max_z:
        return None
    z_span = preferred_max_z - preferred_min_z
    if z_span <= EPS:
        center_z = preferred_min_z
    else:
        front_start = preferred_min_z + 0.50 * z_span
        if rng.random() < 0.65 and front_start <= preferred_max_z:
            center_z = rng.uniform(front_start, preferred_max_z)
        else:
            center_z = rng.uniform(preferred_min_z, preferred_max_z)
    return center_x, center_z


def sample_back_position(
    dynamic_bounds: np.ndarray,
    object_bounds: np.ndarray,
    rng: random.Random,
) -> Optional[Tuple[float, float]]:
    dynamic_half_x = 0.5 * float(dynamic_bounds[1, 0] - dynamic_bounds[0, 0])
    object_half_x = 0.5 * float(object_bounds[1, 0] - object_bounds[0, 0])
    object_half_z = 0.5 * float(object_bounds[1, 2] - object_bounds[0, 2])

    world_min_center_x = -WORLD_LIMIT + object_half_x + 0.04
    world_max_center_x = WORLD_LIMIT - object_half_x - 0.04
    world_min_center_z = -WORLD_LIMIT + object_half_z + 0.04
    back_max_center_z = float(dynamic_bounds[0, 2] - object_half_z - max(0.08, 0.5 * STATIC_CENTER_CLEARANCE_Z))
    if world_min_center_x > world_max_center_x or world_min_center_z > back_max_center_z:
        return None

    outer_padding = max(0.05, dynamic_half_x * 0.15)
    left_range = (world_min_center_x, min(world_max_center_x, -outer_padding))
    right_range = (max(world_min_center_x, outer_padding), world_max_center_x)
    back_ranges: List[Tuple[float, float, float]] = []
    if left_range[0] <= left_range[1]:
        back_ranges.append((left_range[0], left_range[1], 0.4))
    if right_range[0] <= right_range[1]:
        back_ranges.append((right_range[0], right_range[1], 0.4))
    back_ranges.append((world_min_center_x, world_max_center_x, 0.2))
    valid_ranges = [entry for entry in back_ranges if entry[0] <= entry[1]]
    range_weights = [entry[2] for entry in valid_ranges]
    chosen_range = rng.choices(valid_ranges, weights=range_weights, k=1)[0]
    center_x = rng.uniform(chosen_range[0], chosen_range[1])
    center_z = rng.uniform(world_min_center_z, back_max_center_z)
    return center_x, center_z


def visibility_corridor_footprint(dynamic_bounds: np.ndarray) -> Tuple[float, float, float, float]:
    dynamic_half_z = 0.5 * float(dynamic_bounds[1, 2] - dynamic_bounds[0, 2])
    back_margin = max(0.08, 0.45 * dynamic_half_z + 0.04)
    front_margin = max(0.28, dynamic_half_z + STATIC_CENTER_CLEARANCE_Z + 0.08)
    return (
        float(dynamic_bounds[0, 0] - STATIC_CENTER_CLEARANCE_X),
        float(dynamic_bounds[1, 0] + STATIC_CENTER_CLEARANCE_X),
        float(dynamic_bounds[0, 2] - back_margin),
        float(dynamic_bounds[1, 2] + front_margin),
    )


def place_static_objects(
    static_assets: Sequence[StaticAsset],
    placed_dynamic: Sequence[PlacedDynamicSequence],
    target_count: int,
    rng: random.Random,
) -> List[PlacedStaticObject]:
    if target_count <= 0:
        return []

    dynamic_bounds = compute_union_bounds([dynamic.bounds for dynamic in placed_dynamic])
    dynamic_footprints = [dynamic.footprint for dynamic in placed_dynamic]
    visibility_corridor = visibility_corridor_footprint(dynamic_bounds)

    placed: List[PlacedStaticObject] = []
    occupied_dynamic_footprints = list(dynamic_footprints)
    occupied_static_footprints: List[Tuple[float, float, float, float]] = []
    pool_size = min(len(static_assets), max(target_count * 32, 160))
    candidate_assets = rng.sample(list(static_assets), pool_size)
    dynamic_center_x = 0.5 * float(dynamic_bounds[0, 0] + dynamic_bounds[1, 0])
    dynamic_back_edge = float(dynamic_bounds[0, 2])

    for asset in candidate_assets:
        if len(placed) >= target_count:
            break

        prepared_asset = load_prepared_static_asset(asset.obj_path)
        if prepared_asset is None:
            continue
        base_scene, base_extents = prepared_asset

        for _ in range(STATIC_PLACEMENT_ATTEMPTS):
            yaw_deg = choose_static_yaw(base_extents, rng)
            rotated = transform_scene(base_scene, yaw_deg=yaw_deg)
            extents = scene_bounds(rotated)[1] - scene_bounds(rotated)[0]
            if np.any(extents <= EPS):
                continue

            scale = choose_static_scale(asset, extents, dynamic_bounds, len(placed_dynamic), rng)
            if scale <= 0.02:
                continue

            scaled = transform_scene(rotated, scale=scale)
            scaled_bounds = scene_bounds(scaled)

            left_count = sum(
                1 for static_object in placed
                if 0.5 * (static_object.footprint[0] + static_object.footprint[1]) < dynamic_center_x
            )
            right_count = len(placed) - left_count
            preferred_side: Optional[str] = None
            if left_count < right_count:
                preferred_side = "left"
            elif right_count < left_count:
                preferred_side = "right"

            back_count = sum(
                1 for static_object in placed
                if 0.5 * (static_object.footprint[2] + static_object.footprint[3]) < dynamic_back_edge - 0.04
            )
            allow_back = back_count == 0 and target_count >= 2
            placement_modes: List[str] = []
            if allow_back and rng.random() < 0.35:
                placement_modes.append("back")
            placement_modes.append("side")
            if allow_back and "back" not in placement_modes:
                placement_modes.append("back")

            sampled_center: Optional[Tuple[float, float]] = None
            for placement_mode in placement_modes:
                if placement_mode == "back":
                    sampled_center = sample_back_position(dynamic_bounds, scaled_bounds, rng)
                else:
                    sampled_center = sample_side_position(
                        dynamic_bounds,
                        scaled_bounds,
                        asset,
                        preferred_side,
                        rng,
                    )
                if sampled_center is not None:
                    break
            if sampled_center is None:
                continue
            target_x, target_z = sampled_center
            bbox_center_x = 0.5 * float(scaled_bounds[0, 0] + scaled_bounds[1, 0])
            bbox_center_z = 0.5 * float(scaled_bounds[0, 2] + scaled_bounds[1, 2])
            translation = np.array(
                [target_x - bbox_center_x, 0.0, target_z - bbox_center_z],
                dtype=np.float64,
            )
            placed_scene = transform_scene(scaled, translation=translation)
            placed_bounds = scene_bounds(placed_scene)
            if not bounds_fit_world(placed_bounds):
                continue

            footprint = footprint_from_bounds(placed_bounds, padding=FOOTPRINT_PADDING)
            if footprints_overlap(footprint, visibility_corridor):
                continue
            if any(
                not footprints_have_clearance(footprint, other, DYNAMIC_STATIC_CLEARANCE)
                for other in occupied_dynamic_footprints
            ):
                continue
            if any(
                not footprints_have_clearance(footprint, other, STATIC_OBJECT_CLEARANCE)
                for other in occupied_static_footprints
            ):
                continue

            occupied_static_footprints.append(footprint)
            placed.append(
                PlacedStaticObject(
                    name=asset.name,
                    scene=placed_scene,
                    obj_path=asset.obj_path,
                    super_category=asset.super_category,
                    category=asset.category,
                    scale=float(scale),
                    yaw_deg=yaw_deg,
                    translation=translation,
                    bounds=placed_bounds,
                    footprint=footprint,
                )
            )
            break

    return placed


def place_static_objects_with_retries(
    static_assets: Sequence[StaticAsset],
    placed_dynamic: Sequence[PlacedDynamicSequence],
    target_count: int,
    required_count: int,
    rng: random.Random,
) -> List[PlacedStaticObject]:
    best_placed: List[PlacedStaticObject] = []
    for _ in range(STATIC_PLACEMENT_PASSES):
        placed = place_static_objects(static_assets, placed_dynamic, target_count, rng)
        if len(placed) > len(best_placed):
            best_placed = placed
        if len(placed) >= required_count:
            return placed
    return best_placed


def generate_scene_job(
    job: SceneJob,
    static_assets: Sequence[StaticAsset],
    config: GenerationConfig,
    progress_bar: Optional[object] = None,
) -> Tuple[int, int]:
    rng = set_seed(job.seed)
    scene_dir = config.output_path / job.scene_name
    temp_scene_dir = config.output_path / f".{job.scene_name}.tmp.{job.seed:x}"

    prepared_dynamic = [
        prepare_dynamic_sequence(asset, config.num_frames, config.frame_stride, rng)
        for asset in job.dynamic_assets
    ]
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str("dynamic prepared")

    placed_dynamic = place_dynamic_sequences(prepared_dynamic, rng)
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str("dynamic placed")

    target_static = choose_static_target_count(
        config.min_static_objects,
        config.max_static_objects,
        len(placed_dynamic),
        rng,
    )
    required_static = min(config.min_static_objects, target_static)
    placed_static = place_static_objects_with_retries(
        static_assets,
        placed_dynamic,
        target_static,
        required_static,
        rng,
    )
    if len(placed_static) < required_static:
        raise RuntimeError(
            f"Only placed {len(placed_static)} static objects but need at least {required_static}."
        )
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"static placed ({len(placed_static)}/{target_static})")

    if temp_scene_dir.exists():
        shutil.rmtree(temp_scene_dir)
    temp_scene_dir.mkdir(parents=True, exist_ok=False)
    if progress_bar is not None:
        progress_bar.update(1)
        progress_bar.set_postfix_str("exporting frames")

    try:
        write_scene(
            scene_dir=temp_scene_dir,
            scene_name=job.scene_name,
            placed_static=placed_static,
            placed_dynamic=placed_dynamic,
            num_frames=config.num_frames,
            frame_stride=config.frame_stride,
            progress_bar=progress_bar,
        )
        if scene_dir.exists():
            if config.overwrite:
                shutil.rmtree(scene_dir)
            else:
                raise FileExistsError(
                    f"Scene directory already exists and --overwrite is disabled: {scene_dir}"
                )
        temp_scene_dir.rename(scene_dir)
    except Exception:
        if temp_scene_dir.exists():
            shutil.rmtree(temp_scene_dir, ignore_errors=True)
        raise

    return len(placed_dynamic), len(placed_static)


def initialize_scene_worker(
    static_assets: Sequence[StaticAsset],
    config: GenerationConfig,
) -> None:
    global WORKER_STATIC_ASSETS, WORKER_CONFIG
    WORKER_STATIC_ASSETS = tuple(static_assets)
    WORKER_CONFIG = config


def run_scene_job(job: SceneJob) -> SceneJobResult:
    if WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration has not been initialized.")
    try:
        dynamic_count, static_count = generate_scene_job(
            job=job,
            static_assets=WORKER_STATIC_ASSETS,
            config=WORKER_CONFIG,
            progress_bar=None,
        )
    except Exception as exc:
        return SceneJobResult(
            scene_name=job.scene_name,
            dynamic_count=0,
            static_count=0,
            error=str(exc),
        )
    return SceneJobResult(
        scene_name=job.scene_name,
        dynamic_count=dynamic_count,
        static_count=static_count,
        error=None,
    )


def build_frame_scene(
    placed_static: Sequence[PlacedStaticObject],
    placed_dynamic: Sequence[PlacedDynamicSequence],
    frame_index: int,
) -> trimesh.Scene:
    scene = trimesh.Scene()
    for static_index, static_object in enumerate(placed_static):
        append_scene_geometry(
            scene,
            static_object.scene,
            prefix=f"static{static_index}",
            source=static_object.obj_path,
        )
    for dynamic_index, dynamic_object in enumerate(placed_dynamic):
        append_scene_geometry(
            scene,
            dynamic_object.frames[frame_index],
            prefix=f"dynamic{dynamic_index}",
            source=dynamic_object.frame_paths[frame_index],
        )
    return scene


def write_scene(
    scene_dir: Path,
    scene_name: str,
    placed_static: Sequence[PlacedStaticObject],
    placed_dynamic: Sequence[PlacedDynamicSequence],
    num_frames: int,
    frame_stride: int,
    progress_bar: Optional[object] = None,
) -> None:
    frame_scenes = [
        build_frame_scene(placed_static, placed_dynamic, frame_index)
        for frame_index in range(num_frames)
    ]
    sequence_bounds = compute_global_bounds(frame_scenes)
    final_translation = center_translation_from_bounds(sequence_bounds)

    metadata = {
        "scene_name": scene_name,
        "num_frames": num_frames,
        "frame_stride": frame_stride,
        "world_bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "global_centering_translation": serialize_array(final_translation),
        "dynamic_objects": [],
        "static_objects": [],
        "frames": [],
    }

    for dynamic_object in placed_dynamic:
        translated_bounds = translate_bounds(dynamic_object.bounds, final_translation)
        metadata["dynamic_objects"].append(
            {
                "name": dynamic_object.name,
                "start_index": dynamic_object.start_index,
                "frame_stride": frame_stride,
                "source_frames": [str(path) for path in dynamic_object.frame_paths],
                "scale": round(float(dynamic_object.scale), 6),
                "translation": serialize_array(dynamic_object.translation + final_translation),
                "global_bounds": serialize_array(translated_bounds),
                "footprint_xz": serialize_footprint(
                    translate_footprint(dynamic_object.footprint, final_translation)
                ),
            }
        )

    for static_object in placed_static:
        translated_bounds = translate_bounds(static_object.bounds, final_translation)
        metadata["static_objects"].append(
            {
                "name": static_object.name,
                "source_obj": str(static_object.obj_path),
                "super_category": static_object.super_category,
                "category": static_object.category,
                "scale": round(float(static_object.scale), 6),
                "yaw_deg": round(float(static_object.yaw_deg), 6),
                "translation": serialize_array(static_object.translation + final_translation),
                "bounds": serialize_array(translated_bounds),
                "footprint_xz": serialize_footprint(
                    translate_footprint(static_object.footprint, final_translation)
                ),
            }
        )

    for frame_index, frame_scene in enumerate(frame_scenes):
        frame_scene = frame_scene.copy()
        frame_scene.apply_translation(final_translation)
        frame_bounds = scene_bounds(frame_scene)
        if not bounds_fit_centered_world(frame_bounds):
            raise RuntimeError(
                f"Frame {frame_index} of scene {scene_name} exceeds the centered world bounds."
            )
        frame_name = f"frame_{frame_index:04d}_combined.glb"
        frame_path = scene_dir / frame_name
        frame_scene.export(frame_path, file_type="glb")
        metadata["frames"].append(
            {
                "frame_index": frame_index,
                "path": frame_name,
                "bounds": serialize_array(frame_bounds),
            }
        )
        if progress_bar is not None:
            progress_bar.update(1)

    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if progress_bar is not None:
        progress_bar.update(1)


def max_unique_scene_count(dynamic_asset_count: int, min_dynamic: int, max_dynamic: int) -> int:
    total = 0
    for count in range(min_dynamic, min(max_dynamic, dynamic_asset_count) + 1):
        total += math.comb(dynamic_asset_count, count)
    return total


def run_serial_generation(
    dynamic_assets: Sequence[DynamicAsset],
    static_assets: Sequence[StaticAsset],
    config: GenerationConfig,
    target_scene_count: int,
    min_dynamic_objects: int,
    max_dynamic_objects: int,
    planned_scene_signatures: set[Tuple[str, ...]],
    rng: random.Random,
) -> None:
    generated = 0
    attempts = 0
    max_attempts = target_scene_count * SCENE_SAMPLE_ATTEMPTS

    with tqdm(total=target_scene_count, desc="Scenes", unit="scene", position=0) as all_scenes_bar:
        while generated < target_scene_count:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Stopped after {attempts} attempts and generated only {generated}/{target_scene_count} scenes."
                )
            attempts += 1

            job = plan_scene_job(
                dynamic_assets,
                min_dynamic_objects,
                max_dynamic_objects,
                tuple(planned_scene_signatures),
                rng,
            )
            planned_scene_signatures.add(scene_signature_for_assets(job.dynamic_assets))

            with tqdm(
                total=config.num_frames + 5,
                desc=f"Scene {generated + 1}/{target_scene_count}: {job.scene_name}",
                unit="step",
                leave=False,
                position=1,
            ) as scene_bar:
                try:
                    dynamic_count, static_count = generate_scene_job(
                        job=job,
                        static_assets=static_assets,
                        config=config,
                        progress_bar=scene_bar,
                    )
                except Exception as exc:
                    tqdm.write(f"[warn] Skipping scene '{job.scene_name}': {exc}")
                    continue

            generated += 1
            all_scenes_bar.update(1)
            all_scenes_bar.set_postfix_str(job.scene_name)
            tqdm.write(
                f"[ok] {job.scene_name}: {dynamic_count} dynamic objects, "
                f"{static_count} static objects, {config.num_frames} frames."
            )


def run_parallel_generation(
    dynamic_assets: Sequence[DynamicAsset],
    static_assets: Sequence[StaticAsset],
    config: GenerationConfig,
    target_scene_count: int,
    min_dynamic_objects: int,
    max_dynamic_objects: int,
    planned_scene_signatures: set[Tuple[str, ...]],
    rng: random.Random,
    num_workers: int,
) -> None:
    generated = 0
    attempts = 0
    max_attempts = target_scene_count * SCENE_SAMPLE_ATTEMPTS
    pending_jobs: dict[object, SceneJob] = {}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=initialize_scene_worker,
        initargs=(tuple(static_assets), config),
    ) as executor:
        with tqdm(total=target_scene_count, desc="Scenes", unit="scene", position=0) as all_scenes_bar:
            while generated < target_scene_count:
                max_pending_jobs = min(num_workers, target_scene_count - generated)
                while len(pending_jobs) < max_pending_jobs and attempts < max_attempts:
                    try:
                        job = plan_scene_job(
                            dynamic_assets,
                            min_dynamic_objects,
                            max_dynamic_objects,
                            tuple(planned_scene_signatures),
                            rng,
                        )
                    except RuntimeError:
                        break

                    attempts += 1
                    planned_scene_signatures.add(scene_signature_for_assets(job.dynamic_assets))
                    pending_jobs[executor.submit(run_scene_job, job)] = job

                if not pending_jobs:
                    break

                completed, _ = wait(tuple(pending_jobs.keys()), return_when=FIRST_COMPLETED)
                for future in completed:
                    job = pending_jobs.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        tqdm.write(f"[warn] Skipping scene '{job.scene_name}': worker failed: {exc}")
                        continue
                    if result.error is not None:
                        tqdm.write(f"[warn] Skipping scene '{result.scene_name}': {result.error}")
                        continue

                    generated += 1
                    all_scenes_bar.update(1)
                    all_scenes_bar.set_postfix_str(result.scene_name)
                    tqdm.write(
                        f"[ok] {result.scene_name}: {result.dynamic_count} dynamic objects, "
                        f"{result.static_count} static objects, {config.num_frames} frames."
                    )
                    if generated >= target_scene_count:
                        break

    if generated < target_scene_count:
        raise RuntimeError(
            f"Stopped after {attempts} attempts and generated only {generated}/{target_scene_count} scenes."
        )


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)

    static_assets = discover_static_assets(args.models_path, args.model_info_path)
    required_dynamic_source_frames = 1 + (args.num_frames - 1) * args.frame_stride
    dynamic_assets = discover_dynamic_assets(
        args.dynamic_objects_path,
        args.dynamic_pattern,
        required_dynamic_source_frames,
    )
    dynamic_assets = shuffle_dynamic_assets(dynamic_assets, rng)
    config = build_generation_config(args)

    max_unique = max_unique_scene_count(
        len(dynamic_assets),
        args.min_dynamic_objects,
        args.max_dynamic_objects,
    )
    args.output_path.mkdir(parents=True, exist_ok=True)

    planned_scene_signatures = existing_scene_signatures(args.output_path) if not args.overwrite else set()
    available_unique = max(0, max_unique - len(planned_scene_signatures))
    target_scene_count = min(args.num_scenes, available_unique)
    if target_scene_count < args.num_scenes:
        print(
            f"[warn] Requested {args.num_scenes} scenes but only {available_unique} unused dynamic combinations exist. "
            f"Generating {target_scene_count} scenes instead."
        )
    if target_scene_count <= 0:
        return

    if args.num_workers == 1:
        run_serial_generation(
            dynamic_assets=dynamic_assets,
            static_assets=static_assets,
            config=config,
            target_scene_count=target_scene_count,
            min_dynamic_objects=args.min_dynamic_objects,
            max_dynamic_objects=args.max_dynamic_objects,
            planned_scene_signatures=planned_scene_signatures,
            rng=rng,
        )
        return

    run_parallel_generation(
        dynamic_assets=dynamic_assets,
        static_assets=static_assets,
        config=config,
        target_scene_count=target_scene_count,
        min_dynamic_objects=args.min_dynamic_objects,
        max_dynamic_objects=args.max_dynamic_objects,
        planned_scene_signatures=planned_scene_signatures,
        rng=rng,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

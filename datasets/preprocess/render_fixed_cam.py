"""
python datasets/preprocess/render_fixed_cam.py \
    --input /data/animesh/animals_scaled/ \
    --output ./data/animesh/animals_scaled_render/ \
    --workers 4 \
    --auto-exposure \
    --use-palette-color

python datasets/preprocess/render_fixed_cam.py \
    --input /data/animesh/humanoids/ \
    --output ./data/animesh/humanoids_render_new/ \
    --workers 4 \
    --auto-exposure
"""

import os
import argparse
import sys
import concurrent.futures
from typing import List, Tuple, Optional
import random

# Limit math libraries' threads per worker to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

import trimesh
import numpy as np
from PIL import Image  # type: ignore

from src.utils.render_utils import (
    render_single_view,
    compute_global_center_and_radius,
)


# ---------- Rendering defaults (brighter & safer) ----------
RADIUS = 4
IMAGE_SIZE = (2048, 2048)

# More light + ambient fill (CLI can override defaults)
DEFAULT_LIGHT_INTENSITY = 6.0
DEFAULT_NUM_ENV_LIGHTS = 3
DEFAULT_ENV_LIGHT_FRACTION = 0.45  # each fill light uses this fraction of the key light
DEFAULT_SOLID_COLOR = "0,64,255"  # muted warm tone that keeps details readable

FIT_SCALE = 2.0  # used when auto-fitting the fixed camera to a group's global radius
FOV_MIN_DEG = 35.0
FOV_MAX_DEG = 70.0
FOV_PAD_RATIO = 1.1  # multiplicative padding applied when fitting the subject within the view
ZNEAR_EPS = 0.01
ZFAR_PAD_MULT = 3.0

RGB = [
    (82, 170, 220),
    (215, 91, 78),
    (45, 136, 117), 
    (247, 172, 83),
    (124, 121, 121),
    (127, 171, 209),
    (243, 152, 101),
    (145, 204, 192),
    (150, 59, 121),
    (181, 206, 78),
    (189, 119, 149),
    (199, 193, 222),
    (200, 151, 54),
    (236, 110, 102),
    (238, 182, 212),
]

def _palette_color(index: int) -> Tuple[int, int, int]:
    if not RGB:
        raise ValueError("Color palette RGB is empty.")
    return RGB[index % len(RGB)]


def is_valid_glb(path: str) -> bool:
    """Return True if path is a top-level .glb file we should process.
    Rules: must be a file, extension .glb, and base name must not end with `_full`.
    """
    if not os.path.isfile(path):
        return False
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if ext.lower() != ".glb":
        return False
    if name.endswith("_full"):
        return False
    return True


def _load_mesh(path: str):
    # Load as-is (no per-mesh normalization); preserve motion across frames
    return trimesh.load(path, process=False)


def _min_camera_distance_for_fov(scene_radius: float) -> float:
    """Minimum camera distance that still allows the max FOV to fit the subject with padding."""
    if scene_radius <= 0.0:
        return float(RADIUS)
    tan_half_max = np.tan(np.deg2rad(FOV_MAX_DEG / 2.0))
    if tan_half_max <= 1e-6:
        return float(RADIUS)
    return float((FOV_PAD_RATIO * scene_radius) / tan_half_max)


def _compute_camera_fov(cam_radius: float, scene_radius: float) -> float:
    """Derive a clamped FOV so the subject fits with padding at the chosen radius."""
    if cam_radius <= 0.0:
        return float(FOV_MIN_DEG)
    ratio = (FOV_PAD_RATIO * scene_radius) / max(cam_radius, 1e-6)
    ratio = max(ratio, 1e-6)
    fov = np.degrees(2.0 * np.arctan(ratio))
    return float(np.clip(fov, FOV_MIN_DEG, FOV_MAX_DEG))


def _compute_depth_planes(cam_radius: float, scene_radius: float) -> Tuple[float, float]:
    """Choose znear/zfar planes that comfortably encapsulate the subject."""
    padded_extent = FOV_PAD_RATIO * scene_radius
    znear = max(ZNEAR_EPS, cam_radius - padded_extent * 1.25)
    zfar = max(znear + 1e-3, cam_radius + padded_extent * ZFAR_PAD_MULT)
    return float(znear), float(zfar)


def _prepare_group_camera(
    meshes: List[trimesh.base.Trimesh | trimesh.Scene],
) -> Tuple[np.ndarray, float, float, float]:
    """Compute a fixed camera for a group of meshes.
    Returns (global_center, camera_distance, scene_radius, fov_degrees).
    """
    center, rad = compute_global_center_and_radius(meshes)
    # Auto-fit camera radius based on global radius; keep a minimum RADIUS for stability
    cam_radius = max(
        float(RADIUS),
        float(FIT_SCALE * rad),
        _min_camera_distance_for_fov(rad),
        1e-4,
    )
    fov_deg = _compute_camera_fov(cam_radius, rad)
    return center, cam_radius, float(rad), fov_deg


# ------------------ Tone / exposure helpers ------------------

def _to_float_image(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def _from_float_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
    return Image.fromarray(arr, mode=mode)

def _apply_post_exposure(
    img: Image.Image,
    auto_exposure: bool = True,
    brightness: float = 1.0,
    gamma: float = 1.0,
) -> Image.Image:
    """Light-weight, safe brightening:
       - optional auto-exposure (percentile-based gain on luminance)
       - user brightness multiplier
       - gamma (gamma<1 brightens mid-tones)
    """
    arr = _to_float_image(img)
    has_alpha = arr.shape[-1] == 4
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] if has_alpha else None

    if auto_exposure:
        # target the 95th percentile of luma around ~0.85
        luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        p95 = float(np.percentile(luma, 95))
        if p95 > 1e-6 and p95 < 0.85:
            gain = min(3.0, 0.85 / p95)  # clamp gain to avoid overblow
            rgb = np.clip(rgb * gain, 0.0, 1.0)

    if brightness != 1.0:
        rgb = np.clip(rgb * float(brightness), 0.0, 1.0)

    if gamma != 1.0:
        # sRGB-esque gamma adjust
        rgb = np.clip(rgb, 0.0, 1.0) ** (float(gamma))

    if has_alpha:
        out = np.concatenate([rgb, alpha], axis=-1)
    else:
        out = rgb
    return _from_float_image(out)


def _apply_solid_color(
    mesh_or_scene: trimesh.base.Trimesh | trimesh.Scene,
    rgb: Tuple[int, int, int],
) -> None:
    """Paint the entire mesh/scene with a solid RGB color."""
    rgba = np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)

    def _paint_geometry(geom: trimesh.Trimesh) -> None:
        try:
            geom.visual = trimesh.visual.ColorVisuals(
                mesh=geom,
                vertex_colors=rgba,
            )
        except Exception:
            pass

    if isinstance(mesh_or_scene, trimesh.Scene):
        for geom in mesh_or_scene.geometry.values():
            _paint_geometry(geom)
    else:
        _paint_geometry(mesh_or_scene)


def _parse_rgb_triplet(spec: str) -> Tuple[int, int, int]:
    """Parse an 'R,G,B' string into a tuple of ints."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values for RGB, got: {spec!r}")
    values: List[int] = []
    for idx, part in enumerate(parts):
        if not part:
            raise ValueError(f"Empty value in RGB triplet at position {idx}: {spec!r}")
        try:
            val = int(part, 10)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"RGB component '{part}' is not an integer") from exc
        if not (0 <= val <= 255):
            raise ValueError(f"RGB component '{part}' must be between 0 and 255")
        values.append(val)
    return values[0], values[1], values[2]


def _render_with_fixed_camera(
    mesh: trimesh.base.Trimesh | trimesh.Scene,
    center: np.ndarray,
    cam_radius: float,
    scene_radius: float,
    fov_deg: float,
    light_intensity: float,
    num_env_lights: int,
    env_light_intensity: Optional[float],
    solid_color: Optional[Tuple[int, int, int]],
    auto_exposure: bool,
    brightness: float,
    gamma: float,
) -> "Image.Image":
    """Render a front view from a fixed camera distance while looking at the shared center."""
    
    if isinstance(mesh, trimesh.Scene):
        scene_or_geom = mesh.copy()
        if solid_color is not None:
            _apply_solid_color(scene_or_geom, solid_color)
    else:
        scene_or_geom = mesh.copy()
        if solid_color is not None:
            _apply_solid_color(scene_or_geom, solid_color)

    znear, zfar = _compute_depth_planes(cam_radius, scene_radius)
    target_center = np.asarray(center, dtype=np.float64).reshape(-1)
    if target_center.size != 3:
        raise ValueError("center must be a 3D vector")

    img = render_single_view(
        scene_or_geom,
        azimuth=0.0,
        elevation=0.0,
        radius=cam_radius,
        image_size=IMAGE_SIZE,
        fov=fov_deg,
        light_intensity=light_intensity,
        num_env_lights=num_env_lights,
        env_light_intensity=env_light_intensity,
        return_type="pil",
        bg_color=(1.0, 1.0, 1.0, 1.0),
        znear=znear,
        zfar=zfar,
        target=target_center,
    )
    # Post-process lift: helps a lot with dark PBR assets
    img = _apply_post_exposure(
        img,
        auto_exposure=auto_exposure,
        brightness=brightness,
        gamma=gamma,
    )
    return img


def _process_file_task(args):
    """Worker wrapper for per-file rendering inside a group.
    Returns (ok, fpath, saved_path, error_msg).
    """
    (fpath, out_root, group_name, frame_stem, center, cam_radius,
     scene_radius, fov_deg, light_intensity, num_env_lights,
     env_light_intensity, solid_color, auto_exposure, brightness, gamma) = args
    out_dir = os.path.join(out_root, group_name)
    out_path = os.path.join(out_dir, f"{frame_stem}.png")
    if os.path.isfile(out_path):
        return True, fpath, out_path, "already_exists"
    try:
        mesh = _load_mesh(fpath)
        img = _render_with_fixed_camera(
            mesh,
            center=center,
            cam_radius=cam_radius,
            scene_radius=scene_radius,
            fov_deg=fov_deg,
            light_intensity=light_intensity,
            num_env_lights=num_env_lights,
            env_light_intensity=env_light_intensity,
            solid_color=solid_color,
            auto_exposure=auto_exposure,
            brightness=brightness,
            gamma=gamma,
        )
        os.makedirs(out_dir, exist_ok=True)
        img.save(out_path)
        return True, fpath, out_path, None
    except Exception as e:
        failed_target = os.path.join(group_name, f"{frame_stem}.png")
        return False, fpath, failed_target, str(e)


def _collect_groups(input_root: str, max_files_in_group: int = 64) -> List[Tuple[str, List[str]]]:
    """Collect groups of GLB paths by their parent folder.
    Returns list of (group_name, file_paths_sorted).
    group_name is the immediate subdirectory under input_root.
    """
    groups: List[Tuple[str, List[str]]] = []
    # Scan immediate subdirectories as potential groups, with progress
    subdirs = [d for d in sorted(os.listdir(input_root)) if os.path.isdir(os.path.join(input_root, d))]
    for group in tqdm(subdirs, total=len(subdirs), desc="Collecting groups"):
        group_dir = os.path.join(input_root, group)
        # Scan files inside this group with progress
        files: List[str] = []
        dir_list = sorted(os.listdir(group_dir))
        for f in tqdm(dir_list, total=len(dir_list), desc=f"Index {group}", leave=False):
            fpath = os.path.join(group_dir, f)
            if is_valid_glb(fpath):
                files.append(fpath)
            if len(files) >= max_files_in_group:
                break
        if files:
            groups.append((group, files))
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Render GLBs with a fixed camera per group (grouped by parent folder)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/objects/scissors.glb",
        help=(
            "Path to a GLB file or a directory structured as "
            "<root>/<group>/*.glb. Files ending with '_full' are ignored."
        ),
    )
    parser.add_argument("--output", type=str, default="preprocessed_data")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (use 1 for GPU safety)",
    )

    # ------- NEW brightness/lighting controls -------
    parser.add_argument("--light-intensity", type=float, default=DEFAULT_LIGHT_INTENSITY,
                        help="Key/head/environment light intensity (higher = brighter).")
    parser.add_argument("--env-lights", type=int, default=DEFAULT_NUM_ENV_LIGHTS,
                        help="Number of soft environment lights for ambient fill.")
    parser.add_argument("--env-light-fraction", type=float, default=DEFAULT_ENV_LIGHT_FRACTION,
                        help="Relative intensity for each environment light vs. the key light.")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="Enable percentile-based auto exposure to lift dark renders.")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Post brightness multiplier (e.g., 1.1 to brighten).")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Post gamma (gamma<1 brightens mid-tones; e.g., 0.9).")
    parser.add_argument("--apply-solid-color", action="store_true",
                        help="If set, apply a uniform color to every mesh before rendering.")
    parser.add_argument("--solid-color", type=str, default=None,
                        help="RGB triplet 'R,G,B' (0-255). Used with --apply-solid-color; defaults to a warm neutral.")
    parser.add_argument("--use-palette-color", action="store_true",
                        help="Assign a unique color per group using the built-in palette.")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    key_light_intensity = max(0.0, float(args.light_intensity))
    num_env_lights = max(0, int(args.env_lights))
    env_light_fraction = max(0.0, float(args.env_light_fraction))
    env_light_intensity = None
    if num_env_lights > 0:
        env_light_intensity = key_light_intensity * env_light_fraction
        env_light_intensity = max(0.0, env_light_intensity)
    auto_exposure = bool(args.auto_exposure)
    post_brightness = float(args.brightness)
    post_gamma = float(args.gamma)
    user_solid_color: Optional[Tuple[int, int, int]] = None
    if args.solid_color:
        try:
            user_solid_color = _parse_rgb_triplet(args.solid_color)
        except ValueError as exc:
            parser.error(str(exc))
    use_palette_color = bool(args.use_palette_color)
    if args.apply_solid_color or args.solid_color:
        if user_solid_color is None:
            user_solid_color = _parse_rgb_triplet(DEFAULT_SOLID_COLOR)

    print(f"Using workers: {args.workers}")

    os.makedirs(output_path, exist_ok=True)

    if os.path.isdir(input_path):
        groups = _collect_groups(input_path)
        groups = groups[::-1]
        if not groups:
            print(
                f"No valid {input_path}/<group>/*.glb files found. Files ending with '_full' are ignored."
            )
        # Iterate groups with progress
        for group_idx, (group_name, file_paths) in enumerate(
            tqdm(groups, total=len(groups), desc="Groups")
        ):
            print(f"Processing group: {group_name} with {len(file_paths)} GLBs")
            # Load all meshes to compute global center and a camera distance
            meshes = []
            for fp in tqdm(file_paths, total=len(file_paths), desc=f"Loading {group_name}", leave=False):
                try:
                    meshes.append(_load_mesh(fp))
                except Exception as e:
                    print(f"[WARN] Failed to load {fp}: {e}")
            if not meshes:
                print(f"[WARN] Skipping empty group {group_name}")
                continue
            center, cam_radius, scene_radius, fov_deg = _prepare_group_camera(meshes)
            group_color: Optional[Tuple[int, int, int]] = None
            if user_solid_color is not None:
                group_color = user_solid_color
            elif use_palette_color:
                group_color = _palette_color(group_idx)

            # Per-file tasks share a fixed camera; outputs go under <output>/<group>/<frame>.png
            tasks = []
            skipped_existing = 0
            for fp in tqdm(file_paths, total=len(file_paths), desc=f"Queueing {group_name}", leave=False):
                stem = os.path.splitext(os.path.basename(fp))[0]
                out_path = os.path.join(output_path, group_name, f"{stem}.png")
                if os.path.isfile(out_path):
                    skipped_existing += 1
                    continue
                tasks.append(
                    (
                        fp,
                        output_path,
                        group_name,
                        stem,
                        center,
                        cam_radius,
                        scene_radius,
                        fov_deg,
                        key_light_intensity,
                        num_env_lights,
                        env_light_intensity,
                        group_color,
                        auto_exposure,
                        post_brightness,
                        post_gamma,
                    )
                )
            if skipped_existing:
                print(f"[INFO] {group_name}: skipped {skipped_existing} existing renders.")
            if not tasks:
                print(f"[INFO] All renders already exist for {group_name}; skipping group.")
                continue

            worker_count = max(1, int(args.workers))
            if worker_count == 1:
                for t in tqdm(tasks, total=len(tasks), desc=f"Rendering {group_name}"):
                    ok, fpath, saved_path, err = _process_file_task(t)
                    if err == "already_exists":
                        continue
                    if not ok:
                        print(f"[WARN] Skipping {fpath} -> {saved_path} due to error: {err}")
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
                    total = len(tasks)
                    for ok, fpath, saved_path, err in tqdm(
                        ex.map(_process_file_task, tasks),
                        total=total,
                        desc=f"Rendering {group_name}",
                        leave=True,
                    ):
                        if err == "already_exists":
                            continue
                        if not ok:
                            print(f"[WARN] Skipping {fpath} -> {saved_path} due to error: {err}")
    else:
        # Single file path: treat as a group of one
        if not is_valid_glb(input_path):
            raise ValueError(
                f"Input must be a .glb file (not ending with _full) or a directory. Got: {input_path}"
            )
        mesh_name = os.path.basename(input_path).split(".")[0]
        out_dir = os.path.join(output_path, mesh_name)
        out_path = os.path.join(out_dir, "rendering.png")
        if os.path.isfile(out_path):
            print(f"[INFO] Rendering already exists at {out_path}; skipping.")
        else:
            # Loading
            for _ in tqdm([input_path], total=1, desc="Loading mesh"):
                mesh = _load_mesh(input_path)
            # Camera prep
            for _ in tqdm([0], total=1, desc="Preparing camera", leave=False):
                center, cam_radius, scene_radius, fov_deg = _prepare_group_camera([mesh])
            single_color: Optional[Tuple[int, int, int]] = None
            if user_solid_color is not None:
                single_color = user_solid_color
            elif use_palette_color:
                single_color = _palette_color(0)
            # Rendering
            for _ in tqdm([0], total=1, desc="Rendering", leave=False):
                img = _render_with_fixed_camera(
                    mesh,
                    center=center,
                    cam_radius=cam_radius,
                    scene_radius=scene_radius,
                    fov_deg=fov_deg,
                    light_intensity=key_light_intensity,
                    num_env_lights=num_env_lights,
                    env_light_intensity=env_light_intensity,
                    solid_color=single_color,
                    auto_exposure=auto_exposure,
                    brightness=post_brightness,
                    gamma=post_gamma,
                )

            os.makedirs(out_dir, exist_ok=True)
            for _ in tqdm([0], total=1, desc="Saving", leave=False):
                img.save(out_path)

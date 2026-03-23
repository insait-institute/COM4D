from src.utils.typing_utils import *

import ctypes.util
import glob
import os


def _has_display_server() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_accessible_dri_render_node() -> bool:
    for path in sorted(glob.glob("/dev/dri/renderD*")):
        if os.access(path, os.R_OK | os.W_OK):
            return True
    return False


def _has_osmesa() -> bool:
    return ctypes.util.find_library("OSMesa") is not None


def _configure_pyopengl_platform() -> None:
    chosen = os.environ.get("PYOPENGL_PLATFORM")
    if not chosen:
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
            # Last resort for headless nodes without X/Wayland: use surfaceless EGL.
            chosen = "egl"

    if chosen:
        os.environ["PYOPENGL_PLATFORM"] = chosen
        if chosen.lower() == "egl":
            os.environ.setdefault("EGL_PLATFORM", "surfaceless")
            if not _has_accessible_dri_render_node():
                # Mesa can emit noisy permission-denied warnings while probing /dev/dri on shared nodes.
                os.environ.setdefault("EGL_LOG_LEVEL", "fatal")


_configure_pyopengl_platform()

import numpy as np
from PIL import Image
import trimesh
from trimesh.transformations import rotation_matrix
import pyrender
from diffusers.utils import export_to_video
from diffusers.utils.loading_utils import load_video
import torch
from torchvision.utils import make_grid

def render(
    scene: pyrender.Scene,
    renderer: pyrender.Renderer,
    camera: pyrender.Camera,
    pose: np.ndarray,
    light: Optional[pyrender.Light] = None,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Image.Image, Image.Image]]:
    camera_node = scene.add(camera, pose=pose)
    if light is not None:
        light_node = scene.add(light, pose=pose)
    image, depth = renderer.render(
        scene, 
        flags=flags
    )
    scene.remove_node(camera_node)
    if light is not None:
        scene.remove_node(light_node)
    if normalize_depth or return_type == 'pil':
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    if return_type == 'pil':
        image = Image.fromarray(image)
        depth = Image.fromarray(depth.astype(np.uint8))
    return image, depth

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

def create_circular_camera_positions(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    height: float = 0.0,
    center: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    # Create a list of positions for a circular camera trajectory
    # around the given axis with the given radius.
    positions = []
    axis = axis / np.linalg.norm(axis)
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)
    for i in range(num_views):
        theta = 2 * np.pi * i / num_views
        position = np.array([
            np.sin(theta) * radius,
            0.0,
            np.cos(theta) * radius
        ])
        if not np.allclose(axis, np.array([0.0, 1.0, 0.0])):
            R = rotation_matrix_from_vectors(np.array([0.0, 1.0, 0.0]), axis)
            position = R @ position
        position = position + axis * height + center
        positions.append(position)
    return positions

def create_circular_camera_poses(
    num_views: int,
    radius: float,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    height: float = 0.0,
    target: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    # Create a list of poses for a circular camera trajectory
    # around the given axis with the given radius.
    # The camera always looks at the provided target (defaults to origin).
    axis = axis / np.linalg.norm(axis)
    if target is None:
        target = np.zeros(3)
    target = np.asarray(target, dtype=float)
    positions = create_circular_camera_positions(
        num_views=num_views,
        radius=radius,
        axis=axis,
        height=height,
        center=target,
    )
    poses: List[np.ndarray] = []
    for position in positions:
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            raise ValueError("Camera position coincides with the target point.")
        forward /= forward_norm

        up_hint = axis
        up_hint = up_hint / np.linalg.norm(up_hint)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-8:
            # Fallback to canonical up vectors if the camera is aligned with the axis
            for fallback_up in (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ):
                fallback_up = fallback_up / np.linalg.norm(fallback_up)
                right = np.cross(forward, fallback_up)
                if np.linalg.norm(right) >= 1e-8:
                    up_hint = fallback_up
                    break
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            raise ValueError("Unable to compute a stable camera orientation basis.")
        right /= right_norm
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)

        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, true_up, -forward], axis=1)
        pose[:3, 3] = position
        poses.append(pose)
    return poses

def render_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    camera_height: float = 0.0,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0, 
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = None
) -> Union[
        List[Image.Image], 
        List[np.ndarray], 
        Tuple[List[Image.Image], List[Image.Image]], 
        Tuple[List[np.ndarray], List[np.ndarray]]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)

    bounds = getattr(mesh, "bounds", None)
    if bounds is not None:
        center = bounds.mean(axis=0)
    else:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)

    scene = pyrender.Scene.from_trimesh_scene(mesh)
    # Optionally set background color (default: black)
    if bg_color is not None:
        scene.bg_color = np.array(bg_color)
    light = pyrender.DirectionalLight(
        color=np.ones(3), 
        intensity=light_intensity
    ) if light_intensity is not None else None
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0]/image_size[1],
        znear=znear,
        zfar=zfar
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    camera_poses = create_circular_camera_poses(
        num_views, 
        radius, 
        axis=axis,
        height=camera_height,
        target=center,
    )

    images, depths = [], []
    for pose in camera_poses:
        image, depth = render(
            scene, renderer, camera, pose, light, 
            normalize_depth=normalize_depth,
            flags=flags,
            return_type=return_type
        )
        images.append(image)
        depths.append(depth)

    renderer.delete()

    if return_depth:
        return images, depths
    return images

def render_normal_views_around_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    num_views: int = 36,
    radius: float = 3.5,
    axis: np.ndarray = np.array([0.0, 1.0, 0.0]),
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)
) -> Union[
        List[Image.Image], 
        List[np.ndarray], 
        Tuple[List[Image.Image], List[Image.Image]], 
        Tuple[List[np.ndarray], List[np.ndarray]]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_views_around_mesh(
        mesh=mesh,
        num_views=num_views,
        radius=radius,
        axis=axis,
        camera_height=0.0,
        image_size=image_size,
        fov=fov,
        light_intensity=light_intensity,
        znear=znear,
        zfar=zfar,
        normalize_depth=normalize_depth,
        flags=flags,
        return_depth=return_depth,
        return_type=return_type,
        bg_color=bg_color,
    )

def create_camera_pose_on_sphere(
    azimuth: float = 0.0,  # in degrees
    elevation: float = 0.0,  # in degrees
    radius: float = 3.5,
) -> np.ndarray:
    """Return a camera pose located on a sphere that keeps world-up stable."""
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)

    direction = np.array([
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad),
        np.cos(elevation_rad) * np.cos(azimuth_rad),
    ])
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-8:
        raise ValueError("Camera direction is ill-defined.")
    backward = direction / direction_norm  # Points from target to camera.

    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, backward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Camera direction is too close to world_up; pick an alternate up hint.
        for up_hint in (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ):
            right = np.cross(up_hint, backward)
            right_norm = np.linalg.norm(right)
            if right_norm >= 1e-6:
                break
        else:
            raise ValueError("Failed to construct a stable camera basis.")
    right /= right_norm
    up = np.cross(backward, right)
    up /= np.linalg.norm(up)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = backward * radius
    return pose

def render_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0, # in degrees
    elevation: float = 0.0, # in degrees
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    num_env_lights: int = 0,
    env_light_intensity: Optional[float] = None,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False, 
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = None,
    target: Optional[np.ndarray] = None,
) -> Union[
        Image.Image, 
        np.ndarray, 
        Tuple[Image.Image, Image.Image], 
        Tuple[np.ndarray, np.ndarray]
    ]:
    
    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Scene(mesh)

    scene = pyrender.Scene.from_trimesh_scene(mesh)
    # Optionally set background color (default: black)
    if bg_color is not None:
        scene.bg_color = np.array(bg_color)
    light = pyrender.DirectionalLight(
        color=np.ones(3), 
        intensity=light_intensity
    ) if light_intensity is not None else None
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0]/image_size[1],
        znear=znear,
        zfar=zfar
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    target_vec: Optional[np.ndarray]
    if target is not None:
        target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if target_arr.size != 3:
            raise ValueError("target must be a 3D vector")
        target_vec = target_arr.astype(np.float64)
    else:
        target_vec = None

    camera_pose = create_camera_pose_on_sphere(
        azimuth,
        elevation,
        radius
    ).copy()
    if target_vec is not None:
        camera_pose[:3, 3] += target_vec

    if num_env_lights > 0:
        env_intensity = env_light_intensity if env_light_intensity is not None else light_intensity
        if env_intensity is not None and env_intensity > 0.0:
            env_light_poses = create_circular_camera_poses(
                num_env_lights,
                radius,
                axis = np.array([0.0, 1.0, 0.0])
            )
            for pose in env_light_poses:
                pose_to_add = pose.copy()
                if target_vec is not None:
                    pose_to_add[:3, 3] += target_vec
                scene.add(pyrender.DirectionalLight(
                    color=np.ones(3),
                    intensity=env_intensity
                ), pose=pose_to_add)
            # set light to None
            light = None
        else:
            # No usable env light intensity -> skip env lights entirely
            num_env_lights = 0

    image, depth = render(
        scene, renderer, camera, camera_pose, light,
        normalize_depth=normalize_depth,
        flags=flags,
        return_type=return_type
    )
    renderer.delete()

    if return_depth:
        return image, depth
    return image

def render_normal_single_view(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    azimuth: float = 0.0, # in degrees
    elevation: float = 0.0, # in degrees
    radius: float = 3.5,
    image_size: tuple = (512, 512),
    fov: float = 40.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 10.0,
    normalize_depth: bool = False,
    flags: int = pyrender.constants.RenderFlags.NONE,
    return_depth: bool = False,
    return_type: Literal['pil', 'ndarray'] = 'pil',
    bg_color: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 1.0)
) -> Union[
        Image.Image,
        np.ndarray,
        Tuple[Image.Image, Image.Image],
        Tuple[np.ndarray, np.ndarray]
    ]:

    if not isinstance(mesh, (trimesh.Trimesh, trimesh.Scene)):
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene object")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=colors
    )
    mesh = trimesh.Scene(mesh)
    return render_single_view(
        mesh, azimuth, elevation, radius, 
        image_size, fov, light_intensity, znear, zfar,
        normalize_depth, flags, 
        return_depth, return_type,
        bg_color
    )

def export_renderings(
    images: List[Image.Image],
    export_path: str,
    fps: int = 36,
    loop: int = 0,
    frame_durations: Optional[List[int]] = None,
):
    export_type = export_path.split('.')[-1]
    if export_type == 'mp4':
        export_to_video(
            images,
            export_path,
            fps=fps,
        )
    elif export_type == 'gif':
        if frame_durations is not None:
            # Use per-frame durations if provided (milliseconds per frame)
            durations = [int(max(1, d)) for d in frame_durations]
            if len(durations) != len(images):
                raise ValueError(
                    f"frame_durations length ({len(durations)}) must match images length ({len(images)})"
                )
            images[0].save(
                export_path,
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=loop,
            )
        else:
            duration = int(max(1, round(1000 / fps)))
            images[0].save(
                export_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop,
            )
    else:
        raise ValueError(f'Unknown export type: {export_type}')
    

# ==========================
# Fixed-camera rendering util
# ==========================
def _mesh_bounds(mesh: Union[trimesh.Trimesh, trimesh.Scene]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, trimesh.Scene):
        bounds = mesh.bounds
    else:
        bounds = mesh.bounds
    return bounds[0].astype(float), bounds[1].astype(float)

def compute_global_center_and_radius(meshes: List[Union[trimesh.Trimesh, trimesh.Scene]]) -> Tuple[np.ndarray, float]:
    """Compute the union AABB center and a radius (half-diagonal) across meshes.
    Returns (center, radius).
    """
    if len(meshes) == 0:
        raise ValueError("compute_global_center_and_radius: empty meshes list")
    mins = []
    maxs = []
    for m in meshes:
        mn, mx = _mesh_bounds(m)
        mins.append(mn)
        maxs.append(mx)
    mn = np.minimum.reduce(mins)
    mx = np.maximum.reduce(maxs)
    center = (mn + mx) / 2.0
    radius = 0.5 * np.linalg.norm(mx - mn)
    # Avoid zero radius
    radius = float(max(radius, 1e-4))
    return center, radius

def render_sequence_fixed_camera(
    meshes: List[Union[trimesh.Trimesh, trimesh.Scene]],
    azimuth: float = 0.0,
    elevation: float = 0.0,
    distance: Optional[float] = None,
    fit_scale: float = 2.0,
    image_size: tuple = (512, 512),
    fov: float = 55.0,
    light_intensity: Optional[float] = 5.0,
    znear: float = 0.1,
    zfar: float = 100.0,
    flags: int = pyrender.constants.RenderFlags.NONE,
    bg_color: Optional[Tuple[float, float, float, float]] = None,
    return_type: Literal['pil', 'ndarray'] = 'pil',
) -> List[Union[Image.Image, np.ndarray]]:
    """Render a list of meshes from a single, fixed camera across frames.
    - Computes a global center/radius across all meshes
    - Uses the same camera pose for every frame
    - Translates each mesh so the global center is at the origin, preserving relative motion
    """
    if len(meshes) == 0:
        return []

    center, rad = compute_global_center_and_radius(meshes)
    cam_radius = float(distance) if (distance is not None and distance > 0) else float(fit_scale * rad)
    cam_radius = max(cam_radius, 1e-4)

    # Build renderer, camera, and optional light once
    camera = pyrender.PerspectiveCamera(
        yfov=np.deg2rad(fov),
        aspectRatio=image_size[0] / image_size[1],
        znear=znear,
        zfar=zfar,
    )
    renderer = pyrender.OffscreenRenderer(*image_size)

    # Prepare fixed camera pose looking at origin, placed on sphere at distance
    cam_pose = create_camera_pose_on_sphere(azimuth=azimuth, elevation=elevation, radius=cam_radius)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=light_intensity) if light_intensity is not None else None

    out_frames: List[Union[Image.Image, np.ndarray]] = []
    for m in meshes:
        # Translate mesh so that global center appears at origin (fixed camera sees consistent world)
        if isinstance(m, trimesh.Scene):
            scene_trimesh = m.copy()
            scene_trimesh.apply_translation(-center)
        else:
            geom = m.copy()
            geom.apply_translation(-center)
            scene_trimesh = trimesh.Scene(geom)

        scene = pyrender.Scene.from_trimesh_scene(scene_trimesh)
        if bg_color is not None:
            scene.bg_color = np.array(bg_color)

        img, _ = render(scene, renderer, camera, cam_pose, light, normalize_depth=False, flags=flags, return_type=return_type)
        out_frames.append(img)

    renderer.delete()
    return out_frames

def make_grid_for_images_or_videos(
    images_or_videos: Union[List[Image.Image], List[List[Image.Image]]],
    nrow: int = 4, 
    padding: int = 0, 
    pad_value: int = 0, 
    image_size: tuple = (512, 512),
    return_type: Literal['pil', 'ndarray'] = 'pil'
) -> Union[Image.Image, List[Image.Image], np.ndarray]:
    if isinstance(images_or_videos[0], Image.Image):
        images = [np.array(image.resize(image_size).convert('RGB')) for image in images_or_videos]
        images = np.stack(images, axis=0).transpose(0, 3, 1, 2) # [N, C, H, W]
        images = torch.from_numpy(images)
        image_grid = make_grid(
            images,
            nrow=nrow,
            padding=padding,
            pad_value=pad_value,
            normalize=False
        ) # [C, H', W']
        image_grid = image_grid.cpu().numpy()
        if return_type == 'pil':
            image_grid = Image.fromarray(image_grid.transpose(1, 2, 0))
        return image_grid
    elif isinstance(images_or_videos[0], list) and isinstance(images_or_videos[0][0], Image.Image):
        image_grids = []
        for i in range(len(images_or_videos[0])):
            images = [video[i] for video in images_or_videos]
            image_grid = make_grid_for_images_or_videos(
                images,
                nrow=nrow,
                padding=padding,
                return_type=return_type
            )
            image_grids.append(image_grid)
        if return_type == 'ndarray':
            image_grids = np.stack(image_grids, axis=0)
        return image_grids
    else:
        raise ValueError(f'Unknown input type: {type(images_or_videos[0])}')

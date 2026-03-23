#!/usr/bin/env python3
"""
Export every animation frame of an FBX file as an individual GLB file.

Each exported GLB contains a static mesh with the frame pose baked into the
geometry (no armature or animation curves).
Run this script with Blender's Python interpreter, for example:

blender -b --python datasets/preprocess/fbx_to_glb.py -- \
    --fbx DeformingThings4D/humanoids/AJ_BigStomachHit/AJ_BigStomachHit.fbx \
    --output-dir ./test_fbx \
    --end-frame 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import bpy


class SimpleProgressBar:
    """Minimal stdout progress indicator that works in Blender's console."""

    def __init__(self, total: int, label: str = "") -> None:
        self.total = max(1, total)
        self.label = label
        self._last_line_length = 0
        self._finished = False

    def update(self, value: int) -> None:
        value = min(max(value, 0), self.total)
        ratio = value / self.total
        bar_width = 30
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"{self.label} [{bar}] {value}/{self.total}"
        padding = " " * max(0, self._last_line_length - len(line))
        sys.stdout.write("\r" + line + padding)
        sys.stdout.flush()
        self._last_line_length = len(line)
        if value >= self.total and not self._finished:
            self.finish()

    def finish(self) -> None:
        if self._finished:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._finished = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export each animation frame of an FBX to posed GLB files."
    )
    parser.add_argument("--fbx", required=True, type=Path, help="Source FBX file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory that will receive per-frame GLBs.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame to export. Defaults to the imported animation start.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to export. Defaults to the imported animation end.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Only export every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--export-hidden",
        action="store_true",
        help="Include hidden objects in the export.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the per-frame progress bar.",
    )

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    return parser.parse_args(argv)


def reset_scene() -> None:
    """Start from Blender's factory defaults to avoid residual data."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_fbx(fbx_path: Path) -> None:
    if not fbx_path.exists():
        raise FileNotFoundError(f"FBX file not found: {fbx_path}")

    bpy.ops.import_scene.fbx(filepath=str(fbx_path), automatic_bone_orientation=True)
    print(f"Imported FBX: {fbx_path}", flush=True)


def should_export_object(
    obj: bpy.types.Object, include_hidden: bool, view_layer: bpy.types.ViewLayer
) -> bool:
    if obj.type != "MESH":
        return False
    if include_hidden:
        return True
    if obj.hide_get():
        return False
    if getattr(obj, "hide_viewport", False):
        return False
    if getattr(obj, "hide_render", False):
        return False
    if not obj.visible_get(view_layer=view_layer):
        return False
    return True


def create_baked_mesh_copies(
    frame: int, include_hidden: bool
) -> tuple[List[bpy.types.Object], List[bpy.types.Mesh], bpy.types.Collection]:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    depsgraph = bpy.context.evaluated_depsgraph_get()

    temp_collection = bpy.data.collections.new(f"__frame_export_{frame:04d}")
    scene.collection.children.link(temp_collection)

    temp_objects: List[bpy.types.Object] = []
    temp_meshes: List[bpy.types.Mesh] = []

    for i in range(len(scene.objects)):
        obj = scene.objects[i]

        if not should_export_object(obj, include_hidden, view_layer):
            continue

        evaluated_obj = obj.evaluated_get(depsgraph)
        mesh_name = f"{obj.name}_frame_{frame:04d}"

        mesh: bpy.types.Mesh | None = None

        try:
            temp_mesh = evaluated_obj.to_mesh(
                preserve_all_data_layers=True, depsgraph=depsgraph
            )
        except RuntimeError:
            temp_mesh = None

        if temp_mesh is not None:
            mesh = temp_mesh.copy()
            mesh.name = mesh_name
            evaluated_obj.to_mesh_clear()
        else:
            mesh = bpy.data.meshes.new_from_object(
                evaluated_obj,
                preserve_all_data_layers=True,
                depsgraph=depsgraph,
            )
            mesh.name = mesh_name

        temp_meshes.append(mesh)

        mesh_obj = bpy.data.objects.new(mesh_name, mesh)
        mesh_obj.matrix_world = obj.matrix_world.copy()
        temp_collection.objects.link(mesh_obj)
        temp_objects.append(mesh_obj)

    return temp_objects, temp_meshes, temp_collection


def cleanup_temp_objects(
    temp_objects: List[bpy.types.Object],
    temp_meshes: List[bpy.types.Mesh],
    temp_collection: bpy.types.Collection,
) -> None:
    for obj in temp_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    if temp_collection.name in bpy.data.collections:
        bpy.data.collections.remove(temp_collection)

    for mesh in temp_meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def export_frames(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    start = args.start_frame if args.start_frame is not None else scene.frame_start
    end = args.end_frame if args.end_frame is not None else scene.frame_end
    step = max(1, args.frame_step)

    if end < start:
        raise ValueError(f"End frame {end} is before start frame {start}.")

    target_dir = args.output_dir.resolve() / args.fbx.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    total_frames = ((end - start) // step) + 1
    progress = None if args.no_progress else SimpleProgressBar(total_frames, args.fbx.stem)

    exported = 0
    
    print(f"Exporting frames {start} to {end} (step {step})...", flush=True)

    for index, frame in enumerate(range(start, end + 1, step), start=1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        out_path = target_dir / f"frame_{frame:04d}.glb"

        print(f"Exporting frame {frame} to {out_path}...", flush=True)
        temp_objects, temp_meshes, temp_collection = create_baked_mesh_copies(
            frame, args.export_hidden
        )
        print(f"  Created {len(temp_objects)} temporary objects for export.", flush=True)

        try:
            if temp_objects:
                if bpy.ops.object.mode_set.poll():
                    bpy.ops.object.mode_set(mode="OBJECT")

                bpy.ops.object.select_all(action="DESELECT")
                for temp_obj in temp_objects:
                    temp_obj.select_set(True)
                bpy.context.view_layer.objects.active = temp_objects[0]

                bpy.ops.export_scene.gltf(
                    filepath=str(out_path),
                    export_format="GLB",
                    export_current_frame=True,
                    export_animations=False,
                    use_selection=True,
                    use_visible=True,
                    export_cameras=False,
                    export_lights=False,
                    export_skins=False,
                    export_bake_animation=False,
                )
                exported += 1
        finally:
            cleanup_temp_objects(temp_objects, temp_meshes, temp_collection)

        if progress:
            progress.update(index)

    if progress:
        progress.finish()

    print(
        f"{args.fbx.name}: exported {exported} frame(s) to {target_dir}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.fbx = args.fbx.resolve()
    args.output_dir = args.output_dir.resolve()

    reset_scene()
    import_fbx(args.fbx)
    export_frames(args)


if __name__ == "__main__":
    main()

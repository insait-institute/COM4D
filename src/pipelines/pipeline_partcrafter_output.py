from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
from diffusers.utils import BaseOutput
from PIL import Image


@dataclass
class PartCrafterPipelineOutput(BaseOutput):
    r"""
    Output class for ShapeDiff pipelines.
    """

    samples: torch.Tensor
    meshes: List[trimesh.Trimesh]
    # Optional: carry denoised latents for downstream processing (e.g., field warping)
    latents: Optional[torch.Tensor] = None
    time: Optional[float] = None


@dataclass
class PartCrafter3D4DOutput(BaseOutput):
    scene_meshes: List[trimesh.Trimesh]
    static_meshes_per_frame: List[List[trimesh.Trimesh]]
    dynamic_meshes: List[List[trimesh.Trimesh]]
    animation_path: Optional[str] = None
    scene_renders: Optional[List[Image.Image]] = None
    time: Optional[float] = None
    
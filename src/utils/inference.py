import os
from glob import glob
from typing import List, Optional
from pathlib import Path

import torch
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers import ModelMixin
from diffusers.image_processor import PipelineImageInput
import numpy as np
import PIL
import PIL.Image
import PIL.ImageFilter

from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)

def _resolve_repo_or_dir(repo_or_dir: str, cache_dir: Path) -> Path:
    if os.path.isdir(repo_or_dir):
        return Path(repo_or_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / Path(repo_or_dir).name
    snapshot_download(repo_id=repo_or_dir, local_dir=str(local_dir))
    return local_dir


def _load_image_sequence(
    input_dir: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
) -> List[Image.Image]:
    if image_paths is None:
        assert input_dir is not None, "Either --input_dir or --image_paths must be provided"
        # common extensions
        exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        all_paths = [
            os.path.join(input_dir, f)
            for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
    else:
        all_paths = image_paths
    if len(all_paths) == 0:
        raise RuntimeError("No images found for the input sequence")
    frames: List[Image.Image] = []
    for p in all_paths:
        im = Image.open(p)
        # In case of animated formats (e.g., webp), take first frame
        if getattr(im, "is_animated", False):
            try:
                im.seek(0)
            except Exception:
                pass
        im = im.convert("RGB")
        frames.append(im)
    return frames


def _gather_images(folder: Path) -> List[Image.Image]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    frame_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in extensions
    )
    if not frame_paths:
        raise ValueError(f"No image frames found in {folder}")
    return [Image.open(p).convert("RGB") for p in frame_paths]


def _gather_masks(folder: Path, suffix: Optional[str] = None) -> Optional[List[Image.Image]]:
    if folder is None:
        return None
    if not folder.exists():
        raise ValueError(f"Mask directory {folder} does not exist")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    mask_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in extensions
    )
    if suffix:
        mask_paths = [p for p in mask_paths if p.name.endswith(suffix)]
    return [Image.open(p).convert("L") for p in mask_paths]

def _combine_masks(masks: List[Image.Image]) -> Image.Image:
    if not masks:
        raise ValueError("No masks provided for combination")
    base_size = masks[0].size
    combined = Image.new("L", base_size, 0)
    for mask in masks:
        if mask.size != base_size:
            mask = mask.resize(base_size, resample=PIL.Image.NEAREST)
        combined = Image.fromarray(np.maximum(np.array(combined), np.array(mask)))
    return combined

def _invert_mask(mask: Image.Image) -> Image.Image:
    inverted = Image.fromarray(255 - np.array(mask))
    return inverted

def _gather_all_masks(folder: Path, num_objects: int) -> Optional[List[Image.Image]]:
    mask_sets = []
    i = 0
    while i < 10:
        suffix = f"object_00{i}.png"
        masks = _gather_masks(folder, suffix=suffix)
        if masks is None or len(masks) == 0:
            print(f"Warning: No masks found for object {i+1} with suffix '{suffix}'")
            i += 1
            continue
        mask_sets.append(masks)
        i += 1

    return mask_sets

def _parse_id_string(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    tokens = [t.strip() for t in value.replace(",", " ").split() if t.strip()]
    if not tokens:
        return None
    return [int(t) for t in tokens]

def _build_multi_channel_dino(base_dino: Dinov2Model, in_channels: int = 96) -> Dinov2Model:
    """Create a DINO model that accepts `in_channels` inputs by repeating the
    original 3-channel patch-embedding projection weights.
    Mirrors the utility used in training (train_partcrafter_frame.py).
    """
    print(f"Building multi-channel DINO with {in_channels} input channels...")
    
    import copy
    model = copy.deepcopy(base_dino)
    proj = model.embeddings.patch_embeddings.projection
    assert isinstance(proj, torch.nn.Conv2d), "Unexpected DINO projection layer type"

    old_w = proj.weight.data  # [out_c, 3, k, k]
    old_b = proj.bias.data if proj.bias is not None else None

    new_conv = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=(proj.bias is not None),
    )
    with torch.no_grad():
        rep = in_channels // old_w.shape[1]
        rem = in_channels - rep * old_w.shape[1]
        new_w = old_w.repeat(1, rep, 1, 1)
        if rem > 0:
            new_w = torch.cat([new_w, old_w[:, :rem]], dim=1)
        new_conv.weight.copy_(new_w)
        if old_b is not None:
            new_conv.bias.copy_(old_b)

    model.embeddings.patch_embeddings.projection = new_conv
    if hasattr(model.config, "num_channels"):
        model.config.num_channels = in_channels
    if hasattr(model.embeddings, "patch_embeddings") and hasattr(model.embeddings.patch_embeddings, "num_channels"):
        model.embeddings.patch_embeddings.num_channels = in_channels
    return model


def _apply_mask(
        image: PipelineImageInput,
        mask: Optional[PipelineImageInput],
        keep_foreground: bool,
        dilation_radius: int = 3,
        erode_radius: int = 0,
    ) -> PipelineImageInput:
        if mask is None:
            return image
        if isinstance(image, PIL.Image.Image):
            base = image.convert("RGB")
        else:
            base = PIL.Image.fromarray(np.array(image))
        if isinstance(mask, PIL.Image.Image):
            mask_img = mask.convert("L")
        else:
            mask_img = PIL.Image.fromarray(np.array(mask)).convert("L")
        if mask_img.size != base.size:
            mask_img = mask_img.resize(base.size, resample=PIL.Image.NEAREST)
        if dilation_radius > 0:
            kernel_size = dilation_radius * 2 + 1
            mask_img = mask_img.filter(PIL.ImageFilter.MaxFilter(kernel_size))
        if erode_radius > 0:
            kernel_size = erode_radius * 2 + 1
            mask_img = mask_img.filter(PIL.ImageFilter.MinFilter(kernel_size))
        mask_arr = np.asarray(mask_img).astype(np.float32) / 255.0
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
        if not keep_foreground:
            mask_arr = 1.0 - mask_arr
        base_arr = np.asarray(base).astype(np.float32) / 255.0
        # Apply mask: keep original where mask=1, white where mask=0
        masked = base_arr * mask_arr[..., None] + (1.0 - mask_arr[..., None])
        masked = masked.clip(0.0, 1.0)
        masked_img = PIL.Image.fromarray((masked * 255.0).astype(np.uint8))
        return masked_img


@torch.no_grad()
def build_pipeline(
    transformer_cls: ModelMixin,
    pipeline_cls: DiffusionPipeline,
    base_dir: str,
    transformer_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    dino_multi_root: Optional[str] = None,
    use_dino_multi: Optional[bool] = True,
    transformer_scene_attn_ids: Optional[List[int]] = None,
    transformer_dynamic_attn_ids: Optional[List[int]] = None,
    global_attn_block_ids: Optional[List[int]] = None,
) -> DiffusionPipeline:
    vae = TripoSGVAEModel.from_pretrained(base_dir, subfolder="vae")
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(base_dir, subfolder="feature_extractor_dinov2")
    image_encoder_dinov2 = Dinov2Model.from_pretrained(base_dir, subfolder="image_encoder_dinov2")

    cand = os.path.join(transformer_dir, "transformer_ema")
    if not (os.path.isdir(cand) and os.path.exists(os.path.join(cand, "diffusion_pytorch_model.safetensors"))):
        cand = os.path.join(transformer_dir, "transformer") if os.path.isdir(os.path.join(transformer_dir, "transformer")) else transformer_dir
    print(f"Loading transformer from: {cand}")
    transformer = transformer_cls.from_pretrained(cand)
    
    if hasattr(transformer, "spatial_global_attn_block_ids") and transformer_scene_attn_ids is not None:
        transformer.spatial_global_attn_block_ids = transformer_scene_attn_ids or transformer.spatial_global_attn_block_ids
    if hasattr(transformer, "temporal_global_attn_block_ids") and transformer_dynamic_attn_ids is not None:
        transformer.temporal_global_attn_block_ids = transformer_dynamic_attn_ids or transformer.temporal_global_attn_block_ids
    if hasattr(transformer, "global_attn_block_ids") and global_attn_block_ids is not None:
        transformer.global_attn_block_ids = global_attn_block_ids or transformer.global_attn_block_ids

    image_encoder_dinov2_multi = None

    if use_dino_multi:
        print("Configuring multi-channel DINO image encoder...")

        image_encoder_dinov2 = Dinov2Model.from_pretrained(base_dir, subfolder="image_encoder_dinov2")        
        tuned_multi_dir = None
        search_root = dino_multi_root or transformer_dir
        cand = os.path.join(search_root, "image_encoder_dinov2_multi")
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            tuned_multi_dir = cand
        elif os.path.isfile(os.path.join(search_root, "config.json")) and os.path.isfile(os.path.join(search_root, "model.safetensors")):
            tuned_multi_dir = search_root
        
        if tuned_multi_dir is not None:
            try:
                print(f"Loading tuned multi-channel DINO from: {tuned_multi_dir}")
                image_encoder_dinov2_multi = Dinov2Model.from_pretrained(tuned_multi_dir)
            except Exception:
                image_encoder_dinov2_multi = _build_multi_channel_dino(image_encoder_dinov2, in_channels=96)
        else:
            image_encoder_dinov2_multi = _build_multi_channel_dino(image_encoder_dinov2, in_channels=96)

        image_encoder_dinov2_multi = image_encoder_dinov2_multi.to(device=device, dtype=dtype)

    pipe = pipeline_cls(
        vae=vae,
        image_encoder_dinov2=image_encoder_dinov2,
        feature_extractor_dinov2=feature_extractor_dinov2,
        transformer=transformer,
        scheduler=RectifiedFlowScheduler.from_pretrained(base_dir, subfolder="scheduler"),
        image_encoder_dinov2_multi=image_encoder_dinov2_multi,
    ).to(device, dtype=dtype)
    

    return pipe
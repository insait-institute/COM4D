import inspect
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import os

from tqdm import tqdm
import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
import PIL.ImageFilter
from collections import defaultdict
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..utils.inference_utils import hierarchical_extract_geometry
from ..utils.inference_utils import hierarchical_extract_fields
from ..utils.inference_utils import eliminate_collisions
from ..utils.inference_utils import field_to_mesh


from ..schedulers import RectifiedFlowScheduler
from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import PartCrafterDiTModel
from ..models.transformers import COM4DDiTModel
from ..models.attention_processor import COM4DAttnProcessor, TripoSGAttnProcessor2_0
from .pipeline_partcrafter_output import PartCrafterPipelineOutput, PartCrafter3D4DOutput
from .pipeline_utils import TransformerDiffusionMixin
from ..utils.data_utils import get_colored_mesh_composition, RGB as DEFAULT_PART_COLORS
from ..utils.inference import _apply_mask, _combine_masks
from ..utils.render_utils import export_renderings, render_sequence_fixed_camera, render_views_around_mesh

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class PartCrafterPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image to 3D part-level object generation.       
    """

    def __init__(
        self,   
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        # Per-frame single-image tokens
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state  # [N, Ts, D]

        # Optional global tokens from multi-channel DINO by stacking frames as channels
        if hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
            N, _, Hf, Wf = image.shape
            mc = image.reshape(N * 3, Hf, Wf)  # [3N, Hf, Wf]
            if mc.shape[0] > 96:
                mc = mc[:96]
            elif mc.shape[0] < 96:
                pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
                mc = torch.cat([mc, pad], dim=0)
            mc = mc.unsqueeze(0)  # [1, 96, Hf, Wf]
            global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state  # [1, Tm, D]
            if global_embeds.device != image_embeds.device or global_embeds.dtype != image_embeds.dtype:
                global_embeds = global_embeds.to(device=image_embeds.device, dtype=image_embeds.dtype)
            global_embeds = global_embeds.expand(N, -1, -1)  # [N, Tm, D]
            image_embeds = torch.cat([image_embeds, global_embeds], dim=1)  # [N, Ts+Tm, D]

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8, 
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = next(self.transformer.parameters()).device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_images_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        start_time = time.time()
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].to(dtype)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        end_time = time.time()


        # 7. decoder mesh
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
       
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes, time=end_time - start_time)

class FourDCrafterPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image to 3D part-level object generation.       
    """

    def __init__(
        self,   
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        # Per-frame single-image tokens
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state  # [N, Ts, D]

        # Optional global tokens from multi-channel DINO by stacking frames as channels
        if hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
            # print("Using multi-channel DINO")
            N, _, Hf, Wf = image.shape
            mc = image.reshape(N * 3, Hf, Wf)  # [3N, Hf, Wf]
            if mc.shape[0] > 96:
                mc = mc[:96]
            elif mc.shape[0] < 96:
                pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
                mc = torch.cat([mc, pad], dim=0)
            mc = mc.unsqueeze(0)  # [1, 96, Hf, Wf]
            global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state  # [1, Tm, D]
            # Ensure device/dtype alignment before concatenation
            global_embeds = global_embeds.expand(N, -1, -1)  # [N, Tm, D]
            image_embeds = torch.cat([image_embeds, global_embeds], dim=1)  # [N, Ts+Tm, D]

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8, 
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = next(self.transformer.parameters()).device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_images_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        start_time = time.time()
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].to(dtype)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        end_time = time.time()

        # 7. decoder mesh
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
       
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes, time=end_time - start_time)


class FourDCrafterARPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Autoregressive pipeline for 4D frames → per-frame 3D parts using the same
    sequential denoising logic as PartCrafterARPipeline, adapted to the 4D setup.

    - Each object/frame uses its own condition (per-frame DINO tokens, plus optional
      multi-channel global tokens if available).
    - Denoises sequentially in contiguous blocks, capped at 8 objects at a time.
    - Output format matches FourDCrafterPipeline (PartCrafterPipelineOutput).
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        """Encode per-frame conditions; also optionally append multi-channel global tokens.
        
        Mirrors FourDCrafterPipeline.encode_image so each frame gets its own condition.
        """
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        # Per-frame single-image tokens
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state  # [N, Ts, D]

        # # Optional global tokens from multi-channel DINO by stacking frames as channels
        # if hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
        #     N, _, Hf, Wf = image.shape
        #     mc = image.reshape(N * 3, Hf, Wf)  # [3N, Hf, Wf]
        #     if mc.shape[0] > 96:
        #         mc = mc[:96]
        #     elif mc.shape[0] < 96:
        #         pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
        #         mc = torch.cat([mc, pad], dim=0)
        #     mc = mc.unsqueeze(0)  # [1, 96, Hf, Wf]
        #     global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state  # [1, Tm, D]
        #     if global_embeds.device != image_embeds.device or global_embeds.dtype != image_embeds.dtype:
        #         global_embeds = global_embeds.to(device=image_embeds.device, dtype=image_embeds.dtype)
        #     global_embeds = global_embeds.expand(N, -1, -1)  # [N, Tm, D]
        #     image_embeds = torch.cat([image_embeds, global_embeds], dim=1)  # [N, Ts+Tm, D]

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        # AR controls
        ar_block_size: int = 8,            # cap to 8 for memory; sequential groups
        history_mode: str = "fixed",        # ["fixed", "soft"] history composition
        history_soft_alpha: float = 0.1,    # blend ratio for soft composition
        history_renoise_sigma: float = 0.0, # std of Gaussian noise added to history latents
        # Encoding controls
        encode_chunk_size: int = 0,         # if >0, chunk single-image DINO encoding over frames
        max_window_size: int = 4,           # max window size for AR denoising context
        # Decode controls
        reuse_hier_coords: bool = False,    # reuse refinement coords across frames for consistent sampling
        # New: optionally skip decoding to return only latents
        do_decode: bool = True,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Infer batch size from image input
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = next(self.transformer.parameters()).device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition (per-object/per-frame)
        # Compute per-frame single-image tokens once. Multi-channel global tokens
        # are computed on-the-fly using a sliding window capped by max_window_size.
        if not isinstance(image, torch.Tensor):
            image_tensor = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values
        else:
            image_tensor = image
        image_tensor = image_tensor.to(device=device, dtype=dtype)  # [N, 3, H, W]

        # Per-frame single-image tokens (optionally chunked over frames to reduce memory)
        N = image_tensor.shape[0]
        
        single_embeds = self.image_encoder_dinov2(image_tensor).last_hidden_state  # [N, Ts, D]
        single_embeds = single_embeds.repeat_interleave(num_images_per_prompt, dim=0)  # [B, Ts, D]
        uncond_single_embeds = torch.zeros_like(single_embeds)

        # 4. Prepare timesteps (reference, per-block schedule is re-created)
        timesteps_ref, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps_ref)

        # 5. Prepare latent variables (one per part)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            single_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. AR denoising: process in contiguous blocks, capped by window + memory limits
        max_window_size = max(1, int(max_window_size))
        block_limit = max(1, min(int(ar_block_size), 8))
        block_size = max(1, min(block_limit, max_window_size))
        self.set_progress_bar_config(
            desc="AR Denoising",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        # Partition parts (batch rows) into contiguous blocks of size block_size
        blocks: List[List[int]] = [list(range(s, min(s + block_size, batch_size))) for s in range(0, batch_size, block_size)]

        total_iters = len(timesteps_ref) * len(blocks)
        start_time = time.time()
        with self.progress_bar(total=total_iters) as progress_bar:
            for S in blocks:
                # full schedule for the current block
                timesteps_block, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
                for i, t in enumerate(timesteps_block):
                    if self.interrupt:
                        progress_bar.update()
                        continue

                    # Sliding window anchored at the block's last index
                    anchor_idx = max(S)
                    win_start = max(0, anchor_idx - (max_window_size - 1))
                    win_end = anchor_idx + 1
                    if win_end - win_start > max_window_size:
                        win_start = win_end - max_window_size
                    W = win_end - win_start

                    cond_embeds_win = single_embeds[win_start:win_end]

                    if self.do_classifier_free_guidance:
                        uncond_win = torch.zeros_like(cond_embeds_win)
                        image_embeds_cat = torch.cat([uncond_win, cond_embeds_win], dim=0)
                    else:
                        image_embeds_cat = cond_embeds_win

                    # Masks within window
                    in_window_targets = [idx for idx in S if win_start <= idx < win_end]
                    update_mask_win = torch.zeros((W, 1, 1), dtype=torch.bool, device=latents.device)
                    if len(in_window_targets) > 0:
                        update_mask_win[[i - win_start for i in in_window_targets]] = True
                    history_mask_win = ~update_mask_win

                    # Copy slice and prepare feed slice (optional renoise for history)
                    latents_prev_slice = latents[win_start:win_end].clone()
                    latents_feed_win = latents_prev_slice.clone()
                    if history_renoise_sigma > 0.0 and history_mask_win.any():
                        hist_noise = torch.randn_like(latents_feed_win)
                        latents_feed_win = torch.where(history_mask_win, latents_feed_win + history_renoise_sigma * hist_noise, latents_feed_win)

                    # CFG duplication on slice
                    latent_model_input = torch.cat([latents_feed_win] * 2) if self.do_classifier_free_guidance else latents_feed_win
                    timestep = t.expand(latent_model_input.shape[0])

                    # Adjust attention kwarg to window size
                    local_attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else {}
                    local_attention_kwargs["num_parts"] = None
                    local_attention_kwargs["num_frames"] = W

                    # print("latent_model_input", latent_model_input.shape,
                    #       "timestep", timestep.shape,
                    #       "image_embeds_cat", image_embeds_cat.shape,
                    #       "attention_kwargs", local_attention_kwargs)

                    # Predict noise for the window
                    # print("Denoising block parts", S, "at timestep", int(t.item()), "window", (win_start, win_end), "image embeds", image_embeds_cat.shape, "latents", latent_model_input.shape)
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeds_cat,
                        attention_kwargs=local_attention_kwargs,
                        return_dict=False,
                    )[0].to(dtype)

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_image - noise_pred_uncond)

                    # Scheduler step for the window
                    latents_dtype = latents.dtype
                    latents_next_win = self.scheduler.step(noise_pred, t, latents_feed_win, return_dict=False)[0]
                    if latents_next_win.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents_next_win = latents_next_win.to(latents_dtype)

                    # Compose slice and write back
                    if history_mode == "fixed":
                        latents_new_slice = torch.where(update_mask_win, latents_next_win, latents_prev_slice)
                    elif history_mode == "soft":
                        alpha = float(history_soft_alpha)
                        hist_blend = (1.0 - alpha) * latents_prev_slice + alpha * latents_next_win
                        latents_new_slice = torch.where(update_mask_win, latents_next_win, hist_blend)
                    else:
                        raise ValueError(f"Unknown history_mode: {history_mode}")

                    latents[win_start:win_end] = latents_new_slice

                    if callback_on_step_end is not None:
                        callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

        end_time = time.time()
        meshes: List[trimesh.Trimesh] = []
        output = []
        if do_decode:
            self.vae.set_flash_decoder()
            self.set_progress_bar_config(
                desc="Decoding",
                ncols=125,
                disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
            )
            # Optional schedule reuse across frames for consistent sampling
            with self.progress_bar(total=batch_size) as progress_bar:
                for i in range(batch_size):
                    geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                    try:
                        mesh_v_f = hierarchical_extract_geometry(
                            geometric_func,
                            device,
                            dtype=latents.dtype,
                            bounds=bounds,
                            dense_octree_depth=dense_octree_depth,
                            hierarchical_octree_depth=hierarchical_octree_depth,
                            max_num_expanded_coords=max_num_expanded_coords,
                        )
                        mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                    except Exception:
                        mesh_v_f = None
                        mesh = None
                    output.append(mesh_v_f)
                    meshes.append(mesh)
                    progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes, latents=latents, time=end_time - start_time)

class FourDCrafterARPipelineAdvancedDecode(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Autoregressive pipeline for 4D frames → per-frame 3D parts using the same
    sequential denoising logic as PartCrafterARPipeline, adapted to the 4D setup.

    - Each object/frame uses its own condition (per-frame DINO tokens, plus optional
      multi-channel global tokens if available).
    - Denoises sequentially in contiguous blocks, capped at 8 objects at a time.
    - Output format matches FourDCrafterPipeline (PartCrafterPipelineOutput).
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        """Encode per-frame conditions; also optionally append multi-channel global tokens.
        
        Mirrors FourDCrafterPipeline.encode_image so each frame gets its own condition.
        """
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        # Per-frame single-image tokens
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state  # [N, Ts, D]

        # Optional global tokens from multi-channel DINO by stacking frames as channels
        if hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
            N, _, Hf, Wf = image.shape
            mc = image.reshape(N * 3, Hf, Wf)  # [3N, Hf, Wf]
            if mc.shape[0] > 96:
                mc = mc[:96]
            elif mc.shape[0] < 96:
                pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
                mc = torch.cat([mc, pad], dim=0)
            mc = mc.unsqueeze(0)  # [1, 96, Hf, Wf]
            global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state  # [1, Tm, D]
            if global_embeds.device != image_embeds.device or global_embeds.dtype != image_embeds.dtype:
                global_embeds = global_embeds.to(device=image_embeds.device, dtype=image_embeds.dtype)
            global_embeds = global_embeds.expand(N, -1, -1)  # [N, Tm, D]
            image_embeds = torch.cat([image_embeds, global_embeds], dim=1)  # [N, Ts+Tm, D]

        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        # AR controls
        ar_block_size: int = 8,            # cap to 8 for memory; sequential groups
        history_mode: str = "fixed",        # ["fixed", "soft"] history composition
        history_soft_alpha: float = 0.1,    # blend ratio for soft composition
        history_renoise_sigma: float = 0.0, # std of Gaussian noise added to history latents
        # Encoding controls
        encode_chunk_size: int = 0,         # if >0, chunk single-image DINO encoding over frames
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Infer batch size from image input
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = next(self.transformer.parameters()).device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition (per-object/per-frame)
        # Compute per-frame single-image tokens once. Multi-channel global tokens
        # are computed on-the-fly using a sliding window of at most 8 frames.
        if not isinstance(image, torch.Tensor):
            image_tensor = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values
        else:
            image_tensor = image
        image_tensor = image_tensor.to(device=device, dtype=dtype)  # [N, 3, H, W]

        # Per-frame single-image tokens (optionally chunked over frames to reduce memory)
        N = image_tensor.shape[0]
        if encode_chunk_size and encode_chunk_size > 0 and N > encode_chunk_size:
            chunks = []
            for s in range(0, N, encode_chunk_size):
                e = min(s + encode_chunk_size, N)
                emb = self.image_encoder_dinov2(image_tensor[s:e]).last_hidden_state
                chunks.append(emb)
            single_embeds = torch.cat(chunks, dim=0)
        else:
            single_embeds = self.image_encoder_dinov2(image_tensor).last_hidden_state  # [N, Ts, D]
        single_embeds = single_embeds.repeat_interleave(num_images_per_prompt, dim=0)  # [B, Ts, D]
        uncond_single_embeds = torch.zeros_like(single_embeds)

        # 4. Prepare timesteps (reference, per-block schedule is re-created)
        timesteps_ref, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps_ref)

        # 5. Prepare latent variables (one per part)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            single_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. AR denoising: process in contiguous blocks, capped at 8
        block_size = max(1, min(int(ar_block_size), 8))
        self.set_progress_bar_config(
            desc="AR Denoising",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        # Partition parts (batch rows) into contiguous blocks of size block_size
        blocks: List[List[int]] = [list(range(s, min(s + block_size, batch_size))) for s in range(0, batch_size, block_size)]

        total_iters = len(timesteps_ref) * len(blocks)
        with self.progress_bar(total=total_iters) as progress_bar:
            for S in blocks:
                # full schedule for the current block
                timesteps_block, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
                for i, t in enumerate(timesteps_block):
                    if self.interrupt:
                        progress_bar.update()
                        continue

                    # Sliding 8-part window anchored at the block's last index
                    anchor_idx = max(S)
                    win_start = max(0, anchor_idx - 7)
                    win_end = anchor_idx + 1
                    W = win_end - win_start

                    # Build encoder hidden states for the window
                    if hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
                        imgs_win = image_tensor[win_start:win_end]
                        F = imgs_win.shape[0]
                        _, _, Hf, Wf = imgs_win.shape
                        mc = imgs_win.reshape(F * 3, Hf, Wf)
                        if mc.shape[0] > 24:
                            mc = mc[:24]
                        if mc.shape[0] < 96:
                            pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
                            mc = torch.cat([mc, pad], dim=0)
                        mc = mc.unsqueeze(0)
                        global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state
                        if global_embeds.device != single_embeds.device or global_embeds.dtype != single_embeds.dtype:
                            global_embeds = global_embeds.to(device=single_embeds.device, dtype=single_embeds.dtype)
                        global_embeds = global_embeds.expand(W, -1, -1)
                        cond_embeds_win = torch.cat([single_embeds[win_start:win_end], global_embeds], dim=1)
                    else:
                        cond_embeds_win = single_embeds[win_start:win_end]

                    if self.do_classifier_free_guidance:
                        uncond_win = torch.zeros_like(cond_embeds_win)
                        image_embeds_cat = torch.cat([uncond_win, cond_embeds_win], dim=0)
                    else:
                        image_embeds_cat = cond_embeds_win

                    # Masks within window
                    in_window_targets = [idx for idx in S if win_start <= idx < win_end]
                    update_mask_win = torch.zeros((W, 1, 1), dtype=torch.bool, device=latents.device)
                    if len(in_window_targets) > 0:
                        update_mask_win[[i - win_start for i in in_window_targets]] = True
                    history_mask_win = ~update_mask_win

                    # Copy slice and prepare feed slice (optional renoise for history)
                    latents_prev_slice = latents[win_start:win_end].clone()
                    latents_feed_win = latents_prev_slice.clone()
                    if history_renoise_sigma > 0.0 and history_mask_win.any():
                        hist_noise = torch.randn_like(latents_feed_win)
                        latents_feed_win = torch.where(history_mask_win, latents_feed_win + history_renoise_sigma * hist_noise, latents_feed_win)

                    # CFG duplication on slice
                    latent_model_input = torch.cat([latents_feed_win] * 2) if self.do_classifier_free_guidance else latents_feed_win
                    timestep = t.expand(latent_model_input.shape[0])

                    # Adjust attention kwarg to window size
                    local_attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else {}
                    local_attention_kwargs["num_parts"] = W

                    # print("latent_model_input", latent_model_input.shape,
                    #       "timestep", timestep.shape,
                    #       "image_embeds_cat", image_embeds_cat.shape,
                    #       "attention_kwargs", local_attention_kwargs)

                    # Predict noise for the window
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeds_cat,
                        attention_kwargs=local_attention_kwargs,
                        return_dict=False,
                    )[0].to(dtype)

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_image - noise_pred_uncond)

                    # Scheduler step for the window
                    latents_dtype = latents.dtype
                    latents_next_win = self.scheduler.step(noise_pred, t, latents_feed_win, return_dict=False)[0]
                    if latents_next_win.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents_next_win = latents_next_win.to(latents_dtype)

                    # Compose slice and write back
                    if history_mode == "fixed":
                        latents_new_slice = torch.where(update_mask_win, latents_next_win, latents_prev_slice)
                    elif history_mode == "soft":
                        alpha = float(history_soft_alpha)
                        hist_blend = (1.0 - alpha) * latents_prev_slice + alpha * latents_next_win
                        latents_new_slice = torch.where(update_mask_win, latents_next_win, hist_blend)
                    else:
                        raise ValueError(f"Unknown history_mode: {history_mode}")

                    latents[win_start:win_end] = latents_new_slice

                    if callback_on_step_end is not None:
                        callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

        # 7. Decode meshes (same as base pipeline)
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except Exception:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes)


class PartCrafterARPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Autoregressive (AR) pipeline for image → 3D part-level generation where parts are generated sequentially.
    Treats each *part* like a *frame*: already generated parts are held fixed while the next part is denoised
    through the entire diffusion schedule. Output format matches PartCrafterPipeline.
    """
    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: PartCrafterDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )

    @classmethod
    def from_pretrained_with_transformer(
        cls,
        base_dir: str,
        transformer_root_dir: str,
        **kwargs,
    ) -> "PartCrafterARPipeline":
        """Load pipeline from `base_dir` but replace the transformer with weights from `transformer_root_dir`.
        Accepts either `<root>/transformer/{config.json,diffusion_pytorch_model.safetensors}` or
        `<root>/{config.json,diffusion_pytorch_model.safetensors}`.
        """
        pipe = cls.from_pretrained(base_dir, **kwargs)
        pipe.swap_transformer_from_dir(transformer_root_dir, **kwargs)
        pipe.scheduler_dir = os.path.join(base_dir, "scheduler")
        return pipe

    def swap_transformer_from_dir(self, transformer_root_dir: str, **kwargs) -> None:
        """Replace `self.transformer` with a `PartCrafterDiTModel` loaded from the given directory."""
        import os
        # Prefer a nested `transformer/` folder, else fall back to the root
        cand = os.path.join(transformer_root_dir, "transformer")
        have_nested = (
            os.path.isfile(os.path.join(cand, "diffusion_pytorch_model.safetensors"))
            and os.path.isfile(os.path.join(cand, "config.json"))
        )
        if not have_nested:
            cand = transformer_root_dir
        assert (
            os.path.isfile(os.path.join(cand, "diffusion_pytorch_model.safetensors"))
            and os.path.isfile(os.path.join(cand, "config.json"))
        ), (
            f"Transformer weights not found under '{transformer_root_dir}'.\n"
            f"Expected files at either '{{root}}/transformer/{{config.json,diffusion_pytorch_model.safetensors}}' "
            f"or '{{root}}/{{config.json,diffusion_pytorch_model.safetensors}}'."
        )
        # Load new transformer with the same device/dtype as the current one
        device = self.transformer.device
        dtype = next(self.transformer.parameters()).dtype
        new_transformer = PartCrafterDiTModel.from_pretrained(cand, **kwargs)
        self.transformer = new_transformer.to(device=device, dtype=dtype)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}."
            )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        mask: PipelineImageInput = None,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        ar_block_size: int = 1,  # number of parts to denoise together
        ar_merge: bool = True,  # interleave blocks so every part progresses step-by-step
        history_mode: str = "fixed",          # ["fixed", "soft"]: how to treat history parts during a step
        history_soft_alpha: float = 0.1,       # blend ratio for soft composition of history parts
        history_renoise_sigma: float = 0.0,    # std of Gaussian noise injected into history latents before a step
        extra_image: Optional[PIL.Image.Image] = None,  # extra image input for the last unconditioned part
        reuse_hier_coords: bool = False,       # reuse refinement coordinates recorded from first decode
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Batch size inferred from image input
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(image, device, num_images_per_prompt)
        
        # Keep separate copies for later; build the concatenated form for the first pass
        if self.do_classifier_free_guidance:
            image_embeds_cat = torch.cat([negative_image_embeds, image_embeds], dim=0)
        else:
            image_embeds_cat = image_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables (one sample per part)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        self.set_progress_bar_config(
            desc="AR Denoising",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        # Partition parts (batch rows) into contiguous blocks of size ar_block_size
        blocks: List[List[int]] = [
            list(range(s, min(s + ar_block_size, batch_size)))
            for s in range(0, batch_size, ar_block_size)
        ]

        print(f"PartCrafterARPipeline: {len(blocks)} blocks of size ~{ar_block_size} for {batch_size} parts")
        
        total_iters = len(timesteps) * len(blocks)
        start_time = time.time()
        with self.progress_bar(total=total_iters) as progress_bar:
            for block_idx, S in enumerate(blocks):

                timesteps_block, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
                for i, t in enumerate(timesteps_block):
                    if self.interrupt:
                        progress_bar.update()
                        continue

                    # Build masks: targets (current block S) vs history (all other parts)
                    update_mask = torch.zeros((latents.shape[0], 1, 1), dtype=torch.bool, device=latents.device)
                    update_mask[S] = True
                    history_mask = ~update_mask

                    # Save a copy for composition
                    latents_prev = latents.clone()

                    # Optionally re-noise history latents before the step (DFoT-style prepare)
                    latents_feed = latents.clone()

                    # classifier-free guidance duplication
                    latent_model_input = (
                        torch.cat([latents_feed] * 2) if self.do_classifier_free_guidance else latents_feed
                    )
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeds_cat,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].to(dtype)

                    # CFG
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_image - noise_pred_uncond)

                    # scheduler step on the (possibly re-noised) latents
                    latents_dtype = latents.dtype
                    latents_next = self.scheduler.step(noise_pred, t, latents_feed, return_dict=False)[0]
                    if latents_next.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents_next = latents_next.to(latents_dtype)

                    latents = torch.where(update_mask, latents_next, latents_prev)

                    if callback_on_step_end is not None:
                        callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)

                    progress_bar.update()

        end_time = time.time()
        # 7. Decode meshes (same as base pipeline)
        self.vae.set_flash_decoder()
        output, meshes = [], []
        decode_batch_size = latents.shape[0]
        self.set_progress_bar_config(
            desc="Decoding",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=decode_batch_size) as progress_bar:
            for i in range(decode_batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except Exception:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)
        return PartCrafterPipelineOutput(samples=output, meshes=meshes, time=end_time - start_time)


    @torch.no_grad()
    def __call__2(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8, 
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        ###
        context_parts: int = 0,  # Number of initial context parts (like context frames in video)
        use_causal_mask: bool = True,  # Whether to use causal attention mask
        scheduling_type: str = "autoregressive",  # "autoregressive" or "full_sequence"
        sliding_window_size: Optional[int] = None,  # For handling many parts
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_images_per_prompt
        )

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )
    
        if sliding_window_size is None:
            sliding_window_size = batch_size

        part_schedulers = [FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config) for _ in range(batch_size)]

        scheduling_matrix = self._generate_scheduling_matrix(
            num_parts=batch_size,
            context_parts=context_parts,
            scheduling_type=scheduling_type,
            num_timesteps=len(timesteps),
            device=device
        )

        print("timesteps:", timesteps.shape,
              "latents:", latents.shape,
              "batch_size:", batch_size,
              "num_images_per_prompt:", num_images_per_prompt,
              "num_tokens:", num_tokens,
              "num_channels_latents:", num_channels_latents,
              "scheduling_matrix:", scheduling_matrix.shape,
            )
    
        self.set_progress_bar_config(
            desc="Denoising", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        with self.progress_bar(total=len(scheduling_matrix)) as progress_bar:
            for part_timesteps in scheduling_matrix:
                if self.interrupt:
                    continue

                valid_parts = (part_timesteps >= 0)

                valid_timesteps = part_timesteps[valid_parts]
                valid_latents = latents[valid_parts]
                valid_image_embeds = image_embeds[torch.cat([valid_parts, valid_parts], dim=0)] if self.do_classifier_free_guidance else image_embeds[valid_parts]

                latent_model_input = (
                    torch.cat([valid_latents] * 2)
                    if self.do_classifier_free_guidance
                    else valid_latents
                )

                timestep = timesteps[len(timesteps) - (valid_timesteps) - 1]
                timestep = torch.cat([timestep] * 2) if self.do_classifier_free_guidance else timestep

                attention_kwargs['num_parts'] = len(valid_latents)

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=valid_image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0].to(dtype)

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                valid_schedulers = []

                for i in valid_parts:
                    valid_schedulers.append(part_schedulers[i])
                
                for i in range(len(valid_latents)):
                    t = timesteps[i]
                    latents_dtype = latents.dtype
                    latents[valid_parts][i] = valid_schedulers[i].step(
                        noise_pred[i].unsqueeze(0), t, valid_latents[i].unsqueeze(0), return_dict=False
                    )[0]

                    if latents[valid_parts][i].dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            latents[valid_parts][i] = latents[valid_parts][i].to(latents_dtype)
                
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        # 7. decoder mesh
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding", 
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
       
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PartCrafterPipelineOutput(samples=output, meshes=meshes)

    @torch.no_grad()
    def old_call(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # AR-specific parameters
        context_parts: int = 0,  # Number of initial context parts (like context frames in video)
        use_causal_mask: bool = True,  # Whether to use causal attention mask
        scheduling_type: str = "autoregressive",  # "autoregressive" or "full_sequence"
        sliding_window_size: Optional[int] = None,  # For handling many parts
        reconstruction_guidance: float = 0.0,  # Guidance to keep context parts consistent
        # Mesh generation parameters
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        tokens_per_part: int = 512,  # Number of tokens per part (2048 total / 4 parts = 512)
        return_dict: bool = True,
    ):
        """
        Generate 3D parts sequentially using autoregressive diffusion forcing.
        
        This method treats each part as a "frame" in the video generation context:
        - Parts are generated sequentially, with previously generated parts serving as context
        - Uses scheduling matrices to control the denoising process across parts
        - Supports sliding window generation for scenes with many parts
        """
        
        # ============================================
        # 1. Setup and Parameter Validation
        # ============================================
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        
        # Determine batch size from input
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")
        
        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype
        
        # Calculate total number of tokens
        
        # Set sliding window size if not specified
        if sliding_window_size is None:
            sliding_window_size = batch_size
        
        # ============================================
        # 2. Encode Image Condition
        # ============================================
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_images_per_prompt
        )
        
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
        
        # ============================================
        # 3. Prepare Timesteps
        # ============================================
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        self._num_timesteps = len(timesteps)
        # One independent scheduler per part, so each part advances its own index.
        part_schedulers = [copy.deepcopy(self.scheduler) for _ in range(batch_size)]
        
        # ============================================
        # 4. Initialize Latents
        # ============================================
        num_channels_latents = self.transformer.config.in_channels
        
        # Initialize all parts with noise
        all_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # ============================================
        # 5. Generate Scheduling Matrix
        # ============================================
        scheduling_matrix = self._generate_scheduling_matrix(
            num_parts=batch_size,
            num_timesteps=len(timesteps),
            scheduling_type=scheduling_type,
            context_parts=context_parts,
            device=device
        )
        
        # ============================================
        # 6. Create Context Mask
        # ============================================
        # Context mask indicates which parts are already generated (1) vs to be generated (0)
        context_mask = torch.zeros(
            batch_size, batch_size, device=device
        )
        if context_parts > 0:
            context_mask[:, :context_parts] = 1

        print("timesteps:", timesteps.shape,
              "scheduling_matrix:", scheduling_matrix.shape,
              "context_mask:", context_mask.shape,
              "all_latents:", all_latents.shape,
              "num_parts:", batch_size,
              "batch_size:", batch_size,
              "sliding_window_size:", sliding_window_size,
              "tokens_per_part:", tokens_per_part,)
        
        # ============================================
        # 7. Autoregressive Generation Loop
        # ============================================
        self.set_progress_bar_config(
            desc="Generating parts autoregressively",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        
        # Process parts using sliding window if necessary
        generated_parts = 0
        while generated_parts < batch_size:
            # Determine which parts to process in this window
            window_start = max(0, generated_parts - sliding_window_size + 1)
            window_end = min(generated_parts + 1, batch_size)
            window_size = window_end - window_start
            
            # Extract window of latents
            window_latents = all_latents[window_start:window_end].clone()
            window_context_mask = context_mask[window_start:window_end].clone()
            
            # Get scheduling matrix for this window
            window_scheduling = scheduling_matrix[:, window_start:window_end]
            
            print(f"Processing parts {window_start} to {window_end-1} (size {window_size}), window_scheduling: {window_scheduling.shape}, {window_scheduling}")
            
            # ============================================
            # 8. Denoising Loop for Current Window
            # ============================================
            with self.progress_bar(total=len(timesteps) * (window_end - generated_parts)) as progress_bar:
                for step_idx in range(len(scheduling_matrix)):
                    if self.interrupt:
                        break
                    
                    # Get noise levels for each part in the window
                    noise_levels = window_scheduling[step_idx]
                    
                    # Determine which parts need denoising at this step
                    parts_to_denoise = (noise_levels >= 0)

                    if not parts_to_denoise.any():
                        continue
                    
                    latent_input = window_latents
                    
                    # Expand for classifier-free guidance
                    if self.do_classifier_free_guidance:
                        latent_model_input = torch.cat([latent_input] * 2)
                    else:
                        latent_model_input = latent_input
                    
                    # Get timestep for current noise level (using first non-context part's level)
                    active_noise_level = len(timesteps) - (noise_levels[parts_to_denoise][0]) - 1
                    if active_noise_level >= 0 and active_noise_level < len(timesteps):
                        t = timesteps[int(active_noise_level)]
                    else:
                        continue
                    
                    timestep = t.expand(latent_model_input.shape[0])

                    # ============================================
                    # 9. Apply Attention Mask for Causality
                    # ============================================
                    if use_causal_mask and attention_kwargs is not None:
                        # Create causal mask at token level
                        causal_mask = self._create_causal_attention_mask(
                            window_size, tokens_per_part, window_context_mask, device
                        )
                        attention_kwargs["attention_mask"] = causal_mask
                    
                    # ============================================
                    # 10. Transformer Forward Pass
                    # ============================================
                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].to(dtype)
                    
                    # ============================================
                    # 11. Apply Classifier-Free Guidance
                    # ============================================
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )
                    
                    # # ============================================
                    # # 12. Reconstruction Guidance (Optional)
                    # # ============================================
                    # if reconstruction_guidance > 0 and window_context_mask.sum() > 0:
                    #     # Apply guidance to maintain context parts
                    #     context_parts_mask = window_context_mask.unsqueeze(-1).unsqueeze(-1)
                    #     reconstruction_loss = F.mse_loss(
                    #         noise_pred * context_parts_mask,
                    #         window_latents * context_parts_mask,
                    #         reduction='none'
                    #     )
                    #     reconstruction_grad = torch.autograd.grad(
                    #         reconstruction_loss.sum(), noise_pred, retain_graph=False
                    #     )[0]
                    #     noise_pred = noise_pred - reconstruction_guidance * reconstruction_grad
                    
                    # ============================================
                    # 13. Denoise Step
                    # ============================================
                    # Map current timestep tensor `t` to its index within the scheduler's timetable.
                    # This allows calling `scheduler.step` multiple times for the same `t` (once per part)
                    # without advancing the internal pointer beyond bounds.
                
                    # Apply the update only to parts whose level == lev
                    for part_idx in range(window_size):
                        if not parts_to_denoise[part_idx]:
                            continue

                        # Global index of this part among all parts
                        global_part_idx = window_start + part_idx

                        part_latents = window_latents[part_idx]
                        part_noise_pred = noise_pred[part_idx]

                        print(f" Denoising part {global_part_idx} at level {int(noise_levels[part_idx].item())} / timestep {t.item()}")

                        # IMPORTANT: use the scheduler that belongs to THIS part.
                        denoised = part_schedulers[global_part_idx].step(
                            part_noise_pred, t, part_latents, return_dict=False
                        )[0]

                        # Update only if this part isn't frozen as context
                        if not window_context_mask[part_idx].any():
                            window_latents[part_idx] = denoised

                    # After processing all active levels for this AR step:
                    progress_bar.update()

                    # Callback
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, step_idx, t, callback_kwargs)
                        window_latents = callback_outputs.pop("latents", window_latents)
            
            # ============================================
            # 14. Update Global Latents and Context
            # ============================================
            # Copy denoised parts back to global latents
            all_latents[window_start:window_end] = window_latents
            
            # Mark newly generated parts as context for next iteration
            newly_generated = window_end - 1
            if newly_generated >= generated_parts:
                context_mask[newly_generated] = 1
                generated_parts = newly_generated + 1
        
        # ============================================
        # 15. Decode to Meshes
        # ============================================
        self.vae.set_flash_decoder()
        output, meshes = [], []
        
        self.set_progress_bar_config(
            desc="Decoding to meshes",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(
                    all_latents[i].unsqueeze(0), sampled_points=x
                ).sample
                
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=all_latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except:
                    mesh_v_f = None
                    mesh = None
                
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()
        
        # Cleanup
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (output, meshes)
        
        return PartCrafterPipelineOutput(
            samples=output, 
            meshes=meshes,
            part_latents=all_latents  # Include part-wise latents for analysis
        )

    @torch.no_grad()
    def __call__overlap(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        ar_block_size: int = 1,  # number of parts to denoise together
        ar_merge: bool = True,  # interleave blocks so every part progresses step-by-step
        history_mode: str = "soft",          # ["fixed", "soft"]: how to treat history parts during a step
        history_soft_alpha: float = 0.1,       # blend ratio for soft composition of history parts
        history_renoise_sigma: float = 0.0,    # std of Gaussian noise injected into history latents before a step
        overlap_steps: int = 0,                # how many steps to overlap between consecutive parts
        extra_image: Optional[PIL.Image.Image] = None,  # extra image input for the last unconditioned part
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Batch size inferred from image input
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(image, device, num_images_per_prompt)
        # Keep separate copies for later; build the concatenated form for the first pass
        if self.do_classifier_free_guidance:
            image_embeds_cat = torch.cat([negative_image_embeds, image_embeds], dim=0)
        else:
            image_embeds_cat = image_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Build per-part schedulers (cap at 50 steps)
        max_steps_per_part = min(int(num_inference_steps), 50)
        part_schedulers = [copy.deepcopy(self.scheduler) for _ in range(batch_size)]
        part_timesteps: List[torch.Tensor] = []
        for ps in part_schedulers:
            ps.set_timesteps(max_steps_per_part, device=device)
            part_timesteps.append(ps.timesteps.to(device))

        # 5. Prepare latent variables (one sample per part)
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # -----------------------
        # Overlapped AR denoising
        # -----------------------
        # If overlap_steps > 0, part k starts when part k-1 has `overlap_steps` steps remaining.
        # We define a global virtual step g and, at each g, update all active parts.
        stride = max(max_steps_per_part - int(overlap_steps), 1)
        start_steps = [p * stride for p in range(batch_size)]
        total_global_steps = (start_steps[-1] + max_steps_per_part) if batch_size > 0 else 0

        self.set_progress_bar_config(
            desc="AR Denoising (overlap)",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        with self.progress_bar(total=total_global_steps) as progress_bar:
            for g in range(total_global_steps):
                if self.interrupt:
                    progress_bar.update()
                    continue

                # Determine active parts and their local-step indices at this global step
                active_parts: List[int] = []
                local_indices: List[int] = []
                for p in range(batch_size):
                    s = g - start_steps[p]  # local step for part p
                    if 0 <= s < max_steps_per_part:
                        active_parts.append(p)
                        local_indices.append(s)

                if len(active_parts) == 0:
                    progress_bar.update()
                    continue

                # Build masks for composition
                update_mask = torch.zeros((latents.shape[0], 1, 1), dtype=torch.bool, device=latents.device)
                update_mask[active_parts] = True
                history_mask = ~update_mask

                # Save previous state for composition
                latents_prev = latents.clone()

                # Optionally re-noise history before the step
                latents_feed = latents.clone()
                if history_renoise_sigma > 0.0:
                    hist_noise = torch.randn_like(latents_feed)
                    latents_feed = torch.where(
                        history_mask,
                        latents_feed + history_renoise_sigma * hist_noise,
                        latents_feed,
                    )

                # Slice active subset for compute efficiency
                act_idx = torch.tensor(active_parts, device=device, dtype=torch.long)
                act_latents = latents_feed.index_select(0, act_idx)

                # Per-part timesteps (each part has its own schedule/state)
                t_per_act_list = []
                for idx_part, li in zip(active_parts, local_indices):
                    t_per_act_list.append(part_timesteps[idx_part][li])
                t_per_act = torch.stack(t_per_act_list, dim=0).to(device)

                # Build encoder states and latent inputs (CFG-aware) for active subset
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([act_latents, act_latents], dim=0)
                    timestep_vec = torch.cat([t_per_act, t_per_act], dim=0)
                    image_embeds_step = torch.cat([
                        negative_image_embeds.index_select(0, act_idx),
                        image_embeds.index_select(0, act_idx),
                    ], dim=0)
                else:
                    latent_model_input = act_latents
                    timestep_vec = t_per_act
                    image_embeds_step = image_embeds.index_select(0, act_idx)

                # Adjust attention kwargs to the active subset size to keep (b*ni) consistent with CFG duplication
                attention_kwargs_step = dict(attention_kwargs) if attention_kwargs is not None else {}
                attention_kwargs_step["num_parts"] = len(active_parts)

                # Forward transformer for active subset
                noise_pred = self.transformer(
                    latent_model_input,
                    timestep_vec,
                    encoder_hidden_states=image_embeds_step,
                    attention_kwargs=attention_kwargs_step,
                    return_dict=False,
                )[0].to(dtype)

                # CFG combine
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = torch.chunk(noise_pred, 2, dim=0)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_image - noise_pred_uncond)

                # Step each active sample individually (scheduler may not vectorize over different t)
                latents_next_act = []
                for k in range(len(active_parts)):
                    t_k = t_per_act[k]
                    lat_k = act_latents[k]
                    eps_k = noise_pred[k]
                    next_k = part_schedulers[active_parts[k]].step(eps_k, t_k, lat_k, return_dict=False)[0]
                    latents_next_act.append(next_k)
                latents_next_act = torch.stack(latents_next_act, dim=0)

                # Scatter the updated active rows back
                latents_scatter = latents_prev.clone()
                latents_scatter.index_copy_(0, act_idx, latents_next_act)

                # Compose with history according to mode
                if history_mode == "fixed":
                    latents = torch.where(update_mask, latents_scatter, latents_prev)
                elif history_mode == "soft":
                    alpha = float(history_soft_alpha)
                    hist_blend = (1.0 - alpha) * latents_prev + alpha * latents_scatter
                    latents = torch.where(update_mask, latents_scatter, hist_blend)
                else:
                    raise ValueError(f"Unknown history_mode: {history_mode}")

                # Optional callback
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                    callback_outputs = callback_on_step_end(self, g, t_per_act, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        # ---- Extra pass: add one more unconditioned part and denoise it conditioned on previous parts ----
        if extra_image is not None:
            # 1) Prepare the extra part's latent
            extra_image_embeds, negative_extra_image_embeds = self.encode_image(extra_image, device, num_images_per_prompt)

            image_embeds = torch.cat([image_embeds, extra_image_embeds], dim=0)
            negative_image_embeds = torch.cat([negative_image_embeds, negative_extra_image_embeds], dim=0) 

            # 1) Append a new noise latent for the extra part
            extra_noise = self.prepare_latents(
                1,
                num_tokens,
                num_channels_latents,
                image_embeds.dtype,
                device,
                generator,
                None,
            )  # (1, T, C)
            latents = torch.cat([latents, extra_noise], dim=0)  # shape now: (batch_size+1, T, C)

            # 2) Build encoder hidden states with zero conditioning for the new part (only in the conditional branch)
            # if self.do_classifier_free_guidance:
            #     cond_plus = torch.cat([cond_embeds, torch.zeros_like(cond_embeds[:1])], dim=0)
            #     uncond_plus = torch.cat([uncond_embeds, torch.zeros_like(uncond_embeds[:1])], dim=0)
            #     image_embeds_extra = torch.cat([uncond_plus, cond_plus], dim=0)
            # else:
            #     cond_plus = torch.cat([cond_embeds, torch.zeros_like(cond_embeds[:1])], dim=0)
            #     image_embeds_extra = cond_plus

            if self.do_classifier_free_guidance:
                image_embeds_cat = torch.cat([negative_image_embeds, image_embeds], dim=0)
            else:
                image_embeds_cat = image_embeds

            # 3) Attention kwargs: bump num_parts by 1 for this extra pass
            extra_attention_kwargs = dict(attention_kwargs) if attention_kwargs is not None else {}
            inferred_parts = extra_attention_kwargs.get("num_parts", latents.shape[0])
            extra_attention_kwargs["num_parts"] = max(inferred_parts, latents.shape[0])

            # 4) Full diffusion schedule updating only the new part index
            self.set_progress_bar_config(
                desc="AR Denoising (extra)",
                ncols=125,
                disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
            )
            new_idx = latents.shape[0] - 1
            timesteps_extra, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)
            with self.progress_bar(total=len(timesteps_extra)) as progress_bar:
                for i, t in enumerate(timesteps_extra):
                    if self.interrupt:
                        progress_bar.update()
                        continue

                    update_mask = torch.zeros((latents.shape[0], 1, 1), dtype=torch.bool, device=latents.device)
                    update_mask[new_idx] = True
                    history_mask = ~update_mask

                    latents_prev = latents.clone()

                    latents_feed = latents.clone()
                    if history_renoise_sigma > 0.0:
                        hist_noise = torch.randn_like(latents_feed)
                        latents_feed = torch.where(
                            history_mask,
                            latents_feed + history_renoise_sigma * hist_noise,
                            latents_feed,
                        )

                    latent_model_input = (
                        torch.cat([latents_feed] * 2) if self.do_classifier_free_guidance else latents_feed
                    )
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=image_embeds_cat,
                        attention_kwargs=extra_attention_kwargs,
                        return_dict=False,
                    )[0].to(dtype)

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_image - noise_pred_uncond)

                    latents_dtype = latents.dtype
                    latents_next = self.scheduler.step(noise_pred, t, latents_feed, return_dict=False)[0]
                    if latents_next.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents_next = latents_next.to(latents_dtype)

                    if history_mode == "fixed":
                        latents = torch.where(update_mask, latents_next, latents_prev)
                    elif history_mode == "soft":
                        alpha = float(history_soft_alpha)
                        hist_blend = (1.0 - alpha) * latents_prev + alpha * latents_next
                        latents = torch.where(update_mask, latents_next, hist_blend)
                    else:
                        raise ValueError(f"Unknown history_mode: {history_mode}")

                    progress_bar.update()

        # 7. Decode meshes (same as base pipeline)
        self.vae.set_flash_decoder()
        output, meshes = [], []
        decode_batch_size = latents.shape[0]
        self.set_progress_bar_config(
            desc="Decoding",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=decode_batch_size) as progress_bar:
            for i in range(decode_batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except Exception:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)
        return PartCrafterPipelineOutput(samples=output, meshes=meshes)


    def _generate_scheduling_matrix(
        self,
        num_parts: int,
        num_timesteps: int,
        scheduling_type: str,
        context_parts: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate scheduling matrix for autoregressive part generation.
        
        The scheduling matrix defines which timestep each part should be at during each
        denoising iteration. This is the key innovation from Diffusion Forcing.
        
        Returns:
            Tensor of shape (num_steps, num_parts) containing timestep indices
        """
        
        if scheduling_type == "full_sequence":
            # All parts denoised together (standard diffusion)
            # Shape: (num_timesteps, num_parts)
            matrix = torch.arange(num_timesteps - 1, -1, -1, device=device)
            matrix = matrix.unsqueeze(1).repeat(1, num_parts)

        elif scheduling_type == "autoregressive":
            # Pyramid scheduling: earlier parts get denoised first
            # This creates a "pyramid" where:
            # - Part 0 goes through all timesteps first
            # - Part 1 starts denoising when Part 0 is partially done
            # - Each subsequent part starts with a delay
            matrix = []
            for step in range(num_timesteps + num_parts - 1):
                row = []
                for part_idx in range(num_parts):
                    # Calculate timestep for this part at this step
                    part_timestep = num_timesteps - 1 - (step - part_idx)

                    # Clamp so parts do not denoise before their start time and only take exactly num_timesteps steps total
                    if part_timestep >= num_timesteps:
                        # Not started yet at this global step -> mark as inactive
                        part_timestep = -1
                    elif part_timestep < 0:
                        # Finished denoising -> mark as inactive
                        part_timestep = -1

                    row.append(part_timestep)
                matrix.append(row)
            matrix = torch.tensor(matrix, device=device)

        else:
            raise ValueError(f"Unknown scheduling type: {scheduling_type}")
        
        # Mark context parts as already denoised (-1)
        if context_parts > 0:
            matrix[:, :context_parts] = -1
        
        # Remove redundant steps where no part changes state
        if matrix.shape[0] > 1:
            diff = matrix[1:] - matrix[:-1]
            keep_steps = (diff != 0).any(dim=1)
            keep_steps = torch.cat([torch.tensor([True], device=device), keep_steps])
            matrix = matrix[keep_steps]

        print("Scheduling matrix:\n", matrix.cpu().numpy())
        
        return matrix


    def _create_causal_attention_mask(
        self,
        num_parts: int,
        tokens_per_part: int,
        context_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create causal attention mask at the token level.
        
        This ensures that:
        1. Tokens in later parts cannot attend to tokens in earlier parts that are being generated
        2. All tokens can attend to context parts
        3. Within a part being generated, normal attention is allowed
        
        Returns:
            Boolean mask of shape (num_parts * tokens_per_part, num_parts * tokens_per_part)
            where True means attention is allowed
        """
        
        total_tokens = num_parts * tokens_per_part
        mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
        
        for part_i in range(num_parts):
            for part_j in range(num_parts):
                token_i_start = part_i * tokens_per_part
                token_i_end = (part_i + 1) * tokens_per_part
                token_j_start = part_j * tokens_per_part
                token_j_end = (part_j + 1) * tokens_per_part
                
                # If part_j is a future part (not context), part_i cannot attend to it
                if part_j > part_i and context_mask[0, part_j] == 0:
                    mask[token_i_start:token_i_end, token_j_start:token_j_end] = False
        
        return mask


class PartCrafter3D4DInferencePipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """Pipeline that reconstructs a static scene and a dynamic 4D object with history-guided diffusion."""

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: COM4DDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
        image_encoder_dinov2_multi: Optional[Dinov2Model] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
            image_encoder_dinov2_multi=image_encoder_dinov2_multi,
        )

    def encode_image(self, image, device, num_images_per_prompt, use_multi=False):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        if use_multi and hasattr(self, 'image_encoder_dinov2_multi') and self.image_encoder_dinov2_multi is not None:
            N, _, Hf, Wf = image.shape
            mc = image.reshape(N * 3, Hf, Wf)
            if mc.shape[0] > 96:
                mc = mc[:96]
            elif mc.shape[0] < 96:
                pad = torch.zeros((96 - mc.shape[0], Hf, Wf), device=mc.device, dtype=mc.dtype)
                mc = torch.cat([mc, pad], dim=0)
            mc = mc.unsqueeze(0)
            global_embeds = self.image_encoder_dinov2_multi(mc).last_hidden_state
            if global_embeds.device != image_embeds.device or global_embeds.dtype != image_embeds.dtype:
                global_embeds = global_embeds.to(device=image_embeds.device, dtype=image_embeds.dtype)
            global_embeds = global_embeds.expand(N, -1, -1)
            image_embeds = torch.cat([image_embeds, global_embeds], dim=1)
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise if latents is None else latents

    def _build_attn_processor_map(self, active_ids: List[int]):
        mapping = {}
        for layer_id in range(self.transformer.config.num_layers):
            for attn_id in [1, 2]:
                key = f"blocks.{layer_id}.attn{attn_id}.processor"
                if layer_id in active_ids:
                    mapping[key] = COM4DAttnProcessor()
                else:
                    mapping[key] = TripoSGAttnProcessor2_0()
        return mapping

    def _set_attention_blocks(self, active_ids: List[int]):
        self.transformer.set_global_attn_block_ids(active_ids)

    def _clone_scheduler(self) -> RectifiedFlowScheduler:
        return RectifiedFlowScheduler.from_config(self.scheduler.config)

    def _run_joint_scene_stage(
        self,
        first_frame: List[PipelineImageInput],
        first_frame_mask: Optional[PipelineImageInput],
        masks: List[torch.Tensor],
        masks_static: List[torch.Tensor],
        all_masks: List[torch.Tensor],
        guidance_scale: float,
        block_size: int,
        num_static: int,
        num_dynamic: int,
        generator,
        device: torch.device,
        num_tokens: int = 2048,
        num_inference_steps: int = 50,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        first_frame_index: int = 0,
        scene_mix_cutoff: int = 10,
    ):
        total_parts = num_static + num_dynamic
        if total_parts == 0:
            return [], torch.empty(0, device=device)

        first_frame = first_frame or []
        base_image = first_frame[0] if isinstance(first_frame, (list, tuple)) and len(first_frame) > 0 else first_frame
        scene_masked_image = (
            first_frame[1]
            if isinstance(first_frame, (list, tuple)) and len(first_frame) > 1
            else base_image
        )

        if base_image is None:
            raise ValueError("At least one conditioning image must be provided")

        dynamic_masked_images: List[PipelineImageInput] = []
        for idx in range(min(len(all_masks or []), num_dynamic)):
            mask = all_masks[idx][first_frame_index]
            dynamic_masked_images.append(_apply_mask(base_image, mask, keep_foreground=True, dilation_radius=0))
            dynamic_masked_images[-1].save(f"debug_dynamic_masked_{idx}.png")

        static_masked_image = _apply_mask(first_frame, first_frame_mask, keep_foreground=True, erode_radius=0, dilation_radius=0)
        static_masked_image.save("debug_static_masked.png")

        static_masked_images: List[PipelineImageInput] = []
        static_masked_images.append(static_masked_image)

        if num_static > 0 and not static_masked_images:
            static_masked_images = [scene_masked_image] * num_static

        if num_dynamic > 0 and not dynamic_masked_images:
            dynamic_masked_images = [base_image] * num_dynamic

        static_full_embeds = None
        static_full_uncond = None
        if num_static > 0:
            static_full_embeds, static_full_uncond = self.encode_image([base_image], device, 1)
            static_full_embeds = static_full_embeds.repeat(num_static, 1, 1)
            static_full_uncond = static_full_uncond.repeat(num_static, 1, 1)

        static_scene_embeds = None
        static_scene_uncond = None
        if num_static > 0:
            static_scene_embeds, static_scene_uncond = self.encode_image(static_masked_images, device, 1)
            if static_scene_embeds.shape[0] != num_static:
                static_scene_embeds = self._tile_history_tensor(static_scene_embeds, num_static)
                static_scene_uncond = self._tile_history_tensor(static_scene_uncond, num_static)

        dynamic_embeds = None
        dynamic_uncond = None
        if num_dynamic > 0:
            dynamic_embeds, dynamic_uncond = self.encode_image(dynamic_masked_images, device, 1, use_multi=True)
            if dynamic_embeds.shape[0] != num_dynamic:
                dynamic_embeds = self._tile_history_tensor(dynamic_embeds, num_dynamic)
                dynamic_uncond = self._tile_history_tensor(dynamic_uncond, num_dynamic)

        dtype = (
            static_full_embeds.dtype
            if static_full_embeds is not None
            else dynamic_embeds.dtype
        )

        scheduler = self._clone_scheduler()
        timesteps_ref, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device
        )

        matrix_steps = min(scene_mix_cutoff, len(timesteps_ref))
        remaining_timesteps = timesteps_ref[matrix_steps:]

        num_channels_latents = self.transformer.config.in_channels
        static_latents = (
            self.prepare_latents(
                num_static,
                num_tokens,
                num_channels_latents,
                dtype,
                device,
                generator,
                None,
            )
            if num_static > 0
            else torch.empty(0, num_tokens, num_channels_latents, device=device, dtype=dtype)
        )
        dynamic_latents = (
            self.prepare_latents(
                num_dynamic,
                num_tokens,
                num_channels_latents,
                dtype,
                device,
                generator,
                None,
            )
            if num_dynamic > 0
            else torch.empty(0, num_tokens, num_channels_latents, device=device, dtype=dtype)
        )

        do_cfg = guidance_scale > 1.0

        base_attention_kwargs = dict(attention_kwargs or {})
        base_attention_kwargs.setdefault("num_parts", total_parts)
        base_attention_kwargs.setdefault("num_frames", 1)

        progress_disable = (
            self._progress_bar_config.get("disable", False)
            if hasattr(self, "_progress_bar_config")
            else False
        )

        static_blocks: List[List[int]] = []
        if num_static > 0:
            block = max(1, block_size)
            static_blocks = [
                list(range(start, min(start + block, num_static)))
                for start in range(0, num_static, block)
            ]

        remaining_len = len(remaining_timesteps)
        static_iters = remaining_len * len(static_blocks)
        dynamic_iters = remaining_len * num_dynamic if num_dynamic > 0 else 0

        total_iters = matrix_steps + static_iters + dynamic_iters
        self.set_progress_bar_config(
            desc="Joint Scene Denoising",
            ncols=125,
            disable=progress_disable,
        )

        static_matrix_embeds = (
            static_full_embeds.unsqueeze(1) if static_full_embeds is not None else None
        )
        # static_matrix_embeds = (
        #     static_scene_embeds.unsqueeze(1) if static_scene_embeds is not None else None
        # )
        dynamic_matrix_embeds = (
            dynamic_embeds.unsqueeze(1) if dynamic_embeds is not None else None
        )

        with self.progress_bar(total=total_iters) as progress_bar:
            for step_idx in range(matrix_steps):
                t = timesteps_ref[step_idx]

                latents_blocks: List[torch.Tensor] = []
                encoder_blocks: List[torch.Tensor] = []

                if num_static > 0:
                    latents_blocks.append(static_latents.unsqueeze(1))
                    encoder_blocks.append(static_matrix_embeds)
                if num_dynamic > 0:
                    latents_blocks.append(dynamic_latents.unsqueeze(1))
                    encoder_blocks.append(dynamic_matrix_embeds)

                latents_matrix = torch.cat(latents_blocks, dim=0)
                encoder_hidden_states_matrix = torch.cat(encoder_blocks, dim=0)

                timestep = t.expand(1)

                transformer_output = self.transformer.forward_matrix(
                    latents_matrix,
                    timestep,
                    encoder_hidden_states_matrix,
                    static_count=num_static,
                    dynamic_count=num_dynamic,
                    return_dict=False,
                    cutoff=False,
                )[0]

                transformer_output_uncond = (
                    self.transformer.forward_matrix(
                        latents_matrix,
                        timestep,
                        torch.zeros_like(encoder_hidden_states_matrix),
                        static_count=num_static,
                        dynamic_count=num_dynamic,
                        return_dict=False,
                        cutoff=False,
                    )[0]
                    if do_cfg
                    else None
                )

                if do_cfg:
                    noise_pred = transformer_output_uncond + guidance_scale * (
                        transformer_output - transformer_output_uncond
                    )
                else:
                    noise_pred = transformer_output

                noise_pred_flat = noise_pred[:, 0]
                latents_cat = torch.cat([static_latents, dynamic_latents], dim=0)

                latents_next = scheduler.step(
                    noise_pred_flat, t, latents_cat, return_dict=False
                )[0]

                if latents_next.dtype != latents_cat.dtype:
                    latents_next = latents_next.to(latents_cat.dtype)

                if num_static > 0:
                    static_latents = latents_next[:num_static]
                if num_dynamic > 0:
                    dynamic_latents = latents_next[num_static:]

                progress_bar.update()

            if remaining_len > 0 and num_static > 0:
                static_attention_kwargs = dict(attention_kwargs or {})
                static_attention_kwargs.setdefault("num_parts", num_static)
                static_attention_kwargs.setdefault("num_frames", 1)

                static_encoder = static_scene_embeds
                static_encoder_uncond = static_scene_uncond

                for block_indices in static_blocks:
                    for t in remaining_timesteps:
                        update_mask = torch.zeros(
                            (static_latents.shape[0], 1, 1),
                            dtype=torch.bool,
                            device=device,
                        )
                        update_mask[block_indices] = True

                        latents_prev = static_latents.clone()
                        latents_feed = static_latents.clone()

                        latent_model_input = (
                            torch.cat([latents_feed, latents_feed], dim=0)
                            if do_cfg
                            else latents_feed
                        )
                        timestep = t.expand(latent_model_input.shape[0])

                        encoder_states = static_encoder
                        encoder_states_cfg = (
                            torch.cat(
                                [static_encoder_uncond, static_encoder], dim=0
                            )
                            if do_cfg
                            else encoder_states
                        )

                        noise_pred = self.transformer.forward(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=encoder_states_cfg
                            if do_cfg
                            else encoder_states,
                            attention_kwargs=static_attention_kwargs,
                            return_dict=False,
                        )[0].to(static_latents.dtype)

                        if do_cfg:
                            noise_uncond, noise_cond = noise_pred.chunk(2)
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_cond - noise_uncond
                            )

                        latents_next = scheduler.step(
                            noise_pred, t, latents_feed, return_dict=False
                        )[0]

                        static_latents = torch.where(
                            update_mask, latents_next, latents_prev
                        )
                        progress_bar.update()

            if remaining_len > 0 and num_dynamic > 0:
                dynamic_attention_kwargs = dict(attention_kwargs or {})
                dynamic_attention_kwargs.setdefault("num_parts", 1)
                dynamic_attention_kwargs.setdefault("num_frames", 1)

                new_dynamic_latents = []
                for idx in range(num_dynamic):
                    scheduler_single = self._clone_scheduler()
                    timesteps_single, _ = retrieve_timesteps(
                        scheduler_single, len(remaining_timesteps), device
                    )

                    latents_single = dynamic_latents[idx : idx + 1].clone()
                    encoder_single = dynamic_embeds[idx : idx + 1]
                    encoder_single_uncond = (
                        dynamic_uncond[idx : idx + 1] if do_cfg else None
                    )

                    for t in timesteps_single:
                        latent_model_input = (
                            torch.cat([latents_single, latents_single], dim=0)
                            if do_cfg
                            else latents_single
                        )
                        timestep = t.expand(latent_model_input.shape[0])

                        encoder_states = encoder_single
                        encoder_states_cfg = (
                            torch.cat(
                                [encoder_single_uncond, encoder_single], dim=0
                            )
                            if do_cfg
                            else encoder_states
                        )

                        noise_pred = self.transformer.forward(
                            latent_model_input,
                            timestep,
                            encoder_hidden_states=encoder_states_cfg
                            if do_cfg
                            else encoder_states,
                            attention_kwargs=dynamic_attention_kwargs,
                            return_dict=False,
                        )[0].to(latents_single.dtype)

                        if do_cfg:
                            noise_uncond, noise_cond = noise_pred.chunk(2)
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_cond - noise_uncond
                            )

                        latents_single = scheduler_single.step(
                            noise_pred, t, latents_single, return_dict=False
                        )[0]
                        progress_bar.update()

                    new_dynamic_latents.append(latents_single.squeeze(0))

                dynamic_latents = torch.stack(new_dynamic_latents, dim=0)

            latents_all = torch.cat([static_latents, dynamic_latents], dim=0) if num_dynamic > 0 else static_latents.clone()

        self.vae.set_flash_decoder()
        meshes: List[trimesh.Trimesh] = []
        outputs = []

        self.set_progress_bar_config(
            desc="Joint Scene Decoding",
            ncols=125,
            disable=progress_disable,
        )

        with self.progress_bar(total=total_parts) as progress_bar:
            for i in range(total_parts):
                geometric_func = lambda x, idx=i: self.vae.decode(
                    latents_all[idx].unsqueeze(0), sampled_points=x
                ).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents_all.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )
                    mesh = trimesh.Trimesh(
                        mesh_v_f[0].astype(np.float32), mesh_v_f[1]
                    )
                except Exception as exc:
                    print(f"Warning: joint mesh extraction failed for item {i}: {exc}")
                    mesh_v_f = None
                    mesh = None
                outputs.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        return meshes, static_latents, dynamic_latents

    def _run_scene_stage(
        self,
        images: List[PipelineImageInput],
        first_frame_masked: List[PipelineImageInput],
        masks: List[torch.Tensor],
        masks_static: List[torch.Tensor],
        scene_attention_kwargs: Optional[Dict[str, Any]],
        num_tokens: int,
        num_inference_steps: int,
        guidance_scale: float,
        num_parts: int,
        block_size: int,
        generator,
        device: torch.device,
        return_latents_timesteps: bool = False,
    ):
        batch_size = num_parts
        print(f"Running scene stage for {batch_size} images with block_size={block_size}")
        if batch_size == 0:
            return [], torch.empty(0)

        image_embeds, uncond_image_embeds = self.encode_image([first_frame_masked], device, 1)

        if image_embeds.shape[0] != batch_size:
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            uncond_image_embeds = uncond_image_embeds.repeat(batch_size, 1, 1)

        do_cfg = guidance_scale > 1.0
        encoder_states = (
            torch.cat([uncond_image_embeds, image_embeds], dim=0)
            if do_cfg
            else image_embeds
        )

        image_embeds_full, uncond_image_embeds_full = self.encode_image([images], device, 1)

        if image_embeds_full.shape[0] != batch_size:
            image_embeds_full = image_embeds_full.repeat(batch_size, 1, 1)
            uncond_image_embeds_full = uncond_image_embeds_full.repeat(batch_size, 1, 1)

        encoder_states_full = (
            torch.cat([uncond_image_embeds_full, image_embeds_full], dim=0
            )
            if do_cfg
            else image_embeds_full
        )

        scheduler = self._clone_scheduler()
        timesteps_ref, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device
        )

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            None,
        )

        blocks: List[List[int]] = [
            list(range(start, min(start + block_size, batch_size)))
            for start in range(0, batch_size, block_size)
        ]

        attention_kwargs = dict(scene_attention_kwargs or {})
        attention_kwargs.setdefault("num_parts", batch_size)
        attention_kwargs.setdefault("num_frames", 1)

        progress_disable = (
            self._progress_bar_config.get('disable', False)
            if hasattr(self, '_progress_bar_config')
            else False
        )
        total_iters = len(timesteps_ref) * len(blocks)
        self.set_progress_bar_config(
            desc="Scene Denoising",
            ncols=125,
            disable=progress_disable,
        )
        for i, transformer_block in enumerate(self.transformer.blocks):
            print(f"transformer block [{i}] {transformer_block.attn1.processor.__class__}")

        latents_timesteps = {}

        with self.progress_bar(total=total_iters) as progress_bar:
            for block_indices in blocks:
                timesteps_block, _ = retrieve_timesteps(
                    scheduler, num_inference_steps, device
                )
                for t_index, t in enumerate(timesteps_block):
                    update_mask = torch.zeros(
                        (latents.shape[0], 1, 1), dtype=torch.bool, device=device
                    )
                    update_mask[block_indices] = True

                    latents_prev = latents.clone()
                    latents_feed = latents.clone()

                    latents_timesteps[t.item()] = latents_feed.clone()

                    latent_model_input = (
                        torch.cat([latents_feed, latents_feed], dim=0)
                        if do_cfg
                        else latents_feed
                    )
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer.forward(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=encoder_states_full if t_index < 5 else encoder_states,
                        # encoder_hidden_states_masked=encoder_states_masked,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0].to(latents.dtype)

                    if do_cfg:
                        noise_uncond, noise_cond = noise_pred.chunk(2)
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_cond - noise_uncond
                        )

                    latents_next = scheduler.step(
                        noise_pred, t, latents_feed, return_dict=False
                    )[0]
                    latents = torch.where(update_mask, latents_next, latents_prev)
                    progress_bar.update()

        self.vae.set_flash_decoder()
        decode_batch_size = latents.shape[0]
        meshes: List[trimesh.Trimesh] = []
        outputs = []
        self.set_progress_bar_config(
            desc="Scene Decoding",
            ncols=125,
            disable=progress_disable,
        )
        with self.progress_bar(total=decode_batch_size) as progress_bar:
            for i in range(decode_batch_size):
                geometric_func = (
                    lambda x, idx=i: self.vae.decode(
                        latents[idx].unsqueeze(0), sampled_points=x
                    ).sample
                )
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
                except Exception as e:
                    print(f"Warning: scene mesh extraction failed for item {i}: {e}")
                    mesh_v_f = None
                    mesh = None
                outputs.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        if not return_latents_timesteps:        
            return meshes, latents
    
        return meshes, latents, latents_timesteps

    def _tile_history_tensor(self, tensor: torch.Tensor, target: int) -> torch.Tensor:
        if tensor.shape[0] == target:
            return tensor
        repeats = (target + tensor.shape[0] - 1) // tensor.shape[0]
        return tensor.repeat(repeats, 1, 1)[:target]

    def _run_dynamic_stage(
        self,
        frames: List[PipelineImageInput],
        masks: Optional[List[PipelineImageInput]],
        masks_static: Optional[List[PipelineImageInput]],
        scene_latents: torch.Tensor,
        scene_latents_timesteps: Optional[torch.Tensor],
        initial_dynamic_latents: Optional[torch.Tensor],
        attention_kwargs: Optional[Dict[str, Any]],
        num_tokens: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator,
        device: torch.device,
        use_object_only_condition: bool,
        ar_block_size: int,
        history_renoise_sigma: float,
        history_soft_alpha: float,
        history_mode: str,
        prevent_collisions: bool = False,
        dynamic_num_parts: Optional[int] = None,
        dynamic_mix_cutoff: int = 10,
        dynamic_max_memory_frames: int = 6,
    ):

        foreground_frames_per_object: List[List[PipelineImageInput]] = []

        for obj_masks in masks:
            masked_frames = []
            for frame_idx, mask in enumerate(obj_masks):
                masked_frames.append(_apply_mask(frames[frame_idx], mask, keep_foreground=True))
 
            foreground_frames_per_object.append(masked_frames)

        dynamic_embeds_per_object = [self.encode_image(masked_frames, device, 1, use_multi=True)[0] for masked_frames in foreground_frames_per_object]
        if len(dynamic_embeds_per_object) == 1:
            dynamic_embeds = dynamic_embeds_per_object[0].unsqueeze(0)
        else:
            dynamic_embeds = torch.cat([demb.unsqueeze(0) for demb in dynamic_embeds_per_object], dim=0)

        static_frames_masked_with_all_masks = []
        for frame_idx, frame in enumerate(frames):
            combined_mask = _combine_masks([masks[obj_idx][frame_idx] for obj_idx in range(len(masks))] + [masks_static[obj_idx][frame_idx] for obj_idx in range(len(masks_static))])
            static_frames_masked_with_all_masks.append(_apply_mask(frame, combined_mask, keep_foreground=True))

        static_embeds = self.encode_image(static_frames_masked_with_all_masks, device, 1)[0]

        masked_frames[0].save("debug_dynamic_stage_static_masked_frame0.png")
        static_frames_masked_with_all_masks[0].save("debug_dynamic_stage_static_masked_with_all_masks_frame0.png")
        static_frames_masked_with_all_masks[-1].save("debug_dynamic_stage_static_masked_with_all_masks_frame-1.png")

        # N is the number of dynamic parts (objects), F is the number of frames
        N, F, _, _ = dynamic_embeds.shape
        
        if N == 0:
            return [], torch.empty(0, device=device)

        scheduler = self._clone_scheduler()
        timesteps_ref, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            N,
            num_tokens,
            num_channels_latents,
            dynamic_embeds.dtype,
            device,
            generator,
            None,
        )
        # Use the same starting noise for the frames of each object
        latents = latents.unsqueeze(1).repeat(1, F, 1, 1)  # (N, F, num_tokens, C)
        latents = latents.view(N, F, num_tokens, num_channels_latents)

        block_size = max(1, int(ar_block_size))
        blocks = [
            list(range(start, min(start + block_size, F)))
            for start in range(0, F, block_size)
        ]

        history_latents_first_block: Optional[torch.Tensor] = None
        history_dynamic_embeds_first_block: Optional[torch.Tensor] = None
        history_static_embed_first_block: Optional[torch.Tensor] = None
        if initial_dynamic_latents is not None and initial_dynamic_latents.numel() > 0:
            history_latents_first_block = initial_dynamic_latents.to(
                device=device, dtype=latents.dtype
            ).unsqueeze(1)
            if history_latents_first_block.shape[0] != N:
                if history_latents_first_block.shape[0] == 1 and N > 1:
                    history_latents_first_block = history_latents_first_block.repeat(
                        N, 1, 1, 1
                    )
                elif history_latents_first_block.shape[0] < N:
                    repeats = (N + history_latents_first_block.shape[0] - 1) // history_latents_first_block.shape[0]
                    history_latents_first_block = history_latents_first_block.repeat(
                        repeats, 1, 1, 1
                    )[:N]
                else:
                    history_latents_first_block = history_latents_first_block[:N]

            history_dynamic_embeds_first_block = dynamic_embeds[:, :1].clone()
            history_static_embed_first_block = static_embeds[:1].clone()
        initial_history_consumed = False

        spatial_layers = list(getattr(self.transformer, "spatial_global_attn_block_ids", []))
        temporal_layers = list(getattr(self.transformer, "temporal_global_attn_block_ids", []))

        print(f"Dynamic stage with {F} frames, block_size={block_size}, blocks={len(blocks)}")
        print(f"  Spatial attn layers: {spatial_layers}")
        print(f"  Temporal attn layers: {temporal_layers}")

        self._set_attention_blocks(temporal_layers)

        do_cfg = guidance_scale > 1.0

        static_latents_all = (
            scene_latents.to(device=device, dtype=latents.dtype)
            if scene_latents is not None and scene_latents.numel() > 0
            else latents.new_zeros((0, latents.shape[2], latents.shape[3]))
        )

        static_count = static_latents_all.shape[0]

        base_attention_kwargs = dict(attention_kwargs or {})
        base_attention_kwargs = base_attention_kwargs or None

        progress_disable = (
            self._progress_bar_config.get("disable", False)
            if hasattr(self, "_progress_bar_config")
            else False
        )
        total_iters = len(timesteps_ref) * len(blocks)
        self.set_progress_bar_config(desc="4D Denoising", ncols=125, disable=progress_disable)

        MAX_FRAMES_IN_MEMORY = dynamic_max_memory_frames

        with self.progress_bar(total=total_iters) as progress_bar:
            for block_indices in blocks:
                block_dynamic_count = len(block_indices)
                timesteps_block, _ = retrieve_timesteps(scheduler, num_inference_steps, device)

                history_start = max(0, block_indices[0] - block_size)
                history_indices = list(range(history_start, block_indices[0]))
                
                use_initial_history = (
                    history_latents_first_block is not None
                    and not initial_history_consumed
                    and block_indices
                    and block_indices[0] == 0
                )
                extra_history = 1 if use_initial_history else 0
                total_frames = block_dynamic_count + len(history_indices) + extra_history
                max_frames_allowed = MAX_FRAMES_IN_MEMORY + extra_history

                if total_frames > max_frames_allowed:
                    overflow = total_frames - max_frames_allowed
                    if overflow > 0 and len(history_indices) > 0:
                        trim = min(overflow, len(history_indices))
                        history_indices = history_indices[trim:]
                    total_frames = block_dynamic_count + len(history_indices) + extra_history
                    overflow = total_frames - max_frames_allowed
                    if overflow > 0 and use_initial_history:
                        use_initial_history = False
                        extra_history = 0
                        total_frames = block_dynamic_count + len(history_indices)
                        max_frames_allowed = MAX_FRAMES_IN_MEMORY
                        overflow = total_frames - max_frames_allowed
                        if overflow > 0 and len(history_indices) > 0:
                            trim = min(overflow, len(history_indices))
                            history_indices = history_indices[trim:]
                            total_frames = block_dynamic_count + len(history_indices)
                    if total_frames > max_frames_allowed:
                        print(
                            "Warning: dynamic block exceeds memory budget even after trimming history; "
                            "consider reducing ar_block_size."
                        )

                block_dynamic_history_count = len(history_indices) + (1 if use_initial_history else 0)

                if use_initial_history and not initial_history_consumed:
                    print("Using initial dynamic latents as history for first dynamic block")

                for t_index, t in enumerate(timesteps_block):
                    cutoff = t_index > dynamic_mix_cutoff

                    static_embed_slices: List[torch.Tensor] = []
                    if use_initial_history and history_static_embed_first_block is not None:
                        static_embed_slices.append(history_static_embed_first_block)
                    if history_indices:
                        static_embed_slices.append(static_embeds[history_indices])
                    static_embed_slices.append(static_embeds[block_indices])
                    encoder_hidden_states_static = torch.cat(static_embed_slices, dim=0).to(device=device)

                    dynamic_embed_slices: List[torch.Tensor] = []
                    if use_initial_history and history_dynamic_embeds_first_block is not None:
                        dynamic_embed_slices.append(history_dynamic_embeds_first_block)
                    if history_indices:
                        dynamic_embed_slices.append(dynamic_embeds[:, history_indices])
                    dynamic_embed_slices.append(dynamic_embeds[:, block_indices])
                    encoder_hidden_states_dynamic = torch.cat(dynamic_embed_slices, dim=1).to(device=device)

                    if not cutoff:
                        # replace dynamic hidden states with static
                        encoder_hidden_states_dynamic[:, :] = encoder_hidden_states_static.unsqueeze(0).repeat(N, 1, 1, 1)

                    # Use all dynamic objects
                    dynamic_latent_slices: List[torch.Tensor] = []
                    if use_initial_history and history_latents_first_block is not None:
                        dynamic_latent_slices.append(history_latents_first_block)
                    if history_indices:
                        dynamic_latent_slices.append(latents[:, history_indices].clone())
                    dynamic_latent_slices.append(latents[:, block_indices].clone())
                    dynamic_latents = torch.cat(dynamic_latent_slices, dim=1).to(device=device)

                    dynamic_len = dynamic_latents.shape[0]
                    dynamic_frame_len = dynamic_latents.shape[1]
                    static_len = scene_latents.shape[0]
                    _, _, T, C = dynamic_latents.shape
                    _, _, K, D = encoder_hidden_states_dynamic.shape

                    latents_matrix = torch.zeros((dynamic_len + static_len, dynamic_frame_len, T, C)).to(dynamic_latents.device).to(dynamic_latents.dtype)
                    encoder_hidden_states_matrix = torch.zeros((dynamic_len + static_len, dynamic_frame_len, K, D)).to(dynamic_latents.device).to(dynamic_latents.dtype)

                    latents_matrix[:static_len] = static_latents_all.unsqueeze(1).repeat(1, dynamic_frame_len, 1, 1)
                    latents_matrix[static_len:] = dynamic_latents

                    encoder_hidden_states_matrix[:static_len] = encoder_hidden_states_static.unsqueeze(0).repeat(static_len, 1, 1, 1)
                    encoder_hidden_states_matrix[static_len:] = encoder_hidden_states_dynamic

                    timestep = t.expand(1)
                    
                    if t_index == 0:
                        print("Latents/encoder_hidden_states matrix shapes:")
                        print(f" latents_matrix {latents_matrix.shape}, encoder_hidden_states_matrix {encoder_hidden_states_matrix.shape}")
                        print(f" static_latents_all {static_latents_all.shape}, dynamic_latents {dynamic_latents.shape}")
                        print(f" encoder_hidden_states_static {encoder_hidden_states_static.shape}, encoder_hidden_states_dynamic {encoder_hidden_states_dynamic.shape}")

                    # print(f" Denoising block frames {block_indices}, history frames {history_indices}, timestep {t.item()}, cutoff {cutoff}, static_count {static_count}, dynamic_count {block_dynamic_history_count + block_dynamic_count}")
                    # print(f"   latents_matrix {latents_matrix.shape}, encoder_hidden_states_matrix {encoder_hidden_states_matrix.shape}")
                    # print(f"   static_latents_all {static_latents_all.shape}, dynamic_latents {dynamic_latents.shape}")
                    # print(f"   encoder_hidden_states_static {encoder_hidden_states_static.shape}, encoder_hidden_states_dynamic {encoder_hidden_states_dynamic.shape}")

                    transformer_output = self.transformer.forward_matrix(
                        latents_matrix,
                        timestep,
                        encoder_hidden_states_matrix,
                        static_count=static_count,
                        dynamic_count=block_dynamic_history_count + block_dynamic_count,
                        return_dict=False,
                        cutoff=cutoff
                    )[0]

                    transformer_output_uncond = self.transformer.forward_matrix(
                        latents_matrix,
                        timestep,
                        torch.zeros_like(encoder_hidden_states_matrix),
                        static_count=static_count,
                        dynamic_count=block_dynamic_history_count + block_dynamic_count,
                        return_dict=False,
                        cutoff=cutoff
                    )[0] if do_cfg else None

                    if do_cfg:
                        noise_uncond, noise_cond = transformer_output_uncond, transformer_output
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_cond - noise_uncond
                        )
                    else:
                        noise_pred = transformer_output

                    noise_dynamic = noise_pred[static_len:]

                    current_dynamic_count = noise_dynamic.shape[0]
                    current_dynamic_frame_count = noise_dynamic.shape[1]

                    noise_dynamic = noise_dynamic.view(current_dynamic_count * current_dynamic_frame_count, T, C)
                    dynamic_latents = dynamic_latents.view(current_dynamic_count * current_dynamic_frame_count, T, C)

                    latents_next = scheduler.step(noise_dynamic, t, dynamic_latents, return_dict=False)[0]
                    if latents_next.dtype != latents.dtype:
                        latents_next = latents_next.to(latents.dtype)

                    latents_next = latents_next.view(current_dynamic_count, current_dynamic_frame_count, T, C)
                    
                    latents[:, block_indices] = latents_next[:, -block_dynamic_count:]
                    progress_bar.update()

                if use_initial_history:
                    initial_history_consumed = True
        
        self.vae.set_flash_decoder()

        static_fields = []
        for i in tqdm(range(static_latents_all.shape[0]), desc="Extracting static fields", disable=progress_disable):
            geometric_func = lambda x, idx=i: self.vae.decode(static_latents_all[idx].unsqueeze(0), sampled_points=x).sample
            try:
                field = hierarchical_extract_fields(
                    geometric_func,
                    device,
                    dtype=static_latents_all.dtype,
                    bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                    dense_octree_depth=8,
                    hierarchical_octree_depth=9,
                    max_num_expanded_coords=1e8,
                )

                static_fields.append(field)
            except Exception as e:
                print(f"Warning: dynamic field extraction failed for item {i}: {e}")

        dynamic_fields = []
        for i in tqdm(range(latents.shape[0]), desc="Extracting dynamic fields", disable=progress_disable):
            dynamic_fields_per_part = []
            for j in tqdm(range(latents.shape[1]), desc=f" Part {i} frames", disable=progress_disable):
                if torch.isnan(latents[i, j]).any():
                    raise ValueError(f"NaN detected in latents at dynamic part {i}, frame {j}")
                
                geometric_func = lambda x, idx=j: self.vae.decode(latents[i, idx].unsqueeze(0), sampled_points=x).sample
                try:
                    field = hierarchical_extract_fields(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        dense_octree_depth=8,
                        hierarchical_octree_depth=9,
                        max_num_expanded_coords=1e8,
                    )

                    dynamic_fields_per_part.append(field)
                except Exception as e:
                    print(f"Warning: dynamic field extraction failed for item {i}: {e}")
                    dynamic_fields_per_part.append(None)
            
            dynamic_fields.append(dynamic_fields_per_part)

        static_meshes_per_frame = []
        dynamic_meshes_per_frame = []

        if not prevent_collisions:
            static_meshes = []
            for static_field in static_fields:
                static_meshes.append(field_to_mesh(
                    static_field,
                    bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                    octree_depth=9,
                    device=device,
                ))
            static_meshes_per_frame = [static_meshes for _ in range(F)]
        else:
            static_meshes = []
            for frame in range(F):
                frame_static_meshes = []
                for static_field in static_fields:
                    current_field = static_field

                    # Use all the dynamic ones from this frame
                    for obj_idx in range(len(dynamic_fields)):
                        current_field = eliminate_collisions(dynamic_fields[obj_idx][frame], current_field)

                    frame_static_meshes.append(field_to_mesh(
                        current_field,
                        bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                        octree_depth=9,
                        device=device,
                    ))
                static_meshes.append(frame_static_meshes)

            static_meshes_per_frame = static_meshes
            
        for frame in range(F):
            frame_dynamic_meshes = []
            for obj_idx in range(len(dynamic_fields)):
                frame_dynamic_meshes.append(field_to_mesh(
                    dynamic_fields[obj_idx][frame],
                    bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
                    octree_depth=9,
                    device=device,
                ))
            dynamic_meshes_per_frame.append(frame_dynamic_meshes)

        return static_meshes_per_frame, dynamic_meshes_per_frame

    def _render_views_around_mesh(
        self,
        mesh: trimesh.Scene,
        path: str,
        fps: int = 12,
        render_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[PIL.Image.Image]:
        frames = render_views_around_mesh(
            mesh,
            num_views=render_kwargs.get('num_views', 36),
            radius=render_kwargs.get('radius', 3.5),
            image_size=render_kwargs.get('image_size', (512, 512)),
        )

        export_renderings(frames, path, fps=fps)

    def _render_animation(
        self,
        scene_meshes: List[trimesh.Trimesh],
        static_meshes_per_frame: List[List[trimesh.Trimesh]],
        dynamic_meshes_per_frame: List[List[trimesh.Trimesh]],
        animation_path: Optional[str],
        fps: int,
        insert_rotation_every: int,
        render_kwargs: Optional[Dict[str, Any]],
        gt_frames: Optional[List[PipelineImageInput]] = None,
    ) -> Optional[str]:
        if animation_path is None:
            return None
        render_kwargs = render_kwargs or {}
        target_size = render_kwargs.get("image_size", (512, 512))
        if isinstance(target_size, (int, float)):
            target_size = (int(target_size), int(target_size))
        else:
            target_size = tuple(int(v) for v in target_size)

        def _to_pil_image(image: PipelineImageInput) -> PIL.Image.Image:
            if isinstance(image, PIL.Image.Image):
                return image.convert("RGB")
            if isinstance(image, torch.Tensor):
                tensor = image.detach().cpu()
                if tensor.ndim == 4:
                    tensor = tensor[0]
                if tensor.ndim == 3:
                    if tensor.shape[0] == 1:
                        tensor = tensor.repeat(3, 1, 1)
                    elif tensor.shape[0] != 3:
                        raise ValueError(f"Unsupported tensor shape for image conversion: {tensor.shape}")
                    array = (
                        tensor.float()
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .numpy()
                    )
                    return PIL.Image.fromarray(array)
                if tensor.ndim == 2:
                    array = (
                        tensor.float()
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .to(torch.uint8)
                        .numpy()
                    )
                    array = np.repeat(array[..., None], 3, axis=2)
                    return PIL.Image.fromarray(array)
                raise ValueError(f"Unsupported tensor ndim for image conversion: {tensor.ndim}")
            if isinstance(image, np.ndarray):
                array = image
                if array.ndim == 2:
                    array = np.repeat(array[..., None], 3, axis=2)
                elif array.ndim == 3:
                    if array.shape[2] == 1:
                        array = np.repeat(array, 3, axis=2)
                    elif array.shape[2] not in (3, 4):
                        raise ValueError(f"Unsupported ndarray channel count: {array.shape[2]}")
                else:
                    raise ValueError(f"Unsupported ndarray ndim for image conversion: {array.ndim}")
                if array.dtype != np.uint8:
                    array = np.clip(array, 0.0, 1.0)
                    array = (array * 255.0).round().astype(np.uint8)
                if array.shape[-1] == 4:
                    array = array[..., :3]
                return PIL.Image.fromarray(array)
            raise TypeError(f"Unsupported image type for conversion: {type(image)}")

        base_scene = (
            get_colored_mesh_composition(scene_meshes, is_random=False) if len(scene_meshes) > 0 else None
        )

        static_colors: List[np.ndarray] = []
        if scene_meshes:
            for idx, mesh in enumerate(scene_meshes):
                vertex_colors = getattr(mesh.visual, "vertex_colors", None)
                if vertex_colors is not None and len(vertex_colors) > 0:
                    color_rgb = np.array(vertex_colors[0][:3], dtype=np.uint8)
                else:
                    color_rgb = np.array(DEFAULT_PART_COLORS[idx % len(DEFAULT_PART_COLORS)], dtype=np.uint8)
                static_colors.append(color_rgb)

        dynamic_colors: Dict[int, np.ndarray] = {}
        dynamic_palette_offset = len(static_colors)

        def _color_for_dynamic_index(index: int) -> np.ndarray:
            if index in dynamic_colors:
                return dynamic_colors[index]
            color_idx = (dynamic_palette_offset + index) % len(DEFAULT_PART_COLORS)
            color_rgb = np.array(DEFAULT_PART_COLORS[color_idx], dtype=np.uint8)
            dynamic_colors[index] = color_rgb
            return color_rgb

        def _apply_color(mesh: trimesh.Trimesh, color_rgb: np.ndarray) -> trimesh.Trimesh:
            colored = mesh.copy()
            if colored.vertices.size == 0:
                return colored
            rgba = np.array([color_rgb[0], color_rgb[1], color_rgb[2], 255], dtype=np.uint8)
            vertex_colors = np.tile(rgba, (colored.vertices.shape[0], 1))
            face_colors = np.tile(rgba, (colored.faces.shape[0], 1))
            visuals = trimesh.visual.ColorVisuals(mesh=colored, vertex_colors=vertex_colors, face_colors=face_colors)
            colored.visual = visuals
            return colored

        total_frames = max(len(static_meshes_per_frame), len(dynamic_meshes_per_frame))
        if total_frames == 0 and base_scene is not None:
            total_frames = 1

        composite_meshes: List[trimesh.Scene] = []
        for frame_idx in range(total_frames):
            scene = trimesh.Scene()

            use_base_scene = True
            if frame_idx < len(static_meshes_per_frame):
                static_parts = static_meshes_per_frame[frame_idx] or []
                if static_parts:
                    use_base_scene = False
                    for part_idx, static_mesh in enumerate(static_parts):
                        if static_mesh is None:
                            continue
                        color_idx = part_idx if part_idx < len(static_colors) else part_idx % len(DEFAULT_PART_COLORS)
                        if color_idx < len(static_colors):
                            color_rgb = static_colors[color_idx]
                        else:
                            color_rgb = np.array(
                                DEFAULT_PART_COLORS[color_idx % len(DEFAULT_PART_COLORS)], dtype=np.uint8
                            )
                        scene.add_geometry(_apply_color(static_mesh, color_rgb))

            if use_base_scene and base_scene is not None:
                scene = base_scene.copy() if isinstance(base_scene, trimesh.Scene) else trimesh.Scene(base_scene)

            if frame_idx < len(dynamic_meshes_per_frame):
                dynamic_parts = dynamic_meshes_per_frame[frame_idx] or []
                for part_idx, dyn_mesh in enumerate(dynamic_parts):
                    if dyn_mesh is None:
                        continue
                    color_rgb = _color_for_dynamic_index(part_idx)
                    scene.add_geometry(_apply_color(dyn_mesh, color_rgb))

            composite_meshes.append(scene)

        rendered_frames_raw = render_sequence_fixed_camera(composite_meshes, **render_kwargs)

        rendered_frames: List[PIL.Image.Image] = []
        for idx, frame in enumerate(rendered_frames_raw):
            try:
                pil_frame = _to_pil_image(frame)
                if pil_frame.size != target_size:
                    pil_frame = pil_frame.resize(target_size, PIL.Image.LANCZOS)
                rendered_frames.append(pil_frame)
            except Exception as exc:
                print(f"Warning: failed to convert rendered frame {idx} to PIL: {exc}")

        gt_images: List[Optional[PIL.Image.Image]] = [None] * len(rendered_frames)
        if gt_frames:
            max_pairs = min(len(gt_frames), len(rendered_frames))
            for idx in range(max_pairs):
                try:
                    gt_img = _to_pil_image(gt_frames[idx])
                except Exception as exc:
                    print(f"Warning: failed to convert GT frame {idx} to PIL: {exc}")
                    continue
                if gt_img.size != target_size:
                    gt_img = gt_img.resize(target_size, PIL.Image.LANCZOS)
                gt_images[idx] = gt_img

        def _compose_side_by_side(left: PIL.Image.Image, right: PIL.Image.Image) -> PIL.Image.Image:
            if left.size != target_size:
                left = left.resize(target_size, PIL.Image.LANCZOS)
            if right.size != target_size:
                right = right.resize(target_size, PIL.Image.LANCZOS)
            canvas = PIL.Image.new("RGB", (target_size[0] * 2, target_size[1]))
            canvas.paste(left, (0, 0))
            canvas.paste(right, (target_size[0], 0))
            return canvas

        output_frames: List[PIL.Image.Image] = []
        last_gt_image: Optional[PIL.Image.Image] = None
        for idx, frame in enumerate(rendered_frames):
            gt_img = gt_images[idx] if idx < len(gt_images) else None
            if gt_img is not None:
                last_gt_image = gt_img
            if last_gt_image is None:
                last_gt_image = frame
            try:
                combined = _compose_side_by_side(last_gt_image, frame)
                output_frames.append(combined)
            except Exception as exc:
                print(f"Warning: failed to compose comparison frame {idx}: {exc}")
                output_frames.append(frame)

            if insert_rotation_every and insert_rotation_every > 0:
                if (idx + 1) % insert_rotation_every == 0:
                    rotation_frames = render_views_around_mesh(
                        composite_meshes[idx],
                        num_views=render_kwargs.get('num_views', 36),
                        radius=render_kwargs.get('radius', 3.5),
                        image_size=render_kwargs.get('image_size', (512, 512)),
                    )
                    for rot_idx, rot_frame in enumerate(rotation_frames):
                        try:
                            rot_image = _to_pil_image(rot_frame)
                            if rot_image.size != target_size:
                                rot_image = rot_image.resize(target_size, PIL.Image.LANCZOS)
                            ref_gt = last_gt_image if last_gt_image is not None else frame
                            combined_rot = _compose_side_by_side(ref_gt, rot_image)
                            output_frames.append(combined_rot)
                        except Exception as exc:
                            print(f"Warning: failed to convert rotation frame {rot_idx} to PIL: {exc}")

        if not output_frames:
            remaining_gt = [img for img in gt_images if img is not None]
            if remaining_gt:
                for img_idx, img in enumerate(remaining_gt):
                    try:
                        output_frames.append(_compose_side_by_side(img, img))
                    except Exception as exc:
                        print(f"Warning: failed to compose fallback GT frame {img_idx}: {exc}")
                        output_frames.append(img)

        if len(output_frames) == 0:
            return None
        export_renderings(output_frames, animation_path, fps=fps)
        return animation_path

    @torch.no_grad()
    def __call__(
        self,
        first_frame: PipelineImageInput,
        first_frame_mask: PipelineImageInput,
        frames: List[PipelineImageInput],
        masks: Optional[List[PipelineImageInput]] = None,
        masks_static: Optional[List[PipelineImageInput]] = None,
        all_masks: Optional[List[PipelineImageInput]] = None,
        num_tokens: int = 2048,
        scene_inference_steps: int = 50,
        dynamic_inference_steps: int = 50,
        guidance_scale_scene: float = 7.0,
        guidance_scale_dynamic: float = 7.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_object_only_condition: bool = True,
        dynamic_ar_block_size: int = 8,
        history_renoise_sigma: float = 0.0,
        history_soft_alpha: float = 0.1,
        history_mode: str = "fixed",
        animation_path: Optional[str] = None,
        animation_fps: int = 12,
        insert_rotation_every: int = 0,
        render_kwargs: Optional[Dict[str, Any]] = None,
        scene_attention_ids: Optional[List[int]] = None,
        dynamic_attention_ids: Optional[List[int]] = None,
        scene_num_parts: Optional[int] = None,
        dynamic_num_parts: Optional[int] = None,
        scene_block_size: int = 4,
        use_latents_with_timesteps: bool = False,
        return_dict: bool = True,
        prevent_collisions: bool = False,
        first_frame_index: int = 0,
        scene_mix_cutoff: int = 10,
        dynamic_mix_cutoff: int = 10,
        dynamic_max_memory_frames: int = 6

    ):
        if len(frames) == 0:
            raise ValueError("At least one frame is required for inference")

        print("Starting PartCrafter 3D4D inference"
              f"\n  Number of tokens: {num_tokens}"
              f"\n  First frame index: {first_frame_index}"
              f"\n  Scene mix cutoff: {scene_mix_cutoff}"
              f"\n  Dynamic mix cutoff: {dynamic_mix_cutoff}"
              f"\n  Dynamic max memory frames: {dynamic_max_memory_frames}"
        )
        
        device = next(self.transformer.parameters()).device
        scene_attention_ids = scene_attention_ids or list(getattr(self.transformer, 'spatial_global_attn_block_ids', []))
        dynamic_attention_ids = dynamic_attention_ids or list(getattr(self.transformer, 'temporal_global_attn_block_ids', []))
        
        first_frame_masked = _apply_mask(first_frame, first_frame_mask, keep_foreground=True, dilation_radius=0, erode_radius=4)
        first_frame_masked.save("scene_input.png")
        first_frame_mask.save("scene_mask.png")

        scene_part_count = scene_num_parts

        print(f"Running PartCrafter 3D4D with {scene_part_count} scene parts and {len(frames)} frames"  )

        if scene_part_count != 0:
            self._set_attention_blocks(scene_attention_ids)
            scene_attention_kwargs = dict({})
            scene_attention_kwargs.setdefault("num_parts", scene_part_count)
            scene_attention_kwargs.setdefault("num_frames", 1)

            scene_meshes, static_latents, dynamic_latents = self._run_joint_scene_stage(
                first_frame,
                first_frame_mask,
                masks,
                masks_static,
                all_masks,
                guidance_scale_scene,
                block_size=scene_block_size,
                num_static=scene_part_count,
                num_dynamic=dynamic_num_parts,
                generator=generator,
                device=device,
                num_tokens=num_tokens,
                first_frame_index=first_frame_index,
                scene_mix_cutoff=scene_mix_cutoff,
            )

            merged_scene = (
                get_colored_mesh_composition(scene_meshes, is_random=False) if len(scene_meshes) > 0 else None
            )

            self._render_views_around_mesh(
                merged_scene,
                animation_path.replace(".gif", "_scene.gif"),
                animation_fps,
                render_kwargs,
            )

            print(animation_path.replace(".gif", "_scene.gif"))
        else:
            scene_meshes = []
            static_latents = torch.empty(0, device=device)
            dynamic_latents = torch.empty(0, device=device)

        # Dynamic stage
        self._set_attention_blocks(dynamic_attention_ids)
        dynamic_attention_kwargs = dict({})
        dynamic_attention_kwargs.setdefault("num_parts", scene_part_count + 1)
        dynamic_attention_kwargs.setdefault("num_frames", len(frames))
        static_meshes_per_frame, dynamic_meshes_per_frame = self._run_dynamic_stage(
            frames,
            masks,
            masks_static,
            static_latents,
            None,
            dynamic_latents,
            dynamic_attention_kwargs,
            num_tokens,
            dynamic_inference_steps,
            guidance_scale_dynamic,
            generator,
            device,
            use_object_only_condition,
            dynamic_ar_block_size,
            history_renoise_sigma,
            history_soft_alpha,
            history_mode,
            dynamic_num_parts=dynamic_num_parts,
            dynamic_mix_cutoff=dynamic_mix_cutoff,
            dynamic_max_memory_frames=dynamic_max_memory_frames,
        )
        animation_file = self._render_animation(
            scene_meshes,
            static_meshes_per_frame,
            dynamic_meshes_per_frame,
            animation_path,
            animation_fps,
            insert_rotation_every,
            render_kwargs,
            frames,
        )
        output = PartCrafter3D4DOutput(
            scene_meshes=scene_meshes,
            static_meshes_per_frame=static_meshes_per_frame,
            dynamic_meshes=dynamic_meshes_per_frame,
            animation_path=animation_file,
        )
        if not return_dict:
            return output.scene_meshes, output.dynamic_meshes
        
        return output

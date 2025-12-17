"""
COM4D training entry point.

This script orchestrates dataset construction, model initialization, and the full
training loop for COM4DDiTModel along with its VAE, encoders, schedulers, and
Auxiliary utilities such as EMA and attention processor switching. The overall
workflow, argument structure, and much of the supporting boilerplate were mostly
taken from the original PartCrafter training implementation:
https://github.com/wgsxm/PartCrafter/blob/main/src/train_partcrafter.py
"""


import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")  # ignore all warnings
import diffusers.utils.logging as diffusion_logging
diffusion_logging.set_verbosity_error()  # ignore diffusers warnings

from src.utils.typing_utils import *

import argparse
import logging
import time
import math
import gc
import random
from packaging import version

import trimesh
from PIL import Image
import numpy as np
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as tF
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger as get_accelerate_logger
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3
)

from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from src.schedulers import RectifiedFlowScheduler
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import COM4DDiTModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline, FourDCrafterPipeline

# Datasets: 3D parts and 4D frames (frames-as-parts)
from src.datasets.objaverse_part import (
    ObjaversePartDataset as ObjaversePartDataset3D,
    BatchedObjaversePartDataset as BatchedObjaversePartDataset3D,
    ObjaversePartDatasetOriginal as ObjaversePartDatasetOriginal,
    BatchedObjaversePartDatasetOriginal as BatchedObjaversePartDatasetOriginal,
)
from src.datasets.animated_frame import (
    ObjaversePartDataset as ObjaversePartDataset4D,
    BatchedObjaversePartDataset as BatchedObjaversePartDataset4D,
)
from src.datasets import (
    MultiEpochsDataLoader,
    yield_forever,
)
from src.utils.data_utils import get_colored_mesh_composition
from src.utils.train_utils import (
    MyEMAModel, 
    get_configs,
    get_optimizer,
    get_lr_scheduler,
    save_experiment_params,
    save_model_architecture,
)
from src.utils.render_utils import (
    render_views_around_mesh, 
    render_normal_views_around_mesh, 
    make_grid_for_images_or_videos,
    export_renderings
)
import copy

from src.models.attention_processor import (
    COM4DAttnProcessor,
    TripoSGAttnProcessor2_0,
)

def main():
    PROJECT_NAME = "COM4D"

    parser = argparse.ArgumentParser(
        description="Train a diffusion model for 3D object generation",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--resume_from_iter",
        type=int,
        default=None,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )
    parser.add_argument(
        "--offline_wandb",
        action="store_true",
        help="Use offline WandB for experiment tracking"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="The max iteration step for training"
    )
    parser.add_argument(
        "--max_val_steps",
        type=int,
        default=2,
        help="The max iteration step for validation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of processed spawned by the batch provider"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Pin memory for the data loader"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model for training"
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale lr with total batch size (base batch size: 256)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.,
        help="Max gradient norm for gradient clipping"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Type of mixed precision training"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    parser.add_argument(
        "--val_guidance_scales",
        type=list,
        nargs="+",
        default=[7.0],
        help="CFG scale used for validation"
    )

    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Use DeepSpeed for training"
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],  # https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        help="ZeRO stage type for DeepSpeed"
    )

    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train from scratch"
    )
    parser.add_argument(
        "--load_pretrained_model",
        type=str,
        default=None,
        help="Tag of a pretrained COM4DDiTModel in this project"
    )
    parser.add_argument(
        "--load_pretrained_model_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained COM4DDiTModel checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()
    # Parse the config file
    configs = get_configs(args.config, extras)  # change yaml configs by `extras`

    args.val_guidance_scales = [float(x[0]) if isinstance(x, list) else float(x) for x in args.val_guidance_scales]
    if args.max_val_steps > 0: 
        # If enable validation, the max_val_steps must be a multiple of nrow
        # Always keep validation batchsize 1
        divider = configs["val"]["nrow"]
        args.max_val_steps = max(args.max_val_steps, divider)
        if args.max_val_steps % divider != 0:
            args.max_val_steps = (args.max_val_steps // divider + 1) * divider

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
        
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    eval_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = get_accelerate_logger(__name__, log_level="INFO")
    file_handler = logging.FileHandler(os.path.join(exp_dir, "log.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.logger.addHandler(file_handler)
    logger.logger.propagate = True  # propagate to the root logger (console)

    # Set DeepSpeed config
    if args.use_deepspeed:
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_clipping=args.max_grad_norm,
            zero_stage=int(args.zero_stage),
            offload_optimizer_device="cpu",  # hard-coded here, TODO: make it configurable
        )
    else:
        deepspeed_plugin = None

    # Initialize the accelerator
    accelerator = Accelerator(
        project_dir=exp_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,  # batch size per GPU
        dataloader_config=DataLoaderConfiguration(non_blocking=args.pin_memory),
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info(f"Accelerator state:\n{accelerator.state}\n")

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Build two dataset configs by overriding the 'dataset' key
    cfgs_3d = copy.deepcopy(configs)
    cfgs_4d = copy.deepcopy(configs)
    if 'dataset_3d' in configs:
        cfgs_3d['dataset'] = copy.deepcopy(configs['dataset_3d'])
    if 'dataset_4d' in configs:
        cfgs_4d['dataset'] = copy.deepcopy(configs['dataset_4d'])

    # Train/Val: 3D
    train_dataset_3d = BatchedObjaversePartDataset3D(
        configs=cfgs_3d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )
    val_dataset_3d = ObjaversePartDataset3D(
        configs=cfgs_3d,
        training=False
    )
    train_loader_3d = MultiEpochsDataLoader(
        train_dataset_3d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset_3d.collate_fn,
    )
    val_loader_3d = MultiEpochsDataLoader(
        val_dataset_3d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    random_val_loader_3d = MultiEpochsDataLoader(
        val_dataset_3d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    # Train/Val: 4D (frames as parts)
    train_dataset_4d = BatchedObjaversePartDataset4D(
        configs=cfgs_4d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=True,
        training=True,
    )
    val_dataset_4d = ObjaversePartDataset4D(
        configs=cfgs_4d,
        training=False
    )
    train_loader_4d = MultiEpochsDataLoader(
        train_dataset_4d,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset_4d.collate_fn,
    )
    val_loader_4d = MultiEpochsDataLoader(
        val_dataset_4d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    random_val_loader_4d = MultiEpochsDataLoader(
        val_dataset_4d,
        batch_size=configs["val"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )

    objaverse_dataset_configs = copy.deepcopy(configs)
    objaverse_dataset_configs['dataset'] = configs['dataset_objaverse']
    objaverse_dataset = BatchedObjaversePartDatasetOriginal(
        configs=objaverse_dataset_configs,
        batch_size=configs["train"]["batch_size_per_gpu"],
        is_main_process=accelerator.is_main_process,
        shuffle=False,
        training=True,
    )
    objaverse_train_loader = MultiEpochsDataLoader(
        objaverse_dataset,
        batch_size=configs["train"]["batch_size_per_gpu"],
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
        collate_fn=objaverse_dataset.collate_fn,
    )

    logger.info(
        f"Loaded 3D [{len(train_dataset_3d)}] train / [{len(val_dataset_3d)}] val; "
        f"4D [{len(train_dataset_4d)}] train / [{len(val_dataset_4d)}] val; "
        f"Objaverse [{len(objaverse_dataset)}] train\n"
    )

    single_object_reg_prob = float(configs["train"].get("single_object_regularizer_prob", 0.0))
    single_object_configs = [
        cfg for cfg in getattr(objaverse_dataset, 'data_configs', [])
        if isinstance(cfg, dict) and cfg.get('num_parts', 0) == 1
    ]
    if accelerator.is_main_process:
        logger.info(
            f"Single-object regularizer: prob={single_object_reg_prob} with {len(single_object_configs)} Objaverse candidates\n"
        )

    # Compute the effective batch size and scale learning rate
    total_batch_size = configs["train"]["batch_size_per_gpu"] * \
        accelerator.num_processes * args.gradient_accumulation_steps
    configs["train"]["total_batch_size"] = total_batch_size
    if args.scale_lr:
        configs["optimizer"]["lr"] *= (total_batch_size / 256)
        configs["lr_scheduler"]["max_lr"] = configs["optimizer"]["lr"]
    
    # Initialize the model
    logger.info("Initializing the model...")
    vae = TripoSGVAEModel.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="vae"
    )
    feature_extractor_dinov2 = BitImageProcessor.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="feature_extractor_dinov2"
    )
    image_encoder_dinov2 = Dinov2Model.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="image_encoder_dinov2"
    )

    enable_part_embedding = configs["model"]["transformer"].get("enable_part_embedding", True)
    enable_frame_embedding = configs["model"]["transformer"].get("enable_frame_embedding", True)
    enable_local_cross_attn = configs["model"]["transformer"].get("enable_local_cross_attn", True)
    enable_global_cross_attn = configs["model"]["transformer"].get("enable_global_cross_attn", True)
    enable_static_embedding = configs["model"]["transformer"].get("enable_static_embedding", True)
    enable_dynamic_embedding = configs["model"]["transformer"].get("enable_dynamic_embedding", True)
    # Separate spatial and temporal global-attn block ids; fallback to global_attn_block_ids
    spatial_global_attn_block_ids = configs["model"]["transformer"].get("spatial_global_attn_block_ids", None)
    if spatial_global_attn_block_ids is not None:
        spatial_global_attn_block_ids = list(spatial_global_attn_block_ids)
    temporal_global_attn_block_ids = configs["model"]["transformer"].get("temporal_global_attn_block_ids", None)
    if temporal_global_attn_block_ids is not None:
        temporal_global_attn_block_ids = list(temporal_global_attn_block_ids)
    # Back-compat: if neither provided, use global_attn_block_ids/range
    fallback_global_ids = configs["model"]["transformer"].get("global_attn_block_ids", None)
    if fallback_global_ids is not None:
        fallback_global_ids = list(fallback_global_ids)
    global_attn_block_id_range = configs["model"]["transformer"].get("global_attn_block_id_range", None)
    if global_attn_block_id_range is not None:
        global_attn_block_id_range = list(global_attn_block_id_range)
        if fallback_global_ids is None:
            fallback_global_ids = list(range(global_attn_block_id_range[0], global_attn_block_id_range[1] + 1))
    if spatial_global_attn_block_ids is None:
        spatial_global_attn_block_ids = fallback_global_ids or []
    if temporal_global_attn_block_ids is None:
        temporal_global_attn_block_ids = fallback_global_ids or []
    if args.from_scratch:
        logger.info(f"Initialize COM4DDiTModel from scratch\n")
        transformer = COM4DDiTModel.from_config(
            os.path.join(
                configs["model"]["pretrained_model_name_or_path"],
                "transformer"
            ), 
            enable_part_embedding=enable_part_embedding,
            enable_frame_embedding=enable_frame_embedding,
            enable_static_embedding=enable_static_embedding,
            enable_dynamic_embedding=enable_dynamic_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=spatial_global_attn_block_ids,
            spatial_global_attn_block_ids=spatial_global_attn_block_ids,
            temporal_global_attn_block_ids=temporal_global_attn_block_ids,
            global_attn_block_id_range=None,
        )
    elif args.load_pretrained_model is None:
        logger.info(f"Load pretrained TripoSGDiTModel to initialize COM4DDiTModel from [{configs['model']['pretrained_model_name_or_path']}]\n")
        transformer, loading_info = COM4DDiTModel.from_pretrained(
            configs["model"]["pretrained_model_name_or_path"],
            subfolder="transformer",
            low_cpu_mem_usage=False, 
            output_loading_info=True, 
            enable_part_embedding=enable_part_embedding,
            enable_frame_embedding=enable_frame_embedding,
            enable_static_embedding=enable_static_embedding,
            enable_dynamic_embedding=enable_dynamic_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=spatial_global_attn_block_ids,
            global_attn_block_id_range=None,
        )
    else:
        logger.info(f"Load COM4DDiTModel EMA checkpoint from [{args.load_pretrained_model}] iteration [{args.load_pretrained_model_ckpt:06d}]\n")
        path = os.path.join(
            args.output_dir,
            args.load_pretrained_model, 
            "checkpoints", 
            f"{args.load_pretrained_model_ckpt:06d}"
        )
        transformer, loading_info = COM4DDiTModel.from_pretrained(
            path, 
            subfolder="transformer_ema",
            low_cpu_mem_usage=False, 
            output_loading_info=True, 
            enable_part_embedding=enable_part_embedding,
            enable_frame_embedding=enable_frame_embedding,
            enable_static_embedding=enable_static_embedding,
            enable_dynamic_embedding=enable_dynamic_embedding,
            enable_local_cross_attn=enable_local_cross_attn,
            enable_global_cross_attn=enable_global_cross_attn,
            global_attn_block_ids=spatial_global_attn_block_ids,
            global_attn_block_id_range=None,
        )
    if not args.from_scratch:
        for v in loading_info.values():
            if v and len(v) > 0:
                logger.info(f"Loading info of COM4DDiTModel: {loading_info}\n")
                break

    # Transformer parameters
    transformer.enable_part_embedding = enable_part_embedding
    transformer.enable_frame_embedding = enable_frame_embedding
    transformer.enable_static_embedding = enable_static_embedding
    transformer.enable_dynamic_embedding = enable_dynamic_embedding
    transformer.enable_local_cross_attn = enable_local_cross_attn
    transformer.enable_global_cross_attn = enable_global_cross_attn
    transformer.spatial_global_attn_block_ids = list(spatial_global_attn_block_ids)
    transformer.temporal_global_attn_block_ids = list(temporal_global_attn_block_ids)

    noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    if args.use_ema:
        ema_transformer = MyEMAModel(
            transformer.parameters(),
            model_cls=COM4DDiTModel,
            model_config=transformer.config,
            **configs["train"]["ema_kwargs"]
        )

    # Freeze VAE and image encoder
    vae.requires_grad_(False)
    image_encoder_dinov2.requires_grad_(False)
    vae.eval()
    image_encoder_dinov2.eval()

    trainable_modules = configs["train"].get("trainable_modules", None)
    if trainable_modules is None:
        transformer.requires_grad_(True)
    else:
        trainable_module_names = []
        transformer.requires_grad_(False)
        for name, module in transformer.named_modules():
            for module_name in tuple(trainable_modules.split(",")):
                if module_name in name:
                    for params in module.parameters():
                        params.requires_grad = True
                    trainable_module_names.append(name)
        logger.info(f"Trainable parameter names: {trainable_module_names}\n")

    # transformer.enable_xformers_memory_efficient_attention()  # use `tF.scaled_dot_product_attention` instead

    logger.info(f"Model initialized.\n", transformer)

    # Build processor maps and a switch for spatial (3D) vs temporal (4D) global attention
    def _build_attn_processor_map(active_ids: List[int]) -> Dict[str, Any]:
        attn_processor_dict: Dict[str, Any] = {}
        for layer_id in range(transformer.config.num_layers):
            for attn_id in [1, 2]:
                key = f'blocks.{layer_id}.attn{attn_id}.processor'
                if layer_id in (active_ids or []):
                    attn_processor_dict[key] = COM4DAttnProcessor()
                else:
                    attn_processor_dict[key] = TripoSGAttnProcessor2_0()
        return attn_processor_dict

    attn_map_spatial = _build_attn_processor_map(spatial_global_attn_block_ids)
    attn_map_temporal = _build_attn_processor_map(temporal_global_attn_block_ids)

    def _unwrap_transformer_for_attn() -> COM4DDiTModel:
        # Prefer accelerator.unwrap_model (handles DDP/FSDP); otherwise fall back to .module if present.
        try:
            return accelerator.unwrap_model(transformer)
        except Exception:
            return transformer.module if hasattr(transformer, "module") else transformer

    def switch_to_mode(mode: str):
        base_transformer = _unwrap_transformer_for_attn()
        if mode == "3d":
            # set_attn_processor mutates the dict (pops entries); pass a fresh copy
            base_transformer.set_attn_processor(attn_map_spatial.copy())
            base_transformer.global_attn_block_ids = list(spatial_global_attn_block_ids)
        else:
            base_transformer.set_attn_processor(attn_map_temporal.copy())
            base_transformer.global_attn_block_ids = list(temporal_global_attn_block_ids)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # Create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_transformer.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                # Save models with explicit subfolders
                for i, model in enumerate(models):
                    unwrapped = accelerator.unwrap_model(model)
                    if isinstance(unwrapped, COM4DDiTModel):
                        unwrapped.save_pretrained(os.path.join(output_dir, "transformer"))
                    else:
                        # Fallback: save under a generic name with index to avoid collisions
                        unwrapped.save_pretrained(os.path.join(output_dir, f"model_{i}"))

                    # Make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = MyEMAModel.from_pretrained(os.path.join(input_dir, "transformer_ema"), COM4DDiTModel)
                ema_transformer.load_state_dict(load_model.state_dict())
                ema_transformer.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # Pop models so that they are not loaded again
                model = models.pop()
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, COM4DDiTModel):
                    load_model = COM4DDiTModel.from_pretrained(input_dir, subfolder="transformer")
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                else:
                    # Unknown model; skip
                    pass

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if configs["train"]["grad_checkpoint"]:
        transformer.enable_gradient_checkpointing()

    # Initialize the optimizer and learning rate scheduler
    logger.info("Initializing the optimizer and learning rate scheduler...\n")
    name_lr_mult = configs["train"].get("name_lr_mult", None)
    lr_mult = configs["train"].get("lr_mult", 1.0)
    params, params_lr_mult, names_lr_mult = [], [], []
    for name, param in transformer.named_parameters():
        if name_lr_mult is not None:
            for k in name_lr_mult.split(","):
                if k in name:
                    params_lr_mult.append(param)
                    names_lr_mult.append(name)
            if name not in names_lr_mult:
                params.append(param)
        else:
            params.append(param)
    optimizer = get_optimizer(
        params=[
            {"params": params, "lr": configs["optimizer"]["lr"]},
            {"params": params_lr_mult, "lr": configs["optimizer"]["lr"] * lr_mult}
        ],
        **configs["optimizer"]
    )
    if name_lr_mult is not None:
        logger.info(f"Learning rate x [{lr_mult}] parameter names: {names_lr_mult}\n")

    # Derive total steps; prefer existing value in configs if provided
    if "total_steps" not in configs["lr_scheduler"] or int(configs["lr_scheduler"]["total_steps"]) <= 0:
        approx_len = max(1, max(len(train_loader_3d), len(train_loader_4d)))
        configs["lr_scheduler"]["total_steps"] = configs["train"]["epochs"] * math.ceil(
            approx_len // max(1, accelerator.num_processes) / max(1, args.gradient_accumulation_steps)
        )  # only account updated steps
    configs["lr_scheduler"]["total_steps"] *= accelerator.num_processes  # for lr scheduler setting
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] *= accelerator.num_processes  # for lr scheduler setting
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, **configs["lr_scheduler"])
    configs["lr_scheduler"]["total_steps"] //= accelerator.num_processes  # reset for multi-gpu
    if "num_warmup_steps" in configs["lr_scheduler"]:
        configs["lr_scheduler"]["num_warmup_steps"] //= accelerator.num_processes  # reset for multi-gpu

    # Prepare everything with `accelerator`.
    (
        transformer,
        optimizer,
        lr_scheduler,
        train_loader_3d,
        train_loader_4d,
        val_loader_3d,
        val_loader_4d,
        random_val_loader_3d,
        random_val_loader_4d,
    ) = accelerator.prepare(
        transformer,
        optimizer,
        lr_scheduler,
        train_loader_3d,
        train_loader_4d,
        val_loader_3d,
        val_loader_4d,
        random_val_loader_3d,
        random_val_loader_4d,
    )

    if args.use_ema:
        ema_transformer.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move `vae`, frozen image encoder to gpu and cast to `weight_dtype`
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder_dinov2.to(accelerator.device, dtype=weight_dtype)

    # Training configs after distribution and accumulation setup
    updated_steps_per_epoch = max(1, (configs["lr_scheduler"]["total_steps"] // max(1, int(configs["train"]["epochs"]))))
    total_updated_steps = configs["lr_scheduler"]["total_steps"]
    if args.max_train_steps is None:
        args.max_train_steps = total_updated_steps
    # In mixed setup, allow non-exact divisibility between total steps and epochs
    if accelerator.num_processes > 1 and accelerator.is_main_process:
        print()
    accelerator.wait_for_everyone()
    logger.info(f"Total batch size: [{total_batch_size}]")
    logger.info(f"Learning rate: [{configs['optimizer']['lr']}]")
    logger.info(f"Gradient Accumulation steps: [{args.gradient_accumulation_steps}]")
    logger.info(f"Total epochs: [{configs['train']['epochs']}]")
    logger.info(f"Total steps: [{total_updated_steps}]")
    logger.info(f"Steps for updating per epoch: [{updated_steps_per_epoch}]")
    logger.info(f"Steps for validation: 3D[{len(val_loader_3d)}], 4D[{len(val_loader_4d)}]\n")

    # (Optional) Load checkpoint
    global_update_step = 0
    if args.resume_from_iter is not None:
        if args.resume_from_iter < 0:
            args.resume_from_iter = int(sorted(os.listdir(ckpt_dir))[-1])
        logger.info(f"Load checkpoint from iteration [{args.resume_from_iter}]\n")
        # Load everything
        if version.parse(torch.__version__) >= version.parse("2.4.0"):
            torch.serialization.add_safe_globals([
                int, list, dict, 
                defaultdict,
                Any,
                DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode
            ]) # avoid deserialization error when loading optimizer state
        accelerator.load_state(os.path.join(ckpt_dir, f"{args.resume_from_iter:06d}"))  # torch < 2.4.0 here for `weights_only=False`
        global_update_step = int(args.resume_from_iter)

    # Save all experimental parameters and model architecture of this run to a file (args and configs)
    if accelerator.is_main_process:
        exp_params = save_experiment_params(args, configs, exp_dir)
        save_model_architecture(accelerator.unwrap_model(transformer), exp_dir)

    # WandB logger
    if accelerator.is_main_process:
        if args.offline_wandb:
            os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project=PROJECT_NAME, name=args.tag,
            config=exp_params, dir=exp_dir,
            reinit=True
        )
        # Wandb artifact for logging experiment information
        arti_exp_info = wandb.Artifact(args.tag, type="exp_info")
        arti_exp_info.add_file(os.path.join(exp_dir, "params.yaml"))
        arti_exp_info.add_file(os.path.join(exp_dir, "model.txt"))
        arti_exp_info.add_file(os.path.join(exp_dir, "log.txt"))  # only save the log before training
        wandb.log_artifact(arti_exp_info)

    def get_sigmas(timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(dtype=dtype, device=accelerator.device)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ===== Diffusion-Forcing (per-part noise + target-only loss) helpers =====
    df_enabled = configs["train"].get("df_enabled", False)
    df_context_mode = configs["train"].get("df_context_mode", "prefix_k")  # ["prefix_k", "bernoulli_p"]
    df_context_k = int(configs["train"].get("df_context_k", 1))              # used if prefix_k
    df_context_p = float(configs["train"].get("df_context_p", 0.5))          # used if bernoulli_p (prob a token is HISTORY)

    # choose the cleanest timestep (min sigma) for "history" tokens
    _sigmas_all = noise_scheduler.sigmas.to(device=accelerator.device)
    _timesteps_all = noise_scheduler.timesteps.to(device=accelerator.device)
    _clean_idx = torch.argmin(_sigmas_all)
    _clean_timestep = _timesteps_all[_clean_idx]

    def build_df_context_mask(num_parts_vec: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask over the flattened [N] tokens: True = HISTORY (context), False = TARGET.
        num_parts_vec: shape [M], number of parts per object.
        """
        npv = num_parts_vec.to(accelerator.device)
        starts = torch.cat([torch.zeros(1, device=npv.device, dtype=torch.long), torch.cumsum(npv[:-1], dim=0)])
        N = int(npv.sum().item())
        context_mask = torch.zeros(N, device=npv.device, dtype=torch.bool)
        for m in range(npv.shape[0]):
            s, e = int(starts[m].item()), int((starts[m] + npv[m]).item())
            if df_context_mode == "prefix_k":
                k = min(df_context_k, e - s)
                if k > 0:
                    context_mask[s:s+k] = True
            elif df_context_mode == "bernoulli_p":
                context_mask[s:e] = (torch.rand(e - s, device=npv.device) < df_context_p)
            else:
                raise ValueError(f"Unknown df_context_mode: {df_context_mode}")
        # ensure at least one TARGET exists per object
        for m in range(npv.shape[0]):
            s, e = int(starts[m].item()), int((starts[m] + npv[m]).item())
            if context_mask[s:e].all():
                context_mask[e-1] = False
        return context_mask

    # Start training
    if accelerator.is_main_process:
        print("MAIN PROCESS: Start training...\n")

    logger.info(f"Start training into {exp_dir}\n")
    logger.logger.propagate = False  # not propagate to the root logger (console)
    progress_bar = tqdm(
        range(total_updated_steps),
        initial=global_update_step,
        desc="Training",
        ncols=175,
        disable=not accelerator.is_main_process
    )
    # Mixed training iterators
    train_iter_3d = yield_forever(train_loader_3d)
    train_iter_4d = yield_forever(train_loader_4d)
    objaverse_train_iter = yield_forever(objaverse_train_loader)
    # Probability to pick 4D at each iter (default 0.5)
    p_4d = float(configs["train"].get("prob_4d", 0.5))
    for _ in range(10**12):  # effectively infinite, controlled by max_train_steps

        if global_update_step == args.max_train_steps:
            progress_bar.close()
            logger.logger.propagate = True  # propagate to the root logger (console)
            if accelerator.is_main_process:
                wandb.finish()
            logger.info("Training finished!\n")
            return

        transformer.train()

        with accelerator.accumulate(transformer):
            # Randomly choose dataset for this iteration
            is_single_object_step = random.random() < single_object_reg_prob and single_object_configs != []
            if is_single_object_step:
                batch = next(objaverse_train_iter)
                use_4d = False
                mode = "single"
            else:
                use_4d = random.random() < p_4d
                batch = next(train_iter_4d) if use_4d else next(train_iter_3d)
                mode = "4d" if use_4d else "3d"
                # Switch attention mode
                switch_to_mode(mode)

            images_hw3 = batch["images"] # [N, H, W, 3]
            with torch.no_grad():
                pixel_values = feature_extractor_dinov2(images=images_hw3, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device=accelerator.device, dtype=weight_dtype) # [N, 3, Hf, Wf]
            # Original single-image tokens (frozen DINO)
            with torch.no_grad():
                single_tokens = image_encoder_dinov2(pixel_values).last_hidden_state  # [N, T, D]

            # Group indices by objects using num_parts
            num_parts = batch["num_parts"].to(accelerator.device) # [M]
            num_objects = num_parts.shape[0]
            # Use only per-part single-image DINO tokens
            image_embeds = single_tokens  # [N, Ts, D]

            negative_image_embeds = torch.zeros_like(image_embeds)

            if configs["train"]["cfg_dropout_prob"] > 0:
                # Drop entire conditions per part/frame so multi-frame objects do not lose all context at once
                dropout_mask = torch.rand(image_embeds.shape[0], device=accelerator.device) < configs["train"]["cfg_dropout_prob"]  # [N]
                if dropout_mask.any():
                    image_embeds[dropout_mask] = negative_image_embeds[dropout_mask]

            part_surfaces = batch["part_surfaces"] # [N, P, 6]
            part_surfaces = part_surfaces.to(device=accelerator.device, dtype=weight_dtype)

            with torch.no_grad():
                latents = vae.encode(
                    part_surfaces, 
                    **configs["model"]["vae"]
                ).latent_dist.sample()

            noise = torch.randn_like(latents)
            # For weighting schemes where we sample timesteps non-uniformly
            # ---- DF-aware per-token timestep sampling ----
            if df_enabled:
                # total token count N across all objects/parts
                N_tokens = latents.shape[0]
                # independent per-token density sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=configs["train"]["weighting_scheme"],
                    batch_size=N_tokens,
                    logit_mean=configs["train"]["logit_mean"],
                    logit_std=configs["train"]["logit_std"],
                    mode_scale=configs["train"]["mode_scale"],
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)  # [N]

                # Build history/target split per object and push HISTORY tokens to the clean timestep
                context_mask = build_df_context_mask(num_parts)                        # [N] bool, True = history
                timesteps = timesteps.clone()
                timesteps[context_mask] = _clean_timestep
            else:
                # original: one timestep per object, then repeat for each part
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=configs["train"]["weighting_scheme"],
                    batch_size=num_objects,
                    logit_mean=configs["train"]["logit_mean"],
                    logit_std=configs["train"]["logit_std"],
                    mode_scale=configs["train"]["mode_scale"],
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device) # [M, ]
                # Repeat the timesteps for each part
                timesteps = timesteps.repeat_interleave(num_parts) # [N, ]
                context_mask = None

            sigmas = get_sigmas(timesteps, len(latents.shape), weight_dtype)
            latent_model_input = noisy_latents = (1. - sigmas) * latents + sigmas * noise

            # Note: CFG dropout is applied via gating above, so no in-place
            # replacement here. This preserves the autograd path even when
            # the effective contribution is zero.

            # print(f"latent_model_input.shape = {latent_model_input.shape}, "
            #       f"timesteps.shape = {timesteps.shape}, "
            #       f"sigmas.shape = {sigmas.shape}, "
            #       f"timesteps = {timesteps[:10]}, "
            #       f"num_parts = {num_parts}, "
            #       f"surfaces.shape = {part_surfaces.shape}, "
            #       f"images.shape = {images_hw3.shape}, "
            #       f"image_embeds.shape = {image_embeds.shape}")

            # Compose attention kwargs for 3D/4D
            ones = torch.ones_like(num_parts).to(device=accelerator.device)
            
            attn_kwargs = {"num_parts": num_parts, "num_frames": ones}
            if use_4d:
                attn_kwargs = {"num_frames": num_parts, "num_parts": ones}
            elif is_single_object_step:
                attn_kwargs = {"num_parts": ones, "num_frames": ones}

            model_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                encoder_hidden_states=image_embeds,
                attention_kwargs=attn_kwargs,
            ).sample

            if configs["train"]["training_objective"] == "x0":  # Section 5 of https://arxiv.org/abs/2206.00364
                model_pred = model_pred * (-sigmas) + noisy_latents  # predicted x_0
                target = latents
            elif configs["train"]["training_objective"] == 'v':  # flow matching
                target = noise - latents
            elif configs["train"]["training_objective"] == '-v':  # reverse flow matching
                # The training objective for TripoSG is the reverse of the flow matching objective. 
                # It uses "different directions", i.e., the negative velocity. 
                # This is probably a mistake in engineering, not very harmful. 
                # In TripoSG's rectified flow scheduler, prev_sample = sample + (sigma - sigma_next) * model_output
                # See TripoSG's scheduler https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/schedulers/scheduling_rectified_flow.py#L296
                # While in diffusers's flow matching scheduler, prev_sample = sample + (sigma_next - sigma) * model_output
                # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L454
                target = latents - noise
            else:
                raise ValueError(f"Unknown training objective [{configs['train']['training_objective']}]")

            # For these weighting schemes use a uniform timestep sampling, so post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                configs["train"]["weighting_scheme"],
                sigmas
            )

            diff_loss = weighting * tF.mse_loss(model_pred.float(), target.float(), reduction="none")
            diff_loss = diff_loss.mean(dim=list(range(1, len(diff_loss.shape))))  # [N]

            if df_enabled and context_mask is not None:
                # target-only reduction for diffusion loss
                target_mask = (~context_mask).to(diff_loss.dtype)
                denom = target_mask.sum().clamp_min(1.0)
                loss = (diff_loss * target_mask).sum() / denom
            else:
                loss = diff_loss.mean()

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            loss = accelerator.gather(loss.detach()).mean()

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                f"{mode}_loss": loss.item(),
            }
            # Log DF stats if enabled
            if df_enabled and ("context_mask" in locals()) and (context_mask is not None):
                with torch.no_grad():
                    tgt_ratio = ((~context_mask).float().mean()).item()
                logs.update({"df/target_ratio": tgt_ratio})
            if args.use_ema:
                ema_transformer.step(transformer.parameters())
                logs.update({"ema": ema_transformer.cur_decay_value})

            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            global_update_step += 1

            msg = (
                f"[{global_update_step:06d} / {total_updated_steps:06d}] "
                f"loss: {logs['loss']:.4f}, lr: {logs['lr']:.2e}, mode: {mode}"
            )
            if args.use_ema and 'ema' in logs:
                msg += f", ema: {logs['ema']:.4f}"
            logger.info(msg)

            # Log the training progress
            if (
                global_update_step % configs["train"]["log_freq"] == 0 
                or global_update_step == 1
                or global_update_step % updated_steps_per_epoch == 0 # last step of an epoch
            ):  
                if accelerator.is_main_process:
                    to_log = {
                        "training/loss": logs["loss"],
                        "training/lr": logs["lr"],
                        f"training_{mode}/loss": logs[f"{mode}_loss"],
                    }
                    wandb.log(to_log, step=global_update_step)
                    if args.use_ema:
                        wandb.log({
                            "training/ema": logs["ema"]
                        }, step=global_update_step)

            # Save checkpoint
            if (
                global_update_step % configs["train"]["save_freq"] == 0  # 1. every `save_freq` steps
                or global_update_step % (configs["train"]["save_freq_epoch"] * updated_steps_per_epoch) == 0  # 2. every `save_freq_epoch` epochs
                or global_update_step == total_updated_steps # 3. last step of an epoch
                # or global_update_step == 1 # 4. first step
            ): 

                gc.collect()
                if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
                    # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                elif accelerator.is_main_process:
                    accelerator.save_state(os.path.join(ckpt_dir, f"{global_update_step:06d}"))
                accelerator.wait_for_everyone()  # ensure all processes have finished saving
                gc.collect()

            # Evaluate on the validation set
            if args.max_val_steps > 0 and (
                (global_update_step % configs["train"]["early_eval_freq"] == 0 and global_update_step < configs["train"]["early_eval"])  # 1. more frequently at the beginning
                or global_update_step % configs["train"]["eval_freq"] == 0  # 2. every `eval_freq` steps
                or global_update_step % (configs["train"]["eval_freq_epoch"] * updated_steps_per_epoch) == 0  # 3. every `eval_freq_epoch` epochs
                or global_update_step == total_updated_steps # 4. last step of an epoch
                or (global_update_step == 1 and not args.offline_wandb) # 5. first step
            ):  

                # Use EMA parameters for evaluation
                if args.use_ema:
                    # Store the Transformer parameters temporarily and load the EMA parameters to perform inference
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())

                transformer.eval()

                log_validation(
                    val_loader_3d, random_val_loader_3d,
                    val_loader_4d, random_val_loader_4d,
                    feature_extractor_dinov2, image_encoder_dinov2,
                    vae, transformer,
                    global_update_step, eval_dir,
                    accelerator, logger,
                    args, configs
                )

                if args.use_ema:
                    # Switch back to the original Transformer parameters
                    ema_transformer.restore(transformer.parameters())

                torch.cuda.empty_cache()
                gc.collect()

@torch.no_grad()
def log_validation(
    dataloader_3d, random_dataloader_3d,
    dataloader_4d, random_dataloader_4d,
    feature_extractor_dinov2, image_encoder_dinov2,
    vae, transformer, 
    global_step, eval_dir,
    accelerator, logger,  
    args, configs
):  

    val_noise_scheduler = RectifiedFlowScheduler.from_pretrained(
        configs["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )

    # Build both pipelines sharing the same underlying transformer
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    pipeline_3d = PartCrafterPipeline(
        vae=vae,
        transformer=unwrapped_transformer,
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )
    pipeline_4d = FourDCrafterPipeline(
        vae=vae,
        transformer=unwrapped_transformer,
        scheduler=val_noise_scheduler,
        feature_extractor_dinov2=feature_extractor_dinov2,
        image_encoder_dinov2=image_encoder_dinov2,
    )

    pipeline_3d.set_progress_bar_config(disable=True)
    pipeline_4d.set_progress_bar_config(disable=True)
    # pipeline.enable_xformers_memory_efficient_attention()

    if args.seed >= 0:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    else:
        generator = None
        

    val_progress_bar = tqdm(
        range((len(dataloader_3d) + len(dataloader_4d)) // 2) if args.max_val_steps is None else range(args.max_val_steps),
        desc=f"Validation [{global_step:06d}]",
        ncols=125,
        disable=not accelerator.is_main_process
    )

    medias_dictlist, metrics_dictlist = defaultdict(list), defaultdict(list)

    val_dataloader_3d, random_val_dataloader_3d = yield_forever(dataloader_3d), yield_forever(random_dataloader_3d)
    val_dataloader_4d, random_val_dataloader_4d = yield_forever(dataloader_4d), yield_forever(random_dataloader_4d)
    val_step = 0
    while val_step < args.max_val_steps:
        # randomly select between 3D and 4D validation batches
        use_4d = (torch.rand(1).item() < 0.5)
        if val_step < args.max_val_steps // 2:
            # deterministic half
            batch = next(val_dataloader_4d if use_4d else val_dataloader_3d)
        else:
            batch = next(random_val_dataloader_4d if use_4d else random_val_dataloader_3d)

        images = batch["images"]
        if len(images.shape) == 5:
            images = images[0] # (1, N, H, W, 3) -> (N, H, W, 3)
        images = [Image.fromarray(image) for image in images.cpu().numpy()]
        part_surfaces = batch["part_surfaces"].cpu().numpy()
        if len(part_surfaces.shape) == 4:
            part_surfaces = part_surfaces[0] # (1, N, P, 6) -> (N, P, 6)

        N = len(images)

        val_progress_bar.set_postfix(
            {"num_parts": N}
        )

        # Build attention maps for switching inside validation
        spatial_ids = configs["model"]["transformer"].get("spatial_global_attn_block_ids", configs["model"]["transformer"].get("global_attn_block_ids", []))
        spatial_ids = list(spatial_ids) if spatial_ids is not None else []
        temporal_ids = configs["model"]["transformer"].get("temporal_global_attn_block_ids", configs["model"]["transformer"].get("global_attn_block_ids", []))
        temporal_ids = list(temporal_ids) if temporal_ids is not None else []
        def _build_attn_processor_map_local(model, active_ids):
            d = {}
            for layer_id in range(model.config.num_layers):
                for attn_id in [1, 2]:
                    key = f'blocks.{layer_id}.attn{attn_id}.processor'
                    d[key] = COM4DAttnProcessor() if layer_id in (active_ids or []) else TripoSGAttnProcessor2_0()
            return d

        with torch.autocast("cuda", torch.float16):
            for guidance_scale in sorted(args.val_guidance_scales):
                # Switch attention processors and ids for this mode
                if use_4d:
                    unwrapped_transformer.set_attn_processor(_build_attn_processor_map_local(unwrapped_transformer, temporal_ids))
                    unwrapped_transformer.global_attn_block_ids = list(temporal_ids)
                    pred_part_meshes = pipeline_4d(
                        images, 
                        num_inference_steps=configs['val']['num_inference_steps'],
                        num_tokens=configs['model']['vae']['num_tokens'],
                        guidance_scale=guidance_scale, 
                        attention_kwargs={"num_frames": N, "num_parts": 1},
                        generator=generator,
                        max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                        use_flash_decoder=configs['val']['use_flash_decoder'],
                    ).meshes
                else:
                    unwrapped_transformer.set_attn_processor(_build_attn_processor_map_local(unwrapped_transformer, spatial_ids))
                    unwrapped_transformer.global_attn_block_ids = list(spatial_ids)
                    pred_part_meshes = pipeline_3d(
                        images, 
                        num_inference_steps=configs['val']['num_inference_steps'],
                        num_tokens=configs['model']['vae']['num_tokens'],
                        guidance_scale=guidance_scale, 
                        attention_kwargs={"num_parts": N, "num_frames": 1},
                        generator=generator,
                        max_num_expanded_coords=configs['val']['max_num_expanded_coords'],
                        use_flash_decoder=configs['val']['use_flash_decoder'],
                    ).meshes

                # Save the generated meshes
                if accelerator.is_main_process:
                    mode_str = "4d" if use_4d else "3d"
                    local_eval_dir = os.path.join(eval_dir, f"{global_step:06d}", mode_str, f"guidance_scale_{guidance_scale:.1f}")
                    os.makedirs(local_eval_dir, exist_ok=True)
                    rendered_images_list, rendered_normals_list = [], []
                    # 1. save the gt image
                    images[0].save(os.path.join(local_eval_dir, f"{val_step:04d}.png"))
                    # 2. save the generated part meshes
                    for n in range(N):
                        if pred_part_meshes[n] is None:
                            # If the generated mesh is None (decoing error), use a dummy mesh
                            pred_part_meshes[n] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                        pred_part_meshes[n].export(os.path.join(local_eval_dir, f"{val_step:04d}_{n:02d}.glb"))
                    # 3. render the generated mesh and save the rendered images
                    pred_mesh = get_colored_mesh_composition(pred_part_meshes)
                    rendered_images: List[Image.Image] = render_views_around_mesh(
                        pred_mesh, 
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    rendered_normals: List[Image.Image] = render_normal_views_around_mesh(
                        pred_mesh,
                        num_views=configs['val']['rendering']['num_views'],
                        radius=configs['val']['rendering']['radius'],
                    )
                    export_renderings(
                        rendered_images,
                        os.path.join(local_eval_dir, f"{val_step:04d}.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    export_renderings(
                        rendered_normals,
                        os.path.join(local_eval_dir, f"{val_step:04d}_normals.gif"),
                        fps=configs['val']['rendering']['fps']
                    )
                    rendered_images_list.append(rendered_images)
                    rendered_normals_list.append(rendered_normals)
                    # Build a paired GIF cycling each condition image with the rotating render
                    paired_frames = []
                    for cond_img in images:
                        for frame in rendered_images:
                            w = cond_img.width + frame.width
                            h = max(cond_img.height, frame.height)
                            canvas = Image.new("RGB", (w, h), (255, 255, 255))
                            canvas.paste(cond_img, (0, 0))
                            canvas.paste(frame, (cond_img.width, 0))
                            paired_frames.append(canvas)
                    export_renderings(
                        paired_frames,
                        os.path.join(local_eval_dir, f"{val_step:04d}_paired.gif"),
                        fps=configs['val']['rendering']['fps']
                    )

                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/gt_image"] += [images[0]] # List[Image.Image] TODO: support batch size > 1
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_images"] += rendered_images_list # List[List[Image.Image]]
                    medias_dictlist[f"guidance_scale_{guidance_scale:.1f}/pred_rendered_normals"] += rendered_normals_list # List[List[Image.Image]]

                ################################ Compute generation metrics ################################

                parts_chamfer_distances, parts_f_scores = [], []

                for n in range(N):
                    parts_chamfer_distances.append(0.0)
                    parts_f_scores.append(0.0)

                parts_chamfer_distances = torch.tensor(parts_chamfer_distances, device=accelerator.device)
                parts_f_scores = torch.tensor(parts_f_scores, device=accelerator.device)

                metrics_dictlist[f"parts_chamfer_distance_cfg{guidance_scale:.1f}"].append(parts_chamfer_distances.mean())
                metrics_dictlist[f"parts_f_score_cfg{guidance_scale:.1f}"].append(parts_f_scores.mean())
            
        # Only log the last (biggest) cfg metrics in the progress bar
        val_logs = {
            "parts_chamfer_distance": parts_chamfer_distances.mean().item(),
            "parts_f_score": parts_f_scores.mean().item(),
        }
        val_progress_bar.set_postfix(**val_logs)
        logger.info(
            f"Validation [{val_step:02d}/{args.max_val_steps:02d}] " +
            f"parts_chamfer_distance: {val_logs['parts_chamfer_distance']:.4f}, parts_f_score: {val_logs['parts_f_score']:.4f}"
        )
        logger.info(
            f"parts_chamfer_distances: {[f'{x:.4f}' for x in parts_chamfer_distances.tolist()]}"
        )
        logger.info(
            f"parts_f_scores: {[f'{x:.4f}' for x in parts_f_scores.tolist()]}"
        )
        val_step += 1
        val_progress_bar.update(1)

    val_progress_bar.close()

    if accelerator.is_main_process:
        for key, value in medias_dictlist.items():
            # Ensure nested directory exists for keys like "3d/guidance_scale_7.0/gt_image"
            nested_dir = os.path.join(eval_dir, f"{global_step:06d}", os.path.dirname(key))
            os.makedirs(nested_dir, exist_ok=True)
            if isinstance(value[0], Image.Image): # assuming gt_image
                image_grid = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='pil', 
                )
                image_grid.save(os.path.join(eval_dir, f"{global_step:06d}", f"{key}.png"))
                wandb.log({f"validation/{key}": wandb.Image(image_grid)}, step=global_step)
            else: # assuming pred_rendered_images or pred_rendered_normals
                image_grids = make_grid_for_images_or_videos(
                    value, 
                    nrow=configs['val']['nrow'],
                    return_type='ndarray',
                )
                wandb.log({
                    f"validation/{key}": wandb.Video(
                        image_grids, 
                        fps=configs['val']['rendering']['fps'], 
                        format="gif"
                )}, step=global_step)
                image_grids = [Image.fromarray(image_grid.transpose(1, 2, 0)) for image_grid in image_grids]
                export_renderings(
                    image_grids, 
                    os.path.join(eval_dir, f"{global_step:06d}", f"{key}.gif"), 
                    fps=configs['val']['rendering']['fps']
                )

        for k, v in metrics_dictlist.items():
            wandb.log({f"validation/{k}": torch.tensor(v).mean().item()}, step=global_step)

if __name__ == "__main__":
    main()

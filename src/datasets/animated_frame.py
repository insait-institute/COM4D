from src.utils.typing_utils import *

import json
import os
import random

import accelerate
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.data_utils import load_surface, load_surfaces

def _extract_single_surface_array(surface_data: dict) -> torch.Tensor:
    """Return a single [P,6] surface tensor from a loaded npy dict.

    Supports three shapes:
    - Top-level dict with keys 'surface_points' & 'surface_normals'
    - Dict with key 'object' containing the above
    - Dict with key 'parts' (expects length 1); takes the first part
    """
    if surface_data is None:
        raise ValueError("surface_data is None")
    if 'surface_points' in surface_data and 'surface_normals' in surface_data:
        return load_surface(surface_data)
    if 'object' in surface_data:
        return load_surface(surface_data['object'])
    if 'parts' in surface_data:
        parts = surface_data['parts']
        if isinstance(parts, list) and len(parts) > 0:
            return load_surface(parts[0])
    raise KeyError("Unrecognized surface data format: expected 'surface_points'+'surface_normals', or 'object', or non-empty 'parts'.")


class ObjaversePartDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs: DictConfig, 
        training: bool = True, 
    ):
        super().__init__()
        self.configs = configs
        self.training = training

        self.min_num_parts = configs['dataset']['min_num_parts']
        self.max_num_parts = configs['dataset']['max_num_parts']
        self.val_min_num_parts = configs['val']['min_num_parts']
        self.val_max_num_parts = configs['val']['max_num_parts']

        self.max_iou_mean = configs['dataset'].get('max_iou_mean', None)
        self.max_iou_max = configs['dataset'].get('max_iou_max', None)

        self.shuffle_parts = configs['dataset']['shuffle_parts']
        self.training_ratio = configs['dataset']['training_ratio']
        self.balance_object_and_parts = configs['dataset'].get('balance_object_and_parts', False)

        self.rotating_ratio = configs['dataset'].get('rotating_ratio', 0.0)
        self.rotating_degree = configs['dataset'].get('rotating_degree', 10.0)
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-self.rotating_degree, self.rotating_degree), fill=(255, 255, 255)),
        ])

        # Load dataset configs. Support two formats:
        # 1) Old list format: a list of dicts with keys including 'num_parts', 'valid', etc.
        # 2) New frame format: a dict mapping object_key -> list of frame dicts ({surface_path, image_path, iou_*}).
        def _load_one(path):
            with open(path, 'r') as f:
                return json.load(f)

        raw_entries = []
        if isinstance(configs['dataset']['config'], ListConfig):
            for config_path in configs['dataset']['config']:
                raw_entries.append(_load_one(config_path))
        else:
            raw_entries.append(_load_one(configs['dataset']['config']))

        # Detect format by inspecting the first top-level element
        data_configs: list[dict] = []
        new_format_objects: list[dict] = []
        any_new_format = False
        for entry in raw_entries:
            if isinstance(entry, list):
                # Old flat list format
                data_configs += entry
            elif isinstance(entry, dict):
                # New format: object_key -> list[frame]
                any_new_format = True
                for object_key, frames in entry.items():
                    if not isinstance(frames, list) or len(frames) == 0:
                        continue
                    # Normalize frames: only keep those with both surface_path & image_path
                    norm_frames = []
                    for fr in frames:
                        sp = fr.get('surface_path', None)
                        ip = fr.get('image_path', None)
                        if sp is None or ip is None:
                            continue
                        norm_frames.append({
                            'surface_path': sp,
                            'image_path': ip,
                        })
                    if len(norm_frames) == 0:
                        continue
                    new_format_objects.append({
                        'object_key': object_key,
                        'frames': norm_frames,
                        'num_frames': len(norm_frames),
                        'valid': True,
                    })
            else:
                raise ValueError("Unsupported dataset JSON root type; expected list or dict")

        if any_new_format:
            # We keep dataset min/max parts from the config. Do not override with global frame stats.
            if len(new_format_objects) == 0:
                raise ValueError("No valid objects found in the new frame-format dataset JSON")

            # Split train/val by ratio if not balancing objects/parts
            objects = new_format_objects
            if not self.balance_object_and_parts:
                split = int(len(objects) * self.training_ratio)
                if self.training:
                    objects = objects[:split]
                else:
                    objects = objects[split:]

            # Assign a fixed num_parts to each object (sampled within [min_parts, feasible_max])
            data_configs = []
            for obj in objects:
                n_frames = obj['num_frames']
                # Limit upper bound by object's frames with stride-2 feasibility (0,2,4,...)
                feasible_max = (n_frames + 1) // 2  # maximum K with step=2 contiguous selection
                upper = min(self.max_num_parts, feasible_max)
                # If object has fewer frames than dataset min, lower falls back to what's available
                lower = min(max(1, self.min_num_parts), upper)
                if lower <= 0:
                    continue
                num_parts = random.randint(lower, upper)
                data_configs.append({
                    'object_key': obj['object_key'],
                    'frames': obj['frames'],
                    'num_frames': n_frames,
                    'num_parts': num_parts,
                    'valid': True,
                    # Placeholders for compatibility
                    'iou_mean': 0.0,
                    'iou_max': 0.0,
                })

        # Filter and finalize data_configs
        if len(data_configs) > 0 and 'surface_path' in data_configs[0] or (len(data_configs) > 0 and 'surface_paths' in data_configs[0]):
            # Old format path: apply old filters
            data_configs = [config for config in data_configs if config.get('valid', True)]
            data_configs = [config for config in data_configs if self.min_num_parts <= config['num_parts'] <= self.max_num_parts]
            if self.max_iou_mean is not None and self.max_iou_max is not None:
                data_configs = [config for config in data_configs if config.get('iou_mean', 0.0) <= self.max_iou_mean]
                data_configs = [config for config in data_configs if config.get('iou_max', 0.0) <= self.max_iou_max]
            if not self.balance_object_and_parts:
                if self.training:
                    data_configs = data_configs[:int(len(data_configs) * self.training_ratio)]
                else:
                    data_configs = data_configs[int(len(data_configs) * self.training_ratio):]
                    data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]
        else:
            # New format: we already assigned num_parts and split by ratio above; optional extra filter by val min/max when not training
            if not self.training:
                data_configs = [config for config in data_configs if self.val_min_num_parts <= config['num_parts'] <= self.val_max_num_parts]

        self.data_configs = data_configs
        self.image_size = (512, 512)

    def __len__(self) -> int:
        return len(self.data_configs)
    
    def _get_data_by_config(self, data_config):
        # Support three cases:
        # 1) Old format with a single 'surface_path' + one 'image_path'
        # 2) Old format with 'surface_paths' list + one 'image_path' (replicate image)
        # 3) New format with 'frames' list and fixed 'num_parts' -> sample that many frames
        if 'frames' in data_config:
            # New frame-based format
            frames = data_config['frames']
            k = int(data_config['num_parts'])
            # Select k frames in even-step consecutive order: s, s+2, s+4, ... (no reordering)
            F = len(frames)
            max_start = (F - 1) - 2 * (k - 1)
            max_start = max_start if max_start >= 0 else 0
            # Enforce even starts (0,2,4,...) to get 0-2-4 style
            valid_starts = list(range(0, max_start + 1, 2))
            if len(valid_starts) == 0:
                # Fallback: allow any start, still step by 2 to preserve no-jump guarantee
                valid_starts = list(range(0, max_start + 1))
            s = random.choice(valid_starts)
            chosen = [frames[s + 2 * i] for i in range(k)]
            # Load surfaces per chosen frame
            part_surfaces = []
            images_list = []
            for fr in chosen:
                surface_path = fr['surface_path']
                image_path = fr['image_path']
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(_extract_single_surface_array(surface_data))
                # Load image per frame
                pil_image = Image.open(image_path)
                if getattr(pil_image, "is_animated", False):
                    try:
                        pil_image.seek(0)
                    except Exception:
                        pass
                pil_image.load()
                if pil_image.mode in ("RGBA", "LA"):
                    bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                    pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
                else:
                    pil_image = pil_image.convert("RGB")
                if random.random() < self.rotating_ratio:
                    pil_image = self.transform(pil_image)
                pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
                img = np.asarray(pil_image, dtype=np.uint8)
                images_list.append(torch.from_numpy(img).to(torch.uint8))
            part_surfaces = torch.stack(part_surfaces, dim=0)  # [N, P, 6]
            images = torch.stack(images_list, dim=0)  # [N, H, W, 3]
            return {
                "images": images,
                "part_surfaces": part_surfaces,
            }
        elif 'surface_path' in data_config:
            surface_path = data_config['surface_path']
            surface_data = np.load(surface_path, allow_pickle=True).item()
            # If parts is empty, the object is the only part
            part_surfaces = surface_data['parts'] if len(surface_data['parts']) > 0 else [surface_data['object']]
            if self.shuffle_parts:
                random.shuffle(part_surfaces)
            part_surfaces = load_surfaces(part_surfaces) # [N, P, 6]
            image_path = data_config['image_path']
            # Robustly load WebP/alpha images and resize
            pil_image = Image.open(image_path)
            # Apply optional rotation on the PIL image first (needs RGB mode)
            if getattr(pil_image, "is_animated", False):
                try:
                    pil_image.seek(0)
                except Exception:
                    pass
            pil_image.load()
            if pil_image.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")
            if random.random() < self.rotating_ratio:
                pil_image = self.transform(pil_image)
            pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
            image = np.asarray(pil_image, dtype=np.uint8)
            image = torch.from_numpy(image).to(torch.uint8)  # [H, W, 3]
            images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
            return {
                "images": images,
                "part_surfaces": part_surfaces,
            }
        else:
            part_surfaces = []
            for surface_path in data_config['surface_paths']:
                surface_data = np.load(surface_path, allow_pickle=True).item()
                part_surfaces.append(_extract_single_surface_array(surface_data))
            part_surfaces = torch.stack(part_surfaces, dim=0) # [N, P, 6]
            image_path = data_config['image_path']
            # Robustly load WebP/alpha images and resize
            pil_image = Image.open(image_path)
            # Apply optional rotation on the PIL image first (needs RGB mode)
            if getattr(pil_image, "is_animated", False):
                try:
                    pil_image.seek(0)
                except Exception:
                    pass
            pil_image.load()
            if pil_image.mode in ("RGBA", "LA"):
                bg = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
                pil_image = Image.alpha_composite(bg, pil_image.convert("RGBA")).convert("RGB")
            else:
                pil_image = pil_image.convert("RGB")
            if random.random() < self.rotating_ratio:
                pil_image = self.transform(pil_image)
            pil_image = pil_image.resize(self.image_size, Image.Resampling.BILINEAR)
            image = np.asarray(pil_image, dtype=np.uint8)
            image = torch.from_numpy(image).to(torch.uint8)  # [H, W, 3]
            images = torch.stack([image] * part_surfaces.shape[0], dim=0) # [N, H, W, 3]
            return {
                "images": images,
                "part_surfaces": part_surfaces,
            }
    
    def __getitem__(self, idx: int):
        # The dataset can only support batchsize == 1 training. 
        # Because the number of parts is not fixed.
        # Please see BatchedObjaversePartDataset for batched training.
        data_config = self.data_configs[idx]
        data = self._get_data_by_config(data_config)
        return data
        
class BatchedObjaversePartDataset(ObjaversePartDataset):
    def __init__(
        self,
        configs: DictConfig,
        batch_size: int,
        is_main_process: bool = False,
        shuffle: bool = True,
        training: bool = True,
    ):
        assert training
        assert batch_size > 1
        super().__init__(configs, training)
        self.batch_size = batch_size
        self.is_main_process = is_main_process
        # Exclude any object whose sampled parts equal or exceed the batch size.
        # This guarantees we never form a single-object batch with batch_size parts.
        self.data_configs = [config for config in self.data_configs if config['num_parts'] < batch_size]
        
        if shuffle:
            random.shuffle(self.data_configs)

        self.object_configs = [config for config in self.data_configs if config['num_parts'] == 1]
        self.parts_configs = [config for config in self.data_configs if config['num_parts'] > 1]
        
        self.object_ratio = configs['dataset']['object_ratio']
        # Here we keep the ratio of object to parts
        self.object_configs = self.object_configs[:int(len(self.parts_configs) * self.object_ratio)]

        dropped_data_configs = self.parts_configs + self.object_configs
        if shuffle:
            random.shuffle(dropped_data_configs)

        self.data_configs = self._get_batched_configs(dropped_data_configs, batch_size)
    
    def _get_batched_configs(self, data_configs, batch_size):
        batched_data_configs = []
        num_data_configs = len(data_configs)
        progress_bar = tqdm(
            range(len(data_configs)),
            desc="Batching Dataset",
            ncols=125,
            disable=not self.is_main_process,
        )
        while len(data_configs) > 0:
            temp_batch = []
            temp_num_parts = 0
            unchosen_configs = []
            while temp_num_parts < batch_size and len(data_configs) > 0:
                config = data_configs.pop() # pop the last config
                num_parts = config['num_parts']
                if temp_num_parts + num_parts <= batch_size:
                    temp_batch.append(config)
                    temp_num_parts += num_parts
                    progress_bar.update(1)
                else:
                    unchosen_configs.append(config) # add back to the end
            data_configs = data_configs + unchosen_configs # concat the unchosen configs
            if temp_num_parts == batch_size:
                # Successfully get a batch
                if len(temp_batch) < batch_size:
                    # pad the batch
                    temp_batch += [{}] * (batch_size - len(temp_batch))
                batched_data_configs += temp_batch
                # Else, the code enters here because len(data_configs) == 0
                # which means in the left data_configs, there are no enough 
                # "suitable" configs to form a batch. 
                # Thus, drop the uncompleted batch.
        progress_bar.close()
        return batched_data_configs
        
    def __getitem__(self, idx: int):
        data_config = self.data_configs[idx]
        if len(data_config) == 0:
            # placeholder
            return {}
        data = self._get_data_by_config(data_config)
        return data
    
    def collate_fn(self, batch):
        batch = [data for data in batch if len(data) > 0]
        images = torch.cat([data['images'] for data in batch], dim=0) # [N, H, W, 3]
        surfaces = torch.cat([data['part_surfaces'] for data in batch], dim=0) # [N, P, 6]
        num_parts = torch.LongTensor([data['part_surfaces'].shape[0] for data in batch])
        assert images.shape[0] == surfaces.shape[0] == num_parts.sum() == self.batch_size, \
            f"Batch size mismatch: {images.shape[0]} != {surfaces.shape[0]} != {num_parts.sum()} != {self.batch_size}"
        
        batch = {
            "images": images,
            "part_surfaces": surfaces,
            "num_parts": num_parts,
        }
        return batch

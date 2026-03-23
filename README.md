# Inferring Compositional 4D Scenes without Ever Seeing One

[Paper](https://arxiv.org/abs/2512.05272) | [Project Website](https://com4d.insait.ai) | [BibTeX](#bibtex)

## Authors
[Ahmet Berke Gökmen](https://berkegokmen1.github.io/), [Ajad Chattkuli](https://ajadchhatkuli.github.io/), [Luc Van Gool](https://insait.ai/prof-luc-van-gool/), [Danda Pani Paudel](https://insait.ai/dr-danda-paudel/)


## Abstract
> Scenes in the real world are often composed of several static and dynamic objects. Capturing their 4-dimensional structures, composition and spatio-temporal configuration in-the-wild, though extremely interesting, is equally hard. Therefore, existing works often focus on one object at a time, while relying on some category-specific parametric shape model for dynamic objects. This can lead to inconsistent scene configurations, in addition to being limited to the modeled object categories. We propose COM4D (Compositional 4D), a method that consistently and jointly predicts the structure and spatio-temporal configuration of 4D/3D objects using only static multi-object or dynamic single object supervision. We achieve this by a carefully designed training of spatial and temporal attentions on 2D video input. The training is disentangled into learning from object compositions on the one hand, and single object dynamics throughout the video on the other, thus completely avoiding reliance on 4D compositional training data. At inference time, our proposed attention mixing mechanism combines these independently learned attentions, without requiring any 4D composition examples. By alternating between spatial and temporal reasoning, COM4D reconstructs complete and persistent 4D scenes with multiple interacting objects directly from monocular videos. Furthermore, COM4D provides state-of-the-art results in existing separate problems of 4D object and composed 3D reconstruction despite being purely data-driven.

## TODO
- [x] Release Paper
- [x] Release Website
- [x] Release Training Code
- [x] Release Dataset Preprocessing Code
- [x] Release Inference Code
- [x] Release Video Processing Code
- [ ] Release Checkpoints (Soon)

## Training
### Environment setup
We provide `sh/env.sh` for reproducing our environment. It creates a `com4d`
micromamba environment, installs CUDA 12.4 compatible PyTorch, additional geometry
libraries, and pulls PyTorch3D from source. If you prefer `conda`, simply replace the
`micromamba` commands in this script with their `conda` equivalents—the remainder of
the steps (pip installs, PyTorch3D build) stay the same. The launch script below
expects that sourcing `micromamba activate com4d` (or your equivalent env) works.

### Dataset preprocessing
Fill in the dataset config paths only after preparing the data described in
[Dataset Preprocessing](#dataset-preprocessing). Every YAML under `configs/` contains
placeholders such as `'#PATH'` that must point to the processed Objaverse, 3D-FRONT,
and DeformingThings4D datasets produced by that pipeline, so complete those steps first.

### Config schedule
Training progresses through the configs in `configs/`:
1. `sdemb/mf8_mp8_nt512`
2. `sdemb/mf16_mp16/mf8_mp8_nt512`
3. `sdemb/mf16_mp16/dfot/mf8_mp8_nt512`
4. `sdemb/mf16_mp16/dfot/mask/mf8_mp8_nt512`

Each config has the same structure:
- `model`: pretrained checkpoints (VAE, transformer, scheduler) and embedding toggles.
- `dataset`, `dataset_3d`, `dataset_4d`, `dataset_objaverse`: data sources and filters.
- `optimizer` / `lr_scheduler`: standard AdamW setup with warmup.
- `train` / `val`: EMA knobs, logging cadence, evaluation settings, rendering fidelity.

Move to the next file only after the previous stage has converged; reuse the same
`--tag`/`--output_dir` hierarchy when you want to keep results grouped by stage.

### Download pretrained TripoSG
Run the following command:
```bash
huggingface-cli download VAST-AI/TripoSG --local-dir pretrained_weights/TripoSG
```

### Launch / continue training
The entry point is `sh/train.sh`. Key fields to customize:
- `OUT_DIR`: directory where checkpoints, logs, and renders are stored.
- `PRETRAINED_MODEL_PATH` + `_CKPT`: uncomment to resume from an earlier phase.
- `NUM_MACHINES`, `NUM_LOCAL_GPUS`, `CUDA_VISIBLE_DEVICES`: adapt to your cluster.

Simply run after getting everything ready:
```bash
bash sh/train.sh 0 # and so on...
```

## Dataset Preprocessing
Please refer to [datasets/README_preprocess](./datasets/README_preprocess.md) folder.

## Synthetic Dataset Generation for Evaluation
Please refer to [datasets/README_synthetic](./datasets/README_synthetic.md) folder.

## Model Weights
> Links will be updated soon.

## Inference
First clone [SAM2](https://github.com/facebookresearch/sam2) to the current directory.

Then run:
```sh
mv sam2/sam2 sam2_temp
rm -rf sam2
mv sam2_temp sam2
```

You can process any video using after updating the segmentation parameters for dynamic objects:
First download SAM2.1 Checkpoints by following the guide in [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) repository. The script will automatically load `IDEA-Research/grounding-dino-tiny` weights from HuggingFace. For video background removal, you may use [REMBG](https://github.com/danielgatis/rembg) or any other video background removal tool. Alternatively you can pass in a parameter for static objects in the scene which will skip video matting and remove the background using SAM2 instead. Please look at this [issue](https://github.com/facebookresearch/sam2/issues/337) if you encounter issues with SAM2.

Note: You can ideally not remove the background by providing `""` as the prompt for static objects. However if you choose to remove the background and isolate the objects of interest, you'll need to pass `"load_frames_no_bg": 1` as an argument in `infer.json`.

```sh
bash sh/process_video.sh <video_path> <start_frame> <end_frame> <segmentation_for_notdynamic_objects> <segmentation_for_static_objects (used for background removal)>

# Example, notice the "."s
bash scripts/process_video.sh assets/demo.mp4 0 100 "person. dog." "sofa. chair. lamp."
```
This script will produce the following structure which you can use for inference:
```
assets/demo/
├── frames
├───── 000000.png
├───── 000001.png
├── frames_no_bg
├───── 000000.png
├───── 000001.png
├── masks
├───── 000000_object_001.png
├───── 000000_object_002.png
├───── 000001_object_001.png
└───── 000001_object_002.png
```

For inference update the input paths and model path in the script. Then simply run:
```sh
bash sh/infer.sh <name_in_infer_json>

# e.g.
bash sh/infer.sh teaser2048
```
It should take around 20G of VRAM.

## Note
All experiments were conducted on NVIDIA H200 GPU(s).

## Acknowledgements
Most of the training code was borrowed from [PartCrafter](https://github.com/wgsxm/PartCrafter). We thank the authors for their great work and for sharing the codes.
Also we would like to thank the authors of [TripoSG](https://github.com/VAST-AI-Research/TripoSG) and [MIDI3D](https://github.com/VAST-AI-Research/MIDI-3D) for their incredible work. 
Special thanks to the authors of [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) and [REMBG](https://github.com/danielgatis/rembg).

<hr>

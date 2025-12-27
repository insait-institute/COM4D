# Inferring Compositional 4D Scenes without Ever Seeing One

[Paper](https://arxiv.org/abs/2512.05272) | [Project Website](https://com4d.insait.ai) | [BibTeX](#bibtex)

## Authors
[Ahmet Berke Gökmen](https://berkegokmen1.github.io/), [Ajad Chattkuli](https://ajadchhatkuli.github.io/), [Luc Van Gool](https://insait.ai/prof-luc-van-gool/), [Danda Pani Paudel](https://insait.ai/dr-danda-paudel/)


## Abstract
> Scenes in the real world are often composed of several static and dynamic objects. Capturing their 4-dimensional structures, composition and spatio-temporal configuration in-the-wild, though extremely interesting, is equally hard. Therefore, existing works often focus on one object at a time, while relying on some category-specific parametric shape model for dynamic objects. This can lead to inconsistent scene configurations, in addition to being limited to the modeled object categories. We propose COM4D (Compositional 4D), a method that consistently and jointly predicts the structure and spatio-temporal configuration of 4D/3D objects using only static multi-object or dynamic single object supervision. We achieve this by a carefully designed training of spatial and temporal attentions on 2D video input. The training is disentangled into learning from object compositions on the one hand, and single object dynamics throughout the video on the other, thus completely avoiding reliance on 4D compositional training data. At inference time, our proposed attention mixing mechanism combines these independently learned attentions, without requiring any 4D composition examples. By alternating between spatial and temporal reasoning, COM4D reconstructs complete and persistent 4D scenes with multiple interacting objects directly from monocular videos. Furthermore, COM4D provides state-of-the-art results in existing separate problems of 4D object and composed 3D reconstruction despite being purely data-driven.

## Code will be released!

## TODO
- [x] Release Paper
- [x] Release Website
- [x] Release Training Code
- [x] Release Dataset Preprocessing Code
- [ ] Release Inference Code
- [ ] Release Checkpoints

## Training
### Environment setup
We provide `scripts/env.sh` for reproducing our environment. It creates a `com4d`
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
Training progresses through the numbered configs in `configs/`:
1. `1_mf8_mp8_nt512.yaml`
2. `2_mf16_mp16_nt512.yaml`
3. `3_mf16_mp16_nt512_dfot.yaml`

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
The entry point is `scripts/train_com4d.sh`. Key fields to customize:
- `OUT_DIR`: directory where checkpoints, logs, and renders are stored.
- `PRETRAINED_MODEL_PATH` + `_CKPT`: uncomment to resume from an earlier phase.
- `CONFIG_NAME`: matches one of the YAML files described above.
- `NUM_MACHINES`, `NUM_LOCAL_GPUS`, `CUDA_VISIBLE_DEVICES`: adapt to your cluster.

Simply run after getting everything ready:
```bash
bash scripts/train_com4d.sh
```


## Dataset Preprocessing
Please refer to [datasets](./datasets/README.md) folder.

## Inference
In progress...

## Acknowledgements
Most of the training code was borrowed from [PartCrafter](https://github.com/wgsxm/PartCrafter). We thank the authors for their great work and for sharing the codes.
Also we would like to thank the authors of [TripoSG](https://github.com/VAST-AI-Research/TripoSG) and [MIDI3D](https://github.com/VAST-AI-Research/MIDI-3D) for their incredible work.

<hr>

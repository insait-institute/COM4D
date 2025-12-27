# Dataset Preprocessing

## 3D-FRONT

### Rendering
This dataset comes with pre-rendered GLBs. Test set comes from [MIDI-3D](https://huggingface.co/datasets/huanngzh/3D-Front).

### Point Sampling
```sh
python datasets/preprocess/mesh_to_point.py \
    --input <path-to-3d-front-scene-root> \
    --output <path-to-3d-front-scene-preprocessed-root> \
    --normalize 1 \
    --workers 1
```

### JSON
```sh
python datasets/preprocess/3dfront_json.py \
    --input <path-to-3d-front-scene-preprocessed-root> \
    --render-root <path-to-3d-front-render-root> \
    --glb-root <path-to-3d-front-scene-root> \
    --output ./dataset_json/3dfront.json
```

## DeformingThings4D
`/animals` has only `.anime` files whereas `/humanoids` has both `.anime` and `.fbx` files. We'll process them accordingly.

### Per-frame Mesh Extraction

#### Humanoids
You'll need [BlenderPY](https://docs.blender.org/api/current/info_quickstart.html) installed on your system for this step.

Run the following command to make sure everything works:
```sh
blender -b --python datasets/preprocess/fbx_to_glb.py -- \
    --fbx DeformingThings4D/humanoids/AJ_BigStomachHit/AJ_BigStomachHit.fbx \
    --output-dir ./test_fbx \
    --end-frame 4
```

For processing all humanoids (do not forget to update the paths in .txt file):
```sh
mkdir -p <humanoids-out-dir>

bash datasets/preprocess/run_fbx_range.sh datasets/preprocess/fbx_jobs.txt \
    <humanoids-out-dir> \
    <blender-executable-path> \
    0 201 \
    --start-frame 1 --end-frame 64
```

#### Animals
Extract per-frame meshes using:

```sh
python datasets/preprocess/anime_to_glb.py \
  --input DeformingThings4D/animals \
  --output <animals-out-dir> \
  --max-frames 64 \
  --workers 1
```


### Re-Scale & Center & Orient
GLBs from `.anime` files have wrong orientation. Thus we re-orient them using the following script. We do this only for the animals.
```sh
python datasets/preprocess/reorient_glbs.py \
    --input <animals-out-dir> \
    --workers 8
```

Lastly, we scale the meshes and apply a global translation so that the center of the animation is at the center of the scene. We do this for both animals and humans.
```sh
python datasets/preprocess/center_and_scale_glb_sequence.py \
    <animals-out-dir> \
    <animals-scaled-out-dir> \
    --frame-limit 32 --num-samples 1000 --seed 42

python datasets/preprocess/center_and_scale_glb_sequence.py \
    <humanoids-out-dir> \
    <humanoids-scaled-out-dir> \
    --frame-limit 32 --num-samples 1000 --seed 42
```

### Point Sampling
```sh
python datasets/preprocess/mesh_to_point.py \
    --input <animals-scaled-out-dir> \
    --output <animals-scaled-preprocessed-out-dir> \
    --normalize 0 \
    --workers 1

python datasets/preprocess/mesh_to_point.py \
    --input <humanoids-scaled-out-dir> \
    --output <humanoids-scaled-preprocessed-out-dir> \
    --normalize 0 \
    --workers 1
```

### Rendering
```sh
python datasets/preprocess/render_fixed_cam.py \
    --input <animals-scaled-out-dir> \
    --output <animals-scaled-render-out-dir> \
    --workers 4 \
    --auto-exposure \
    --use-palette-color

python datasets/preprocess/render_fixed_cam.py \
    --input <humanoids-scaled-out-dir> \
    --output <humanoids-scaled-render-out-dir> \
    --workers 4 \
    --auto-exposure
```

### JSON
```sh
python datasets/preprocess/deformingthings_json.py \
        --pair <animals-scaled-preprocessed-out-dir>:<animals-scaled-render-out-dir> \
        --pair <humanoids-scaled-preprocessed-out-dir>:<humanoids-scaled-render-out-dir> \
        -o ./dataset_json/deformingthings.json --pretty
```

## Objaverse

### Scale
```sh
python datasets/preprocess/center_and_scale_glb_folder.py \
    --input <objaverse-dir> \
    --output <objaverse-scaled-out-dir> \
    --workers 4
```

### Point Sampling
```sh
python datasets/preprocess/mesh_to_point.py \
    --input <objaverse-scaled-out-dir> \
    --output <objaverse-scaled-preprocessed-out-dir> \
    --normalize 0 \
    --workers 1
```

### Rendering
```sh
python datasets/preprocess/render_fixed_cam.py \
    --input <objaverse-scaled-out-dir> \
    --output <objaverse-scaled-render-out-dir> \
    --workers 4 \
    --auto-exposure
```

### JSON
```sh
python datasets/preprocess/objaverse_json.py \
  --preprocessed-root <objaverse-scaled-preprocessed-out-dir> \
  --render-root <objaverse-scaled-render-out-dir> \
  --glb-root <objaverse-scaled-out-dir> \
  --output ./dataset_json/objaverse.json
```

## Final
If you have done all the steps correctly and without errors, you should have the following structure:
```
dataset_json/
├── 3dfront.json            # Static multi-object indoor scenes
├── deformingthings.json    # Single-object temporal deformations
└── objaverse.json          # Large-scale object-centric 3D assets
```

If you encounter any issues, feel free to open an issue.
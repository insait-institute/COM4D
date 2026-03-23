# Synthetic Dataset Generation

## Compositional 4D Scene Generation
```bash
python3 datasets/synthetic/gen.py \
  --models-path <3D-FUTURE-MODEL-PATH> \
  --dynamic-objects-path <HUMANOIDS-SCALED-PATH> \
  --output-path <GLB-OUTPUT-PATH> \
  --total-scenes 250 \
  --num-frames 8 \
  --frame-stride 2 \
  --min-dynamic-objects 1 \
  --max-dynamic-objects 2 \
  --min-static-objects 2 \
  --max-static-objects 4 \
  --num-workers 8 \
  --seed 42
```

## Rendering (Requires GPU)
You may use [Arb-Objaverse](https://huggingface.co/datasets/lizb6626/Arb-Objaverse) for HDRIs.

```bash
python3 datasets/synthetic/render.py \
  --input-root <GLB-OUTPUT-PATH> \
  --output-root <RENDER-OUTPUT-PATH> \
  --hdri-root <ENV-MAPS-PATH> \
  --environment-style outdoor-hdri \
  --environment-variation per-scene \
  --environment-seed 42 \
  --num-workers 8 \
  --shadows
```
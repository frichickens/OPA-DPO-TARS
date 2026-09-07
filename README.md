# Train TARS on OPA-DPO data, then evaluate on AMBER

## Exact pipeline

```text
OPA-DPO LLaVA-1.5-7B rollouts (subsets 1 and 2)
  chosen   = AI_pseudo_response       # GPT-4V-corrected response
  rejected = original_generate_response  # original LLaVA-1.5-7B response
                         ↓
Official TARS DPO code and published LLaVA-1.5-7B configuration
                         ↓
OPA-DPO AMBER_generate.py → OPA-DPO AMBER_eval.py
```

This experiment trains the **TARS method**, not OPA-DPO, on OPA-DPO's released
preference pairs. GPT-4V has already annotated the release; no API is required.

The setup pins the public TARS commit and applies one documented execution fix:
[`patches/tars_initialize_original_tokens.patch`](patches/tars_initialize_original_tokens.patch)
initializes token clones outside an optional branch. It does not alter the TARS
loss, data, or published hyperparameters.

## Requirements

- Linux, Conda, Git, `unzip`
- Exact published hardware default: 8 GPUs
- At least 50GB free disk

## Sources

- TARS code: https://github.com/KejiaZhang-Robust/TARS
- TARS pinned commit: `b2e62881bb39dd28a0d12b81d5bcbdf033f7fa9d`
- OPA-DPO data: https://drive.google.com/drive/folders/1Xmrb43zIbk3IzLLRQx65iBf7J4SHXa9j
- LLaVA-1.5-7B: https://huggingface.co/liuhaotian/llava-v1.5-7b
- CLIP ViT-L/14-336: https://huggingface.co/openai/clip-vit-large-patch14-336
- AMBER: https://github.com/junyangwang0410/AMBER
- AMBER images: https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view

## Setup environment

```bash
git clone https://github.com/frichickens/OPA-DPO-TARS.git
cd OPA-DPO-TARS
bash run/setup_tars_environment.sh
```

This creates Conda environment `TARS_DPO`, clones/pins official TARS, and
installs AMBER dependencies.

## One command: download → train → AMBER

Use an explicit large storage directory:

```bash
WORK_ROOT=/mnt/disk4/baodq/tars_on_opadpo \
GPUS=0,1,2,3,4,5,6,7 \
NPROC=8 \
GRAD_ACC=8 \
bash run/one_run_tars_on_opadpo_amber.sh
```

Expected data conversion: 616 rollout files → 4,928 raw pairs → approximately
4.9k usable pairs after OPA-DPO's released filters. Identical pairs are retained
because the official OPA-DPO builder retains them. Exact counts and SHA-256 are
saved in `dataset_manifest.json`.

## Outputs

```text
$WORK_ROOT/models/                         # LLaVA and CLIP
$WORK_ROOT/datasets/OPA-DPO-7B/            # rollouts, parquet, manifest
$WORK_ROOT/datasets/AMBER/                 # 1,004 AMBER images
$WORK_ROOT/checkpoints/tars_on_opadpo_7b/  # trained TARS model
$WORK_ROOT/results/AMBER/answers.jsonl
$WORK_ROOT/results/AMBER/metrics.log
```

Report AMBER CHAIR, Cover, HalRate, and Cog from `metrics.log`.

## Resume individual stages

```bash
# Download/convert OPA-DPO pairs
DATA_ROOT=/mnt/data/OPA-DPO-7B bash run/setup_opadpo_pairs.sh

# Download AMBER
AMBER_ROOT=/mnt/data/AMBER bash run/setup_amber.sh

# Train official TARS
MODEL_DIR=/mnt/models/llava-v1.5-7b \
VISION_TOWER_DIR=/mnt/models/vision_tower-clip336 \
DATA_PATH=/mnt/data/OPA-DPO-7B/opadpo_7b_pairs_tars.parquet \
OUTPUT_DIR=/mnt/checkpoints/tars_on_opadpo_7b \
GPUS=0,1,2,3,4,5,6,7 NPROC=8 GRAD_ACC=8 \
conda run -n TARS_DPO bash run/train_tars_on_opadpo.sh

# Evaluate with OPA-DPO's released AMBER pipeline
MODEL_PATH=/mnt/checkpoints/tars_on_opadpo_7b \
AMBER_ROOT=/mnt/data/AMBER \
EVAL_ROOT=/mnt/results/tars_on_opadpo_amber \
conda run -n TARS_DPO bash run/eval_amber_tars.sh
```

For fewer GPUs, preserve global batch 64: `NPROC × GRAD_ACC = 64`. This is a
hardware adaptation; the 8-GPU command above exactly matches TARS's release.

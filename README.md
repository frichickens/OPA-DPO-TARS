# OPA-DPO on TARS → AMBER

## Objective

Train OPA-DPO (LLaVA-1.5-7B) on 4,800 TARS/RLHF-V pairs, then evaluate on
AMBER with OPA-DPO's released greedy-generation and scoring code.

Important: TARS lacks OPA-DPO's GPT-4V token/image scores. This is therefore the
paper's **w/o hw&iw** approximation, not full token-weighted OPA-DPO. Mapping:

- TARS `chosen` → corrected/reference response
- TARS `rejected` → original response
- Missing token/image losses are disabled; duplicate-reference weight is set to zero
- All compatible published 7B settings remain unchanged

## Requirements

- Linux, Conda, Git, `unzip`
- Recommended: 4×A100/H100 80GB
- At least 50GB free disk
- No GPT API is needed for training or AMBER

## Setup

```bash
git clone https://github.com/frichickens/OPA-DPO-TARS.git
cd OPA-DPO-TARS
bash run/setup_code.sh
conda env create -f environment.yaml
conda activate OPA_DPO
pip install flash-attn==2.5.3 --no-build-isolation
```

## One run: download, train, evaluate

Choose an explicit large folder; do not rely on `$HOME`:

```bash
WORK_ROOT=/mnt/disk4/baodq/opa_dpo_tars_run \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
GPUS_PER_NODE=4 \
bash run/one_run_tars_to_amber.sh
```

It automatically downloads LLaVA-1.5-7B, CLIP, TARS/RLHF-V, and AMBER images;
builds the deterministic seed-42 4.8k corpus; trains OPA then OPA-DPO; and runs
AMBER.

Outputs:

```text
$WORK_ROOT/models/                    # LLaVA and CLIP
$WORK_ROOT/datasets/                  # original/converted TARS and AMBER
$WORK_ROOT/training/llava7b_opa/      # stage-1 checkpoint
$WORK_ROOT/training/llava7b_opadpo/   # final adapter
$WORK_ROOT/evaluation/AMBER/answers.jsonl
$WORK_ROOT/evaluation/AMBER/metrics.log
```

Report AMBER CHAIR, Cover, HalRate, and Cog from `metrics.log`. Preserve
`datasets/converted_tars/dataset_manifest.json` with the results.

## Custom storage paths

```bash
MODEL_ROOT=/models/opa_tars \
TARS_DOWNLOAD_DIR=/data/RLHF-V \
TRAIN_DATA_ROOT=/data/tars_converted \
AMBER_ROOT=/data/AMBER \
TRAIN_OUTPUT_ROOT=/checkpoints/opa_tars \
EVAL_ROOT=/results/opa_tars_amber \
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 \
bash run/one_run_tars_to_amber.sh
```

Individual recovery stages are `setup_base_model_7b.sh`,
`setup_tars_dataset.sh`, `setup_amber.sh`, `train_opa_dpo_on_tars.sh`, and
`eval_amber_tars_opadpo.sh` under `run/`.

Upstream: [OPA-DPO](https://github.com/zhyang2226/OPA-DPO),
[paper](https://arxiv.org/abs/2501.09695). More limitations are recorded in
[`TARS_CROSS_TRAINING.md`](TARS_CROSS_TRAINING.md).

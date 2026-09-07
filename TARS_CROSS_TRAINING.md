# OPA-DPO trained on the TARS/RLHF-V corpus

This fork provides a controlled cross-training experiment: use OPA-DPO's official
LLaVA-1.5-7B two-stage recipe on 4,800 TARS/RLHF-V preference examples, then run
OPA-DPO's official evaluation implementation.

## Reproduce on another server

Hardware matching the release: one node with 4x A100-80GB (or equivalent), plus
COCO and AMBER for evaluation.

```bash
git clone https://github.com/bestxrr/OPA-DPO-TARS.git
cd OPA-DPO-TARS
bash run/setup_code.sh
conda env create -f environment.yaml
conda activate OPA_DPO
pip install flash-attn==2.5.3
bash run/prepare_basemodels.sh
bash run/setup_tars_dataset.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 bash run/train_opa_dpo_on_tars.sh
```

## Exact OPA-DPO evaluation implementation

Provide the dataset paths/API credentials as environment variables and call the
wrapper, which delegates to the released `run/eval_all_metrics.sh`:

```bash
IMAGE_FOLDER_LB=/data/coco/train2017 \
IMAGE_FOLDER_POPE=/data/coco/val2014 \
IMAGE_DIR_AMBER=/data/AMBER/image \
ANNOTATION_FILE=/data/coco/annotations \
OPENAI_ENDPOINT="$OPENAI_ENDPOINT" OPENAI_API_KEY="$OPENAI_API_KEY" \
bash run/eval_tars_corpus_opadpo.sh
```

This runs the released OPA-DPO evaluator for MMHal-Bench, LLaVA-Bench, POPE-Adv,
AMBER, and Object HalBench. GPT-scored tasks require an API key.

## What is exact and what is adapted

All model, LoRA, optimization, CoPO/AncPO, epoch, batch, and evaluation defaults
remain those in the official OPA-DPO 7B scripts. Dataset mapping is necessarily
adapted because RLHF-V has one positive/negative pair, whereas OPA-DPO expects a
reference, on-policy answer, GPT-4V rewrite, and token-level GPT-4V report:

- TARS `chosen` -> `standard_response` and `AI_pseudo_response`
- TARS `rejected` -> `original_generate_response`
- `detailed_report`, `response_score`, and `response_image_relation` are disabled
- `standard_pair_coef=0`, `AI_pair_coef=1` avoids counting the duplicated positive twice

TARS does not release its exact sampled 4.8k indices. The setup uses a recorded,
deterministic seed-42 sample from the 5,733-row release. These limitations mean
the run is an exact OPA-DPO optimizer/evaluator configuration, not an exact
recreation of OPA-DPO's unavailable GPT-4V supervision.

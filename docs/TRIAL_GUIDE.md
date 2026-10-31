# 24M Trial | Public run guide

The Trial is the public engineering version of 130U-FinLLM. The code, default training configurations, and notebooks follow this path. See the [model card](MODEL_CARD.md) and [project overview](PROJECT_OVERVIEW.md) for the architecture and methods of the private 2.4B version.

## Check the complete pipeline

Python 3.10 or later is required.

```bash
python -m pip install -e .
python -m finllm smoke --device cpu --output runs/first-smoke
```

This command trains a small configuration on project-generated synthetic examples and checks tokenizer handling, pretraining, domain training, SFT, the preference-training interface, restoration, calibration, and prediction. Outputs are saved to the specified directory; use a new directory for another run. This is an engineering check. The separate 24M configuration is [`model_24m.json`](../configs/model_24m.json).

## The 24M training path

| Stage | Code and documentation |
|---|---|
| Data fields, eligibility, and partitions | [Data guide](DATA_GUIDE.md) · [Annotation template](../data/examples/annotation_template.jsonl) |
| Initial source text, tokenizer, and shards | [Seed data](SEED_DATA.md) · [Preparation notebook](../notebooks/01a_seed_data.ipynb) |
| Foundation training and expansion | [Model pilot](../notebooks/02a_seed_pilot.ipynb) · [Expanded training](../notebooks/02c_expanded_pretrain.ipynb) |
| Financial-domain adaptation | [Domain handoff](DOMAIN_HANDOFF.md) · [Domain notebook](../notebooks/03a_domain_pretrain.ipynb) |
| Target-stance SFT | [Entity task](ENTITY_SFT_PILOT.md) · [Balanced-exposure script](../scripts/kaggle_balanced_sft.py) |
| Calibration and prediction | [Inference-export script](../scripts/kaggle_inference_export.py) · [CLI](../src/finllm/cli.py) |

Real task runs require data that meets the schema and provenance requirements. The existing headline experiments use private material and a financial-domain checkpoint. The public repository supplies code and records; data and weights remain in their respective workflows.

## Commands and configuration

```bash
python -m finllm --help
python -m finllm train --help
python -m finllm predict --help
```

[`pretrain.json`](../configs/pretrain.json), [`domain.json`](../configs/domain.json), and [`sft.json`](../configs/sft.json) show stage configurations. Before running, follow the data guide to prepare a tokenizer, training and validation material, and output paths. Use `init_checkpoint` to initialize a new stage. Use `resume` to continue the same stage with optimization and data state restored.

## Results and engineering records

The [Trial experiment results](experiment-results.md) retain sample counts, per-class metrics, and run conditions. The [engineering retrospective](ENGINEERING.md) explains class learning, semantic boundaries, and scaling choices. [Trial status](STATUS.md) summarizes the public path. Machine-readable files in `reports/` correspond to their original experiments.

## Development checks

```bash
python -m pip install -e ".[test]"
python -m pytest
python scripts/validate_notebooks.py
```

Eligible model inputs produce a label, three-class probabilities, a direction score, and review information. A text's expressed stance and an asset's subsequent return are different quantities; callers should use outputs according to the task definition.

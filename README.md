# 130U-FinLLM

**Turning personal financial standards into model behavior.**

[Overview](docs/PROJECT_OVERVIEW.md) · [Architecture](docs/MODEL_CARD.md) · [Training system](docs/TRAINING_SYSTEM.md) · [Run the Trial](docs/TRIAL_GUIDE.md) · [Chinese summary](docs/INTRO.zh-CN.md)

<img src="assets/banner.png" alt="130U-FinLLM: a private 2.4B financial language model with a public 24M Trial" width="100%">

130U-FinLLM is a dedicated language-model project for personal financial research. It brings together my original financial commentary and human-reviewed public text, organizing supervision around **the target, opinion holder, evidence scope, and conditions of a judgment**.

The project has a **private 2.4B model** and a **public 24M Trial**. This repository presents the architecture, training methodology, and engineering behind the project, alongside runnable Trial code, configurations, notebooks, and experiment records.

## Two scales. One engineering path.

| | Private 2.4B model | Public 24M Trial |
|---|---|---|
| Role | Personal financial research | Open training practice and reproducible engineering |
| Parameters | 2,397,459,968 | 24,255,744 |
| Backbone | 28 layers · hidden 2,560 · Q20 / KV4 | 6 layers · hidden 512 · Q8 / KV4 |
| Vocabulary / context | 65,536 / 4,096 | 8,192 / 512 |
| Explore | [Architecture and methods](docs/MODEL_CARD.md) | [Configuration](configs/model_24m.json) · [Experiments](docs/experiment-results.md) |

The 24M Trial establishes the data, supervision, training, and evaluation workflow. The 2.4B version carries forward its task definitions and engineering methods, with its own vocabulary, weights, runtime, and calibration.

## Two core designs

<img src="assets/features.png" alt="Financial semantic permissions and objective-preserving scaling" width="100%">

**Financial semantic permissions.** Revenue growth and an author's bullish stance answer different questions. Target changes, quotations, negation, and conditions each specify which outputs should change and which should remain stable. Label preservation and distribution preservation are separate constraints. [Explore the supervision design →](docs/TRAINING_SYSTEM.md)

**Objective-preserving scaling.** Sample-family accounting, valid supervision positions, full-vocabulary loss, segmented attention, distributed reduction, and recovery semantics form one training contract. Scaling the execution keeps the learning objective explicit. [Explore the compute path →](docs/TRAINING_SYSTEM.md)

## From source material to structured judgment

1. **Define tasks and data boundaries.** Record sources, targets, and opinion holders; split sample families before generating training views.
2. **Learn the language foundation.** Train from random initialization on next-token prediction.
3. **Adapt to financial material.** Mix financial text, general-language retention, reading QA, and target-specific tasks.
4. **Train, calibrate, and deliver.** Apply target-conditioned SFT, then separate model selection, probability calibration, acceptance thresholds, and final evaluation.

[Training workflow](assets/training_workflow.svg) · [Core computation](assets/training_compute.svg) · [Project brief PDF](docs/130U-FinLLM.pdf)

## Run the public Trial

```bash
git clone https://github.com/130U/130U-FinLLM.git
cd 130U-FinLLM
python -m pip install -e .
python -m finllm smoke --device cpu --output runs/first-smoke
```

The smoke command uses a small synthetic configuration to exercise tokenization, training, recovery, calibration, and prediction. The full Trial configuration is in [`configs/model_24m.json`](configs/model_24m.json). Follow the [Trial guide](docs/TRIAL_GUIDE.md) to prepare your own data and run individual stages. Training data, checkpoints, and private-model weights are managed separately.

## Go deeper

| Topic | Read |
|---|---|
| Motivation, version evolution, and design choices | [Project overview](docs/PROJECT_OVERVIEW.md) |
| Parameters, layers, attention, and version mapping | [Model card](docs/MODEL_CARD.md) |
| Data, semantic supervision, global objectives, and compute | [Training system](docs/TRAINING_SYSTEM.md) |
| Class collapse, task semantics, attribution, and scaling | [Engineering retrospective](docs/ENGINEERING.md) |
| Public code, data preparation, and runnable entry points | [Trial guide](docs/TRIAL_GUIDE.md) |
| Observed results and per-class analysis | [Trial experiments](docs/experiment-results.md) |

## License and provenance

Public code is licensed under [Apache-2.0](LICENSE). The Trial backbone builds on MiniMind; attribution and modifications are documented in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md). Data and model-weight licenses are managed separately from the code license.

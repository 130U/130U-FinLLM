# Training system | From financial semantics to parameter updates

130U-FinLLM has two technical foundations. Financial semantic permissions define what the model should learn. The training objective and runtime contract define how that supervision is executed accurately. The 2.4B version serves personal research; the public 24M Trial exposes foundational training and engineering methods.

![Training workflow](../assets/training_workflow.svg)

## Data and four training stages

### 1. Define material, tasks, and partitions

My original views, public texts I have reviewed, general-language corpora, and financial source text retain separate provenance. Each task specifies a target, opinion holder, evidence scope, and information availability time. Source-text stance, metric direction, and my own judgment have distinct semantics, so one ambiguous label does not stand in for three questions.

Original text, target queries, QA, and edits about the same event belong to one family. Families are split before derivative training views are created, and related material inherits the same partition. The human-gold recipe allocates 32k records: 12k training, 4k development, 4k calibration, 4k threshold selection, and 8k final test. Source records, training views, repeated exposures, and supervised positions are counted separately.

### 2. Foundational language pretraining

The 2.4B foundation recipe starts from random initialization with next-token prediction and allocates 50B input tokens. A vocabulary of 65,536 encodes the source text. Tokenizer training and pretraining follow the same evaluation-exclusion boundaries.

### 3. Financial mixed adaptation

The 5B-input-token recipe comprises 4B financial source text, 0.75B general source text, 0.20B reading QA, and 0.05B stance/event inputs. Source text supervises valid next-token positions, QA supervises answer positions, and stance/event views supervise task labels. Input shares, supervision shares, and objective weights are managed separately. Recipe allocations are distinct from actual data inventories and consumed tokens.

### 4. SFT, calibration, and use

Target-conditioned SFT learns task standards. The development set selects configurations, an independent calibration set fits probability scaling, a threshold set determines acceptance and review rules, and final testing evaluates the frozen system. Model weights, tokenizer, input protocol, and calibration configuration form one release.

## Financial semantic permissions

“Revenue increased” and “the author is bullish” can agree or diverge. The project models investment stance and financial metric direction separately, with `target / holder / scope / as_of` included in task definitions.

Each edit pair has a separate relation for each task axis: preserve the label, relabel, or impose no relation constraint. Label preservation and distribution preservation require separate permissions. Strict paraphrases can justify nearby probability distributions, while new valid evidence may change confidence.

Correct labels at both endpoints provide cross-entropy anchors. Pairs licensed as distribution-equivalent receive consistency constraints; valid relabeled pairs receive relative-response constraints. For task j, when the label changes from a to b, define:

```text
D = [log q_j(b | edited) − log q_j(a | edited)]
  − [log q_j(b | original) − log q_j(a | original)]
L_change = max(0, margin − D)
```

Reviewed labels, edit-operator names, and permissions are supervision metadata. Model inputs contain the genuinely available text, task, and target conditions, so predictions must follow the textual relationships.

## Training objective and runtime

### Sample families and effective denominators

Reviewed views are aggregated within families before aggregation by task; natural examples retain their own distributional objective. Source text, QA, task labels, and valid pairs have explicitly defined statistical units. Generating more edits from one source should not unintentionally multiply the weight of its entire family.

For objective term j in one optimizer update, let S_rj be the numerator on rank r and C_j the global effective denominator. With a reducer that averages gradients across W ranks:

```text
local_loss_r = W × Σ[j: C_j > 0] λ_j × S_rj / C_j
```

Denominators cover the entire gradient-accumulation window and all ranks. Globally absent terms are omitted; ranks with no local examples still follow the same collective communication order. Sampling frequency, repeated exposure, and task weights are recorded separately.

### Preserve the full-vocabulary objective

For sparse-label tasks, select the hidden states used to predict the supervised answers before applying the shared vocabulary projection:

```text
H        = backbone(input)                  # [B, S, 2560]
H_valid  = gather(H, supervision_positions) # [M, 2560]
logits   = H_valid @ W_vocab.T              # [M, 65536]
loss     = CE_full_vocabulary(logits, gold)
```

The entire context still participates in backbone computation. The omitted work is the output projection at positions without direct supervision. Projecting only three labels would change the full-vocabulary CE normalization, so it is treated as a different training objective.

### Preserve information and state semantics

Packing uses segmented causal attention: an example can attend only to earlier positions within its own segment. Loss masks, position IDs, and attention boundaries serve different roles. Different views of the same family also remain in independent segments.

The shared embedding and output head remain the same parameter object. Distributed execution, precision changes, and restoration paths are validated in stages: effective logits, individual loss terms, gradients, one parameter update, and the next update after restoration. Memory and throughput are measured after these checks.

![Core training computation](../assets/training_compute.svg)

## Public implementation entry points

[Trial model](../src/finllm/model.py) · [Data processing](../src/finllm/data.py) · [Trainer](../src/finllm/training.py) · [Evaluation](../src/finllm/evaluation.py) · [Trial guide](TRIAL_GUIDE.md)

This page describes the project-level 2.4B methods and training system. The executable entry points above belong to the public 24M Trial, as do the public configurations and run records.

## Method foundations

[ARTS](https://aclanthology.org/2020.emnlp-main.292/) and [CheckList](https://aclanthology.org/2020.acl-main.442/) provide relevant foundations for target changes and behavioral testing. [Cut Your Losses](https://arxiv.org/abs/2411.09009) addresses supervised positions and output computation. The project's combined design connects these methods to financial task permissions, global objective accounting, and runtime validation.

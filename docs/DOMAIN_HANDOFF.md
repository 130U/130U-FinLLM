# 24M Trial | 03a Financial-Domain Training and Weight Transfer

Scope: public engineering documentation for the 24M Trial, recording the completed domain stage and its recovery contract. See the [Trial experiment report](experiment-results.md) for subsequent task training and the [project overview](../README.md) for the broader project.

2026-10-30. Entry point: [03a_domain_pretrain.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/03a_domain_pretrain.ipynb).

**Stage acceptance: this segment is complete.** The user subsequently supplied complete Colab output: 74 domain-stage steps, 1,211,960 effective tokens, financial validation loss of 3.2375, general retention-subset loss changing from 3.9842 to 4.0574, and final backup output with `verified: true`. See the [domain-stage record](../reports/colab_t4_domain_20261030.json). The entry-point design and recovery instructions are preserved below. Literal-entity SFT was subsequently completed; the existing domain-stage artifacts do not need to be regenerated. The [review handoff record](POST_DOMAIN_NEXT.md) preserves the material handoff process.

Completed 02d output is preserved in the [user's historical commit](https://github.com/130U/130U-FinLLM/blob/e14b85e5660c4bce175341ee4d9da2e3984a12b5/notebooks/02d_expanded_continue.ipynb); see the [1220-step report](../reports/colab_t4_expanded_continuation_20261030.json) for the read-back record. The expansion stage processed 19,988,195 effective tokens, or 24,903,136 along the path including seed training. This is not the size of a unique corpus. On its first run, 03a reads back Drive files in Colab and verifies the parent checkpoint; stored GitHub output does not replace this check.

## Running the Stage

1. Open 03a with a T4, run from the first cell in order, and mount Drive. There is no need to repeat 01b, 02c, or 02d.
2. The preparation cell restores frozen data and the 8K tokenizer, verifies the full snapshot manifest, the 1220-step parent checkpoint, its matching measured report, and the original seed retention-set after-report, then freezes the parent checkpoint. The `parent_audit` output should show the hashes, steps, and tokens actually read.
3. Measure the before loss on the original seed general retention subset. On the first transition, transfer only model weights and create a new financial-stage AdamW optimizer and learning-rate schedule. Stage steps start at zero.
4. Run the first-step check, then continue to 74 cumulative updates in this financial stage. Save and sync checkpoints every 10 steps. Resuming within the new stage restores the optimizer, scaler, random state, and data position.
5. Measure the after loss on the same original seed retention subset and run the final verified-backup cell. Retain the financial validation loss, segment report, `parent_audit`, and final `verified: true` output.

The user posted T4, Drive-mount, parent-checkpoint SHA, training, and final-backup output, which constitute the evidence for this cloud execution. The agent did not independently download the Drive checkpoint. The initial preparation cell performs the actual cloud verification; if any check fails, preserve its output and existing artifacts.

## Segment Budget

| Item | Value |
|---|---:|
| Existing 80/20 financial/general mixture shards | 1,215,545 tokens |
| Financial-source positions within the mixture | 972,436 |
| General-source positions within the mixture | 243,109 |
| Financial validation shards | 50,999 tokens |
| Batch / gradient accumulation / sequence length | 4 / 8 / 512 |
| Single-step check / cumulative stopping target | 1 / 74 |
| Nominal input positions for 74 full-length steps | 1,212,416 |
| Learning rate / warmup / schedule span | 1e-4 / 5 steps / 74 steps |

This is a small, single-pass segment over existing data without replay. Actual effective tokens, padding, elapsed time, and throughput come from the new report. Pure arithmetic using the preceding 3221 tokens/second gives approximately six minutes of training, excluding cloud verification, measurement, and backup. The mixture rearranges source content and cannot be counted again as independent new data.

The original 100M general / 25M financial plan is a longer cumulative processing budget and remains incomplete. This segment was explicitly reduced to approximately 1.21M positions to accept the financial stage first. Extending it or adding data requires a separate budget decision. Processing 25M tokens from the existing mixture would require approximately 20.57 passes; replay must be reported explicitly and cannot be called 25M unique financial tokens. 03a does not automatically extend training this way.

The single-pass limit is enforced during preparation and before every microbatch. Even if skipped AMP updates consume data, training does not silently enter a second pass to reach 74 successful updates. When the remaining data is insufficient, it stops and retains the last complete checkpoint and original artifacts.

## Files and Recovery

The Drive project root is `MyDrive/130U-FinLLM`:

| Artifact | Relative path |
|---|---|
| Original 1220-step expansion artifacts | `expansions/expansion_v1/run` |
| Frozen parent checkpoint and source reports for this financial stage | `domains/domain_v1/parent` |
| Before/after retention measurements for this stage | `domains/domain_v1/retention` |
| Measured reports from this stage | `domains/domain_v1/report_archive` |
| Final complete snapshot for this stage | `domains/domain_v1/run` |
| Automatic checkpoints for this stage | `live_checkpoints/domain_v1` |

The local active training directory is `/content/130U-FinLLM/runs/domain_v1`. The original seed artifacts, 610-step archive, and 1220-step model are preserved. Because the first cross-stage transition transfers only weights, the parent's original PyTorch version and the new stage's current version may differ; both identities are recorded. Subsequent exact recovery within the same `domain_v1` binds the data, tokenizer, model, training semantics, and new runtime identity.

After a disconnect, first confirm the previous training state, then restore from the first 03a cell in order. If 74 steps are already complete, reuse the matching original measured report instead of replacing it with zero-second throughput. Failed training or measurement must not appear as a complete backup. Preserve errors when environment or input identities differ; do not bypass verification.

## Materials for Task Post-Training

The review material was expanded to 2000 records. The user supplied a CSV and explicitly confirmed 1993 headline labels, with seven remaining unresolved. See the [review record](../reports/human_headline_review_20261030.json). The subsequent literal-entity Trial used 1008 partitioned task records to complete SFT and performed probability calibration separately. This does not automatically complete security-identity verification, full event/time isolation, a new independent test, or DPO preference review. Language-modeling loss is not bullish/bearish asset accuracy.

Runnable notebooks in the repository have no stored output. Save execution output in a personal Drive copy when needed. The complete 02d evidence already saved for this run remains in the historical commit above and the local archive.

## Engineering Validation

All 274 local tests passed in 557.43 seconds, along with format, syntax, and no-output checks for 14 notebooks and dependency checks. Coverage includes execution of 03a in two real isolated Jupyter kernels, resetting the new domain optimizer, a numerical comparison of uninterrupted versus segmented training with dropout, failure and interruption recovery, and preservation of original measurements after completion. Source lineage was also checked against the actual local 8K shards. See the [engineering record](../reports/domain_handoff_checks_20261030.json). These local substitute tests are not verification of the user's actual Drive/T4 run.

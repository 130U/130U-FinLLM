# 24M Trial | Strict Continuation of Expanded General Pretraining in 02d

Scope: public engineering documentation for the 24M Trial, recording the expansion stage's training state and recovery semantics. See the [Trial experiment report](experiment-results.md) for subsequent results and the [project overview](../README.md) for the broader project.

2026-10-11. Entry point: [02d_expanded_continue.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02d_expanded_continue.ipynb).

2026-10-30 update: the user completed this entry point, reaching 1220 cumulative expansion steps. Execution output saved on GitHub was independently read back; see the [1220-step record](../reports/colab_t4_expanded_continuation_20261030.json). The subsequent [03a financial stage](DOMAIN_HANDOFF.md) verified and inherited this parent checkpoint. The original 02d execution and recovery instructions are preserved below.

User-supplied 02c logs showed 610 expansion steps, 9,994,240 supervised tokens, final training loss of 4.1933, and loss of 4.1311 on the new general validation subset. A later screenshot showed after-training loss of 4.4032 on the original seed retention subset and a final seven-file snapshot totaling 615,996,721 bytes with `verified: true`. See the [execution record](../reports/colab_t4_expansion_20261011.json) for the full evidence and its scope. The agent did not independently download the cloud checkpoint. The user did not provide the before value for the original validation set, so no improvement on that set is calculated.

## Execution Order

1. Open 02d, select a T4, and run in order. The notebook mounts Drive by default; complete the normal authorization flow when first prompted.
2. After environment installation and the current-kernel import check, restore the verified `seed_v1` and `expansion_v1` inputs. Verify the completed 610-step model, reports, and original retention measurements, and archive the complete artifacts before training is allowed to begin.
3. Record the original seed retention loss before continuation, or reuse the same measurement if it has already been verified. First strictly resume to step 611 and confirm that saving and restoration work.
4. Continue with strict recovery to 1220 cumulative expansion steps. The optimizer, scheduler, AMP scaler, random state, data cursor, and stage counters are not reset.
5. Measure after-continuation loss at the same original validation positions, then run the final verified-backup cell. Retain the segment's new tokens, cumulative expansion tokens, path-cumulative tokens including the seed stage, and final `verified: true` output.

There is no need to regenerate data or repeat 02c. Once 02d has advanced training, use 02d for recovery. The 02c after-result continues to represent 610 steps and is not rewritten as a 1220-step result. Do not run the training cells of both notebooks concurrently.

## Stage Budget

| Item | Updates / nominal token positions |
|---|---:|
| Completed expansion stage | 610 / 9,994,240 |
| Single-step recovery check | 611 cumulative |
| Stopping target for this segment | 1220 cumulative expansion steps |
| New work in this segment, including the single-step check | 610 / 9,994,240 |
| Cumulative expansion total on completion | 19,988,480 |

These counters still belong to the same `expansion_v1` pretraining experiment; a separate stage is not created from zero. Effective tokens are counted in the report. The table's full-length arithmetic does not account for skipped AMP updates or padding. The seed's 4,914,941 supervised tokens are listed separately; the cumulative model path is not the quantity of unique data.

The original batch size 4, gradient accumulation 8, sequence length 512, learning-rate schedule span 6104, warmup 50, and 100M-token ceiling remain in force. Only the stopping step increases; warmup is not restarted. The existing 41,499,364 new general training tokens can support this segment without replay, but that does not mean the 100M total budget is complete. Based on the preceding segment's end-to-end throughput, this segment is estimated at approximately 45 minutes. Actual time comes from its new report.

## Storage and Recovery

All Drive paths below are relative to `MyDrive/130U-FinLLM/expansions/expansion_v1`:

| Artifact | Path |
|---|---|
| Complete 610-step artifacts and original before/after records | `continuations/updates_000610_to_001220/baseline_run` |
| Before/after measurements on the original validation set for this segment | `continuations/updates_000610_to_001220/retention/{before,after}` |
| Segment reports | `continuations/updates_000610_to_001220/report_archive` |
| Current complete run snapshot | `run` |
| Compatible reports from prior runs | `report_archive` |

Automatic checkpoints continue to write to `live_checkpoints/expansion_v1_pretrain` under the project root. The original `seeds/seed_v1`, expanded data, original tokenizer, and original `retention/{before,after}` remain unchanged. Preserving the complete 610-step artifacts adds approximately 600 MiB of persistent storage. The model and optimizer state must be retained; a text report alone is insufficient.

After disconnection, wait for reconnection and confirm that the old execution has ended or stopped before running 02d from the first cell. A new VM selects the latest complete local/Drive checkpoint for the same version. If 1220 steps are already complete, it performs strict verification and reuses original measurements, without rolling back, restarting timing, or writing zero-second throughput.

Recovery requires matching data, tokenizer, architecture, optimization semantics, device type, precision, and PyTorch/CUDA identity. A checkpoint from a different environment is not treated as an exact resume. Preserve the actual versions in error output instead of bypassing verification. Missing evidence for the preceding complete segment, checkpoint corruption, measurement failure, or backup failure causes an explicit stop while existing artifacts are preserved.

## Acceptance and Scope

02d succeeds when recovery checks pass, cumulative expansion progress reaches 1220 steps, original validation before/after records are verifiable, the segment report is archived, and the final snapshot reports `verified: true`. At initial delivery, all 232 local tests passed in 343.60 seconds, together with format and syntax checks for 13 notebooks and dependency checks. 02d code cells ran in two isolated Jupyter kernels. A small CPU model with dropout also verified exact agreement of model, optimizer, scheduler, RNG, and progress between uninterrupted and segmented continuation. See the [continuation checks](../reports/expanded_continuation_checks_20261011.json) for the engineering record. The user subsequently completed this segment; independent verification of the cloud weights remains the responsibility of the following entry point.

Loss on the new general validation set and the original seed retention subset is reported separately. Both measure language modeling, not bullish/bearish asset accuracy. Subsequent domain training, literal-entity SFT, and probability calibration have separate execution records. DPO and a new independent test are not included in this stage's results.

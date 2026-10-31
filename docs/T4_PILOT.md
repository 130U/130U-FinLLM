# 24M Trial | T4 Pilot and 300-Step Continuation Results

Scope: public engineering documentation for the 24M Trial. The figures on this page are measured records from the seed stage only. See the [Trial experiment report](experiment-results.md) for subsequent results and the [project overview](../README.md).

2026-10-11. User-supplied Colab screenshots showed that `02a_seed_pilot.ipynb` completed training on a T4 and then read `run_summary.json` from Drive. The figures below were manually transcribed from those screenshots. The original report and checkpoint were not downloaded, and the agent did not independently rerun training. See the machine-readable [pilot record](../reports/colab_t4_pilot_20261011.json).

| Metric | Result |
|---|---:|
| Parameters | 24,255,744 |
| Successful / attempted updates | 61 / 61 |
| Actual training tokens | 999,424 |
| Final training loss | 7.1595 |
| Validation loss | 7.2726 |
| Validation positions in this measurement | 102,400 |
| Training-loop time | 296.43 seconds |
| Throughput | 3,371.56 tokens/second |
| Peak GPU memory allocated by PyTorch | 1,136,339,456 bytes, approximately 1.058 GiB |

All 61 attempts completed parameter updates. The final backup output showed `verified: true` and `MyDrive/130U-FinLLM/seeds/seed_v1`. The earlier current-kernel import failure did not block this pilot.

Throughput includes validation and checkpoint-backup time within the training loop. GPU memory comes from `torch.cuda.max_memory_allocated` and does not include the entire CUDA environment or cache footprint. Validation uses the configured limited number of batches, rather than all validation text. `effective_max_tokens: 100000000` is a budget ceiling; actual training processed 999,424 tokens. Pretraining loss does not provide target-asset stance-classification accuracy.

## 300-Step Continuation and Final Backup

On 2026-10-11, the user provided `02b_seed_continue.ipynb` logs covering steps 63–300 consecutively, followed by final backup output. The continuation report was parsed from that log; backup figures came from the user's message. The agent did not independently access the Colab runtime or download the Drive checkpoint. See the machine-readable [continuation record](../reports/colab_t4_continuation_20261011.json).

| Metric | Continuation result |
|---|---:|
| Cumulative successful / attempted updates | 300 / 300 |
| Cumulative effective training tokens | 4,914,941 |
| Cumulative input positions | 4,915,200 |
| Segment update range / new updates | 62→300 / 238 |
| New effective tokens in this segment | 3,899,133 |
| Final training loss | 5.4151 |
| Bounded validation loss | 5.5419 |
| Validation positions in this measurement | 102,400 |
| Segment training-loop time | 1,150.17 seconds, approximately 19 minutes 10 seconds |
| Segment throughput | 3,390.06 tokens/second |
| Peak GPU memory allocated by PyTorch | 1,141,189,120 bytes, approximately 1.063 GiB |

The 238 update log entries were consecutive, all reported `amp_skipped: false`, and every loss was finite. Effective tokens were 259 below input positions because of padding, consistent with the supervised-token counting definition. Validation loss fell from 7.2726 at step 61 to 5.5419 at step 300. It measures text-prediction loss under the same configuration, not financial-stance accuracy. The log establishes completion of continuation from existing weights; it does not alone establish that the runtime was reclaimed or that recovery on a new VM succeeded.

The continuation report archive was shown as `MyDrive/130U-FinLLM/seeds/seed_v1/report_archive/updates_000062_to_000300`. Final run-snapshot backup output reported seven files, 615,936,768 bytes, `verified: true`, and `working_copy: false`, with the project version directory `MyDrive/130U-FinLLM/seeds/seed_v1`. The implementation uses `run` as the run-snapshot subdirectory. The original step-61 pilot report remains in `report_archive/pilot_updates_000061`. These are output records from the user's run; Drive file lists and hashes were not independently read back.

This continuation segment and its final backup are complete; no retraining is needed. The original 100M-token ceiling remains a configuration budget. The actual cumulative training volume is 4,914,941 tokens.

## Continuation Entry Point and Recovery

Open [02b: verify recovery and continue general pretraining](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02b_seed_continue.ipynb), keep T4 and `seed_v1`, and run from the first cell in order.

1. Restore the same data, tokenizer, and training artifacts from the existing Drive backup, preserving the original pilot report.
2. An existing checkpoint must be found and verified. First continue from step 61 to 62. If progress is already beyond that point, reuse it without rolling back.
3. Continue from verified progress to 300 cumulative steps, then back up. This run processed 4,914,941 actual effective training tokens. If a 300-step checkpoint already exists, reuse the completed artifacts.

Continuation preserves the original 8K tokenizer, data version, optimizer state, random state, data cursor, and learning-rate schedule. `warmup_updates=122`, `schedule_updates=6104`, and the original token ceiling remain unchanged; only the stopping step increases. The verified `data/processed/seed_v1/pilot_config.json` is not rewritten. Incompatible PyTorch/CUDA, data, or training semantics cause recovery to stop; checks cannot be skipped or replaced with random initialization.

The original plan from step 61 to 300 implied approximately 3.916 million new positions under the nominal batch budget and an estimated 19.4 minutes. The actual step 62→300 report recorded 3,899,133 new effective tokens and 19.17 minutes; the step 61→62 check forms a separate segment. Step and token counts in each segment report are cumulative, while elapsed time and throughput belong only to that invocation. They must not be summed in a way that double-counts training.

Before this entry point was delivered, all 128 local tests passed, including eight continuation tests, along with format and syntax checks for 10 notebooks. Tests executed actual notebook cells and two new Jupyter kernels using a tiny CPU configuration for 2→3→5-step training, report archiving, and recovery after interruption of the final snapshot. External Git/Drive services were replaced with local substitutes. Engineering checks are recorded separately from the user-supplied T4 results above. See the machine-readable [continuation engineering checks](../reports/seed_continuation_checks.json).

## Follow-Up Plan at the Time and Actual Handoff

This stage had approximately 4.98 million unique general training tokens, so 300 steps approached one pass. The planned next step was to expand authorized general and financial data, create a new data version, and retain the task-text exclusion pool. The [expansion stage](DATA_EXPANSION.md) subsequently implemented that handoff. Across data versions, the trained tokenizer and source checkpoint are retained, but only weights are transferred; exact `--resume` rejects changed data hashes. Cycling the same data to 100 million tokens would take approximately 20 passes, not produce 100 million new tokens of source data.

`03_domain_pretrain.ipynb` uses the original `tokenizer/english`, `runs/pretrain`, and financial-mixture shard paths, so it cannot directly continue from the seed paths. The actual subsequent financial stage was completed through [03a](DOMAIN_HANDOFF.md), followed by literal-entity SFT. DPO preference review and training are separate and are not included in this page's results.

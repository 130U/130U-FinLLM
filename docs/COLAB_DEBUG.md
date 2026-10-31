# 24M Trial | Fixing 01a Reruns and Dependency Conflicts

Scope: public engineering documentation for the 24M Trial, preserving the diagnosis, fix, and validation of this incident. See the [project overview](../README.md) for the broader project.

For the subsequent 02a `No module named 'finllm'` error, stale installation metadata warnings, and repeated training execution, see the [kernel installation fix](COLAB_KERNEL_FIX.md).

2026-10-09. The Colab PDF supplied by the user showed code version `1c265b3` raising `FileExistsError: Use a new version: data/raw/seed_v1/prepared` when the build cell was rerun. Downloads and Drive backups had already reported success. An existing directory does not by itself indicate corrupted artifacts: the old entry point also rejected reruns of a completed build.

A separate issue involved dependency constraints. This Colab environment included Transformers 5.18.0, which requires tokenizers >=0.23.1,<0.24, while the old entry point installed <0.23. The project now supports <0.24, respects the installed Transformers constraints during installation, and checks them afterward. Colab's PyTorch installation is retained. The Hugging Face login warning did not cause this interruption; the log already showed a successful download.

## Corrected Behavior

- Completed downloads, task exclusion pools, tokenizers, and shards are reused read-only after their manifests, sources, configurations, and hashes are verified. Artifacts produced by earlier versions remain compatible.
- Incomplete or corrupted artifacts, or changed parameters, leave the existing files intact and produce an explicit instruction to restore a complete backup or choose a new version. Files are not automatically deleted or overwritten.
- A new runtime can restore raw data, shards, and the tokenizer from `MyDrive/130U-FinLLM/seeds/seed_v1/`. Edits to the human-review CSV are not overwritten by a fresh export.
- 01a revalidates every training input before backup so that interrupted builds are not recorded as complete.
- 01a can run on CPU; 02a requires a GPU. The supplied log reported `torch=2.11.0+cpu` and `cuda_available=false`, so it is not a T4 execution record.

## Continuing the Run

1. Reopen the [latest 01a from GitHub](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb) and save a notebook copy. Installation cells in an old Drive copy do not update automatically when GitHub changes.
2. Keep `SEED_VERSION = "seed_v1"` and run the cells from top to bottom. A new runtime restores the existing Drive snapshot; the same runtime verifies and reuses local artifacts. There is no need to delete directories or retrain the tokenizer manually.
3. Check for `Existing seed verified`, the actual `seed_summary.json`, and the backup completion message. If integrity verification fails, investigate the specific missing files instead of bypassing the check.
4. Open [02a](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb), select a T4 GPU, and confirm `cuda_available=true` before training.

## Validation Performed

The old code reproduced the same existing-directory error on the actual local seed data. After the fix, the build and read-only verification succeeded; SHA256 hashes and modification times for all 69 existing files were unchanged. The machine-readable record is [colab_rerun_fix.json](../reports/colab_rerun_fix.json).

All 112 local tests passed, as did format and syntax checks for nine notebooks. After tokenizers 0.23.1 was installed in isolation, 25 tokenizer, data, and training tests passed, and the existing 8K tokenizer loaded successfully. CI coverage was added for Python 3.12 and 3.13; the Actions run for the corresponding commit is the authoritative CI result. These checks did not replace a rerun in the user's Colab environment or start new GPU pretraining.

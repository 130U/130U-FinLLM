# 24M Trial | Colab Kernel Installation and Training Entry-Point Fixes

Scope: public engineering documentation for the 24M Trial, preserving this kernel failure, recovery semantics, and engineering validation. See the [project overview](../README.md) for the broader project.

Issue record: [GitHub #1](https://github.com/130U/130U-FinLLM/issues/1). On 2026-10-09, user screenshots showed that 02a had a Tesla T4 selected, but its training cell raised `ModuleNotFoundError` while importing `finllm.runtime`. In the same screenshot set, 01a used a CPU runtime. Its disconnection message alone cannot establish an out-of-memory failure or identify the cause of the disconnect.

2026-10-11 update: user-supplied T4 screenshots showed that 02a completed 61 updates and 999,424 tokens, with a successful Drive backup. Subsequent 02b logs and backup output showed 300 cumulative steps and 4,914,941 effective tokens, with the final snapshot verified. The user's run had passed the earlier import failure. See the [T4 training record](T4_PILOT.md) for reported figures and the scope of the evidence. These are user-supplied execution records; the cloud checkpoint was not independently read.

## Root Cause and Fix

The project uses a `src` layout. The notebook's Python kernel starts before `pip install -e .`. The `.pth` file created by an editable installation is read by subsequent Python processes, but an already-running kernel does not automatically reread it. Consequently, a successful `python -m finllm environment` did not establish that a later in-kernel `from finllm.runtime import ...` would succeed.

The shared installation cell now explicitly activates the current repository's `src` directory, refreshes only cached `finllm` modules, and checks the actual import paths and checkpoint recovery interface in both the current kernel and a new Python process. Torch is not reloaded.

When stale project metadata still required `tokenizers<0.23`, upgrading tokenizers before updating the project created a temporary dependency conflict. The editable project and compatible tokenizer are now installed in a single pip transaction, constrained to the full version of the currently installed Torch. Project base dependencies and the Transformers tokenizer constraint are checked afterward.

Two training workflow issues were also corrected:

- Rerunning after the training budget has been reached reuses the original results read-only, preserving the checkpoint, backup, and original throughput and GPU-memory measurements. If only a checkpoint is restored and the original summary is missing, no new throughput or memory result is invented.
- After a training failure, 02a's final backup cell explicitly stops instead of continuing to display completion. During normal training, update counts and loss are reported as training progresses.

## Validation Method and Scope

The new `tests/test_notebook_execution.py` uses a real, persistent Jupyter kernel, starting it before performing the editable installation. The test reproduces the old import failure and executes the generated notebook's installation, 01a configuration restoration, and 02a restoration, configuration, training, backup, and rerun cells. A separate Python environment and kernel then restore the data and checkpoint into a different directory, verify continuation from two updates to three, and confirm matching checkpoint hashes in the persistent and working directories.

External Git and Drive services are represented by local directories in these tests. Dependency installation reuses existing libraries offline, and the GPU budget is explicitly replaced with a tiny CPU test configuration. Package installation, kernel imports, parameter updates, and checkpoint restoration execute for real. This validates the notebook execution chain; it is not an acceptance test of the cloud Google Drive service or T4 performance. Earlier tests covered subprocesses and syntax but missed this kernel-lifecycle distinction.

All 120 local tests passed in 73.26 seconds, along with format and syntax checks for nine notebooks and `pip check`. The machine-readable record is [colab_kernel_fix.json](../reports/colab_kernel_fix.json). CI runs the same full suite on Python 3.12 and 3.13; consult the Actions run for the fix commit for its result.

## Continuing from an Existing Backup

If the 01a `seed_v1` backup is already on Drive, reopen the [latest 02a](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb) and save a copy. Keep the T4 runtime and `SEED_VERSION = "seed_v1"`, then run from the first cell in order. The installation cell should report successful import checks for both the current notebook kernel and a new Python process. Subsequent cells restore the existing data, tokenizer, and progress.

There is no need to delete Drive files or repeat 01a. A notebook already open or saved in Drive does not update its cell code automatically; use the updated entry point above. Rerunning after training completes returns `resumed_no_training: true`, and the original `run_summary.json` retains the measurements from the actual training run.

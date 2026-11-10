# Notebook runtime and recovery

The Trial notebooks use a shared installation cell and explicit checkpoint contracts. Run cells from the beginning in the published order. A saved Drive copy has its own cell code; open the repository notebook when selecting a release.

## Installation

The first cell resolves a Git reference to a revision, verifies the repository location, and checks out that revision. It keeps the installed Colab CUDA/PyTorch build and resolves the project plus compatible tokenizer requirements in one pip transaction. Both the project constraints and any installed Transformers tokenizer constraints are checked.

Editable installations expose `src/` to new Python processes. The running notebook kernel explicitly activates that directory, refreshes only `finllm` modules, and checks imports. A separate process verifies the CLI import path. Torch and live CUDA state are not reloaded.

## Artifacts and reruns

Completed input downloads, exclusion pools, tokenizers and shards are reused after configuration, provenance and hash checks. A new runtime restores the matching Drive snapshot. Human-review CSV edits remain user-owned. Incomplete or mismatched artifacts stop with an error; they are not silently overwritten.

The 01a preparation notebook can run on CPU. Training notebooks require their specified device. Confirm `cuda_available=true` before a T4 run. A tiny CPU fixture validates execution contracts without establishing GPU throughput or cloud-service availability.

## Continuation and completion

Resume restores model, optimizer, schedule, random state, data position and progress from a verified checkpoint. A new training stage transfers compatible model weights and initializes fresh optimization state. Keep these two operations distinct.

When a training budget is already complete, rerunning verifies and reuses results without adding updates or replacing measured throughput and memory figures. A checkpoint without its original report does not create new performance measurements. Backup cells require successful training and verify the resulting snapshot before reporting completion.

## Validation

The CPU test suite executes generated notebook cells in persistent Jupyter kernels, checks editable installation, and restores checkpoints across isolated local environments. Git and Drive services are represented by local directories; tokenization, parameter updates, imports and checkpoint integrity execute for real. These checks do not replace a live cloud run. Actual Trial measurements and their evidence scope are recorded in [experiment results](experiment-results.md).

# 24M Trial | Full Transfer from Colab to Kaggle

Scope: public engineering documentation for the 24M Trial, preserving the 04b/04c full-transfer workflow. Subsequent actual SFT used the [04d/04e lightweight workflow](KAGGLE_LITE.md); that does not change the execution record for this workflow. See the [project overview](../README.md).

This migration reads the completed `domain_v1` checkpoint, user-confirmed entity labels, 8K tokenizer, and upstream exclusion pools while preserving the original Drive artifacts. Kaggle SFT is a new experiment: it loads only the domain model's weights and initializes a new optimizer. The T4 ×2 setting in the screenshot provides two GPUs; this implementation uses one and does not treat their combined memory as a single allocation pool.

## 1. Export on Colab CPU

Open [04b Kaggle export](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/04b_kaggle_export.ipynb), select Runtime → Change runtime type → None, run in order, and mount the original Drive. The program verifies the domain weights, tokenizer, task labels, four actual pretraining exclusion pools, and upstream weight lineage before streaming the package to disk.

Record the ZIP SHA256 and `manifest_sha256` from the output. When the browser prompts for a download, wait for `130u_kaggle_transfer.zip` to finish. The file may exceed 1GB; use the actual reported size. It contains model weights and original headlines and should be stored on the user's own computer and treated as private data. The user's Drive has not been independently read; actual cloud verification is established by execution output.

## 2. Prepare Kaggle Inputs and the Notebook

In your own Kaggle account, create a **Private** dataset from the transfer ZIP and verify that the notebook itself is also Private. In the existing notebook, use File → Import Notebook to import this project's `notebooks/04c_kaggle_sft.ipynb`, then use Add Input to attach the private dataset. Retain GPU T4 ×2 and Internet on in Session options.

Paste the ZIP SHA256 and manifest SHA256 reported by 04b into 04c's configuration cell. Kaggle may automatically extract an uploaded ZIP. The notebook detects the extracted directory and verifies its manifest SHA and per-file SHAs; if the ZIP is retained, it verifies the ZIP SHA. Leave both recovery SHAs blank on the first run.

## 3. Run and Back Up

Run in order and record the full 40-character code SHA from the first cell. The program verifies the domain → expansion → seed parent-model chain and actual exclusion pools, performs a single-step check, then trains and compares development results. Outputs are in `/kaggle/working`. Download the final private recovery ZIP and both of its SHAs, or ensure a private notebook version has saved the outputs. Do not treat the working directory as a permanent backup.

To resume across sessions, attach the original transfer data and private recovery data, enter the recovery manifest or ZIP SHA, and pin REF in the first cell to the full code SHA used on the initial run. Strict recovery also requires matching paths, data, device, AMP, PyTorch/CUDA, and other runtime identities. A mismatch stops execution instead of being presented as an exact resume. A sudden interruption can lose local progress that was not separately saved.

This workflow's task is literal-entity stance in news headlines. Security identity and complete event/time isolation have not been verified, so results do not represent the X setting or investment returns. The tests below are engineering validation of the full-transfer workflow. The later lightweight workflow ran in the user's Kaggle environment; see the [Trial experiment report](experiment-results.md).

Local validation: 31 combined tests covering transfer, four-pool checks, SFT, and interruption recovery passed, along with format and Python syntax checks for 17 notebooks. Tests covered simulated Kaggle automatic extraction and a tiny seed → domain training → export → receipt → single-step SFT → recovery sequence. They do not establish success in the user's actual cloud environment.

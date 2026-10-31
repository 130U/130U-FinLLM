# 24M Trial | Lightweight Handoff of the Existing Model to Kaggle

Scope: public engineering documentation for the 24M Trial, recording the published 04d/04e migration and recovery workflow. This workflow subsequently produced the initial and balanced SFT results; see the [Trial experiment report](experiment-results.md) and [project overview](../README.md).

This workflow retains the trained `domain_v1` weights and 8K tokenizer without requiring a download of the 1.54 GiB full transfer package. The lightweight package contains model weights, the tokenizer, 1,008 user-reviewed headline samples, and an identity manifest. Its actual size is reported by the export, approximately 100 MB. The original model and data in Colab are not deleted.

## 1. Export the Lightweight Package on Colab CPU

Open the [04d lightweight export notebook](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/04d_colab_lite_export.ipynb). Select Runtime → Change runtime type → None, run from the first cell, and authorize the existing Drive. The program verifies the domain checkpoint, tokenizer, and task data, then exports `130u_kaggle_lite.zip`. Record the reported `manifest_sha256` and ZIP SHA256, and wait for the download to finish.

The user's earlier 04b output showed that all four actual pretraining exclusion pools covered the 1,008 headlines, each with `missing: 0`. The lightweight package records this reported evidence but does not include the four historical exclusion pools. Kaggle cannot independently reverify the omitted original data.

## 2. Import the Training Notebook and Private Data into Kaggle

In the user's own Kaggle account, create a **Private dataset** from the lightweight ZIP and confirm that the notebook itself is also Private. Open the existing Kaggle notebook, use File → Import Notebook to import this project's `notebooks/04e_kaggle_lite_sft.ipynb`, and use Add Input to attach the private dataset.

Select T4 x2 and enable Internet. The current implementation uses only the first T4. Enter the `manifest_sha256` reported by 04d. If Kaggle retains the ZIP rather than automatically extracted files, also enter the ZIP SHA256. Leave other recovery settings blank for the first run.

## 3. Training, Results, and Saving

Run the cells in order. The first run loads only the 74-step domain model's weights and creates a new SFT optimizer; it does not restore the Colab optimizer state. The program compares development-set results before and after SFT and outputs a recovery checkpoint with its identified SHA256.

`/kaggle/working` is not guaranteed to persist. After training, save a Private notebook version and confirm that the actual outputs were saved, or download `latest.pt` and `latest.pt.json`. Also retain the private lightweight dataset, the initial code commit SHA, and tokenizer. Before resuming across sessions, restore the original run directory and use matching code, data, paths, device, PyTorch/CUDA, and AMP settings.

This entry point concerns exploratory stance classification toward literal entities in news headlines. Asset identity, semantic-event and temporal isolation, and the X setting have not been validated; it does not demonstrate security-price prediction ability. Lightweight export, Kaggle input, training, and recovery first underwent local tiny-model tests. The user later completed both initial and balanced SFT in the actual Kaggle environment. See the [Trial experiment report](experiment-results.md) for results and the evidence available for each backup.

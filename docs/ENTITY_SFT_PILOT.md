# 24M Trial | 04a Entity-Stance SFT Pilot

Scope: public engineering documentation for the 24M Trial, preserving the 04a entry point and failure records. Subsequent SFT ran through the lightweight Kaggle workflow; see the [Trial experiment report](experiment-results.md) for results and the [project overview](../README.md) for the broader project.

This pilot starts from the completed 74-step `domain_v1` checkpoint, reuses the original 8K tokenizer, and trains three-way stance classification toward literal targets in headlines. It is a research pilot intended to produce a measurable model promptly. Its targets are not verified securities, and its samples are financial news headlines rather than X posts.

## Running in Colab

1. Open the [04a notebook](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/04a_entity_sft_pilot.ipynb) and select a T4 GPU.
2. The private upload package already exists locally at `runs/task_pilot/entity_v1_upload.zip`. Run from the top, mount Drive, and select this ZIP when prompted to upload.
3. The notebook checks the package against the existing pretraining exclusion manifests, verifies the domain checkpoint and original tokenizer on Drive, freezes the parent model, and runs a single-step check followed by two training epochs. It then compares the domain parent and the best SFT model on the same development set and performs the final backup.
4. Retain the final training and development output and the backup result with `verified: true`. After disconnection, confirm the old training process has stopped before restoring from the first cell. Complete-state checks are automatic; do not run two training copies concurrently.

The ZIP contains 1008 user-confirmed headline-stance records: 705 training, 101 development, 101 calibration, and 101 test. The last two partitions are not used for model selection or automatic evaluation in this notebook. Lexical and source-ID isolation are in place, but semantic-event and temporal isolation have not been fully verified; development metrics are exploratory. Third-party rights held by the original publishers were not independently verified. The data remains local and on personal Drive, rather than being uploaded to public GitHub.

The model in this entry point accepts a single-word target appearing literally in the headline. Evaluation and inference require explicit `--target-mode literal_entity`. 04a outputs three uncalibrated class probabilities and a directional score. The later balanced-SFT inference export separately fitted temperature and a review threshold. Verified security mapping, more samples, DPO, and a new independent test remain separate work items; headline-stance performance is not stock-price prediction ability.

At delivery, 04a passed 23 combined local checks covering data, entity prediction, SFT, and recovery, plus format checks for 15 notebooks. A real tiny CPU test exercised domain training → SFT → development evaluation → backup. These are engineering checks for this entry point; the actual subsequent SFT execution evidence is in the Trial experiment report linked above. Package SHA256: `e87d12013271eb8f93f926b0b2c71cef15b0c777cc58d6e08348773dbd6a7be6`.

## Fix for the 04a Exclusion-Manifest Mismatch

User screenshots showed successful verification of the uploaded task package and Drive snapshot, followed by `Task package exclusions differ from the actual pretrained-data lineage` during preparation. SFT had not started. The entry point incorrectly required identical file SHA values for the locally prepared exclusions and the historical cloud pretraining exclusions. They were built independently; byte equality alone cannot establish whether the headlines are covered.

The fix reads four actual exclusion manifests from Drive: seed and expansion, each with training and validation manifests. It verifies their snapshots and their association with the actual pretraining checkpoint, shards, and domain mixture, then checks every headline in all five task partitions against the corresponding exclusion sets. Different manifest versions allow training to proceed only after every check passes. Missing manifests or headlines, corrupted manifests, and lineage mismatches still raise errors. Actual cloud state must be verified by running the user's Colab notebook; local tests do not replace cloud results.

After the fix is merged, restart the Colab runtime and run from the first cell so the kernel loads the latest code. An existing `tasks/entity_v1` snapshot is reused after verification, so no repeated upload is required. Retain actual output and the verified final backup.

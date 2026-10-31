# 130U-FinLLM | Models and versions

130U-FinLLM comprises a private 2.4B model and a public 24M Trial. Both scales follow the project's task design and training principles: learning stance and relationships in financial text through explicit targets, opinion holders, and evidence scopes.

The executable code, default configurations, notebooks, and run records in this repository belong to the **24M Trial**. This page also presents the architecture of the **2.4B private version**. Private data and weights are managed within the personal research workflow.

## Architecture

| Component | 2.4B private model | 24M public Trial |
|---|---:|---:|
| Total parameters | 2,397,459,968 | 24,255,744 |
| Transformer layers | 28 | 6 |
| Hidden size | 2,560 | 512 |
| FFN intermediate size | 8,320 | 1,664 |
| Query heads per layer | 20 | 8 |
| Key/value heads per layer | 4 | 4 |
| Head dimension | 128 | 64 |
| Vocabulary size | 65,536 | 8,192 |
| Context limit | 4,096 | 512 |
| RoPE theta | 1,000,000 | 1,000,000 |
| Input embeddings / output projection | Tied weights | Tied weights |

The backbone uses a dense autoregressive decoder, GQA, SwiGLU, RMSNorm, and QK normalization. In the 2.4B architecture, each KV group serves 5 query heads. Each layer contains one attention module and one feed-forward module. The 2.4B stance and metric-direction tasks share the backbone and vocabulary head through conditioned inputs; the executable public Trial interface implements the stance task only.

### Parameter breakdown

Each 2.4B layer contains 79,631,616 parameters; 28 layers contain 2,229,685,248. The shared vocabulary matrix contributes 167,772,160 parameters, and the final RMSNorm contributes 2,560, totaling **2,397,459,968**. These are total parameters, not adapter parameters or activated MoE parameters.

## Task interface

Investment stance, financial metric direction, and personal judgment have separate definitions. Source-text stance identifies what a specified opinion holder expresses. The metric task identifies a change relative to an explicit comparator. Personal judgment incorporates the information and constraints available at the time.

The public Trial accepts English text and a specified target. It returns `bearish / neutral / bullish`, normalized probabilities over the three classes, a direction score, and review status. The direction score is `100 × (P(bullish) − P(bearish))`, representing the relative direction of the expressed stance. Valid inputs with low confidence retain their probabilities and are flagged for review. If input eligibility checks fail, the label, probabilities, and score are `null`.

The Trial headline experiments use literal-entity target mode with a default context limit of 512 tokens. Input checks cover format, length, and literal-target or audited-alias matching, depending on the target mode. Callers expand multiple targets into separate requests. Overlength inputs and those that fail matching are flagged for review. See [`model_24m.json`](../configs/model_24m.json).

## Training system

The 2.4B system combines foundational language pretraining, financial mixed adaptation, target-conditioned SFT, and calibrated probability use. Semantic permissions control edit supervision. Global objective accounting manages sample families, task weights, and supervised positions. Runtime optimizations preserve full-vocabulary loss and sample information boundaries. The 32k human-gold allocation and 55B input tokens are recipe quantities; actual inventories, consumption, and performance are tracked separately in run records. See the [training system](TRAINING_SYSTEM.md).

The public Trial demonstrates data preparation, foundational and financial-domain training, stance SFT, calibration, and inference export. Its observations, sample counts, and experimental conditions are recorded separately in the [Trial experiment report](experiment-results.md); those tables describe the public version.

## Versioned delivery

Model weights, tokenizer, task protocol, calibration parameters, and training state are bound to a version. Resuming the same run restores its optimizer, random state, and data position. A stage transition transfers compatible weights and creates new optimization state. Moving from 24M to 2.4B carries forward task and engineering methods; each version uses its own vocabulary, parameters, and calibration configuration.

## Code provenance and licensing

The public Trial backbone references a fixed MiniMind revision; modifications are documented in the [third-party notices](../THIRD_PARTY_NOTICES.md). Code is licensed under [Apache-2.0](../LICENSE). Corpora, derived data, and model weights have separately tracked provenance and usage conditions.

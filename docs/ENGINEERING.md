# Engineering retrospective | From 24M to 2.4B

The 24M Trial is the public engineering entry point; the 2.4B version serves private use. The engineering objective is to keep data semantics, optimization targets, and runtime execution aligned. The following lessons connect concrete Trial observations to the project's two core designs.

## Class learning: Accuracy can conceal directional failure

The initial Trial predicted neutral for all 101 development examples. Because 76 were neutral, accuracy remained 75.25%. Recall for both directional classes was zero, and macro-F1 was 0.2863.

After balancing class exposure and increasing the training budget, development macro-F1 reached 0.6549. The model correctly classified 17 more directional examples while correctly classifying 14 fewer neutral examples, raising total correct predictions from 76 to 79. This connects learning opportunities, directional recall, and the cost of false positives. See the [experiment report](experiment-results.md) for per-class results and experimental conditions.

The engineering choice is to separate the training distribution from natural-distribution evaluation. Sampling can increase minority-class exposure while development and acceptance tests retain the observed class structure. Independent examples, repeated exposures, and successful updates are counted separately to make recipe changes interpretable.

## Task semantics: The model and annotator must answer the same question

An author's expressed stance, a change in an operating metric, and a user's personal judgment are different supervision targets. Task fields specify the target, opinion holder, and evidence scope. Personalized judgments additionally reference the information and constraints available at the time.

Financial semantic permissions apply these definitions to edited examples. Reviewers separately identify relations that should change, should be preserved, or should receive no constraint. The model should preserve judgments under irrelevant transformations and respond appropriately when a target or valid evidence changes.

## Data accounting: More views do not create independent information

Original text, QA, different target queries, and edits concerning the same event belong to one family. Splitting families before generating views extends evaluation boundaries to all derivative material. Provenance and version accompany the content; personal historical material also distinguishes an original judgment from a later revision.

Training combines source text, QA, and single-label supervision. Input tokens, valid loss positions, and family weights are managed separately. Distributed execution and gradient accumulation use term-specific effective denominators across the entire update window. The resulting learning weights can be reconstructed from the ledger.

## Scale and runtime: Optimize execution while preserving the objective

For the 2.4B architecture, FP32 parameters, gradients, and two Adam state tensors require approximately 38.36 GB, excluding activations and temporary tensors. Precision, state sharding, recomputation, and packing jointly determine runtime requirements.

Single-label tasks select supervised prediction positions before the output projection while retaining full-vocabulary CE. Packing uses segmented causal attention. Input embeddings and the output head remain the same parameter object. Equivalence checks cover effective logits, loss, gradients, one update, and the next update after restoration.

This places performance optimization within explicit mathematical and information boundaries, allowing scale changes to be interpreted against the same experimental objective. Formulas and computation diagrams are in the [training system](TRAINING_SYSTEM.md).

## Bringing outputs into the workflow

Classification, probability scaling, and automatic acceptance rules are handled separately. Calibration and threshold settings are bound to the model and input protocol, with risk and coverage reported together. Model weights, tokenizer, task definitions, and calibration parameters constitute a versioned release.

[Project overview](PROJECT_OVERVIEW.md) · [Model card](MODEL_CARD.md) · [Trial guide](TRIAL_GUIDE.md)

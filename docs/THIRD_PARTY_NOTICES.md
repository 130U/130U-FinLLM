# Third-party attribution

## MiniMind

The dense decoder in `src/finllm/model.py` is adapted from the MiniMind project by
jingyaogong and contributors, distributed under the Apache License, Version 2.0.

- Project: https://github.com/jingyaogong/minimind
- Pinned reference: `f659b55761b754d306bd140573493a6543cafd7f`
- Source: https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/model/model_minimind.py
- License: https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/LICENSE
- A copy of the Apache-2.0 license is included in this repository's `LICENSE`.

Changes made for 130U-FinLLM in 2026: a native PyTorch dataclass configuration;
a 24,255,744-parameter dense default; no Transformers, MoE, generation cache,
or internal next-token loss shift; explicit causal/padding masks; input checks;
and tests for causal isolation, padding behavior and finite gradients. RMSNorm,
RoPE, grouped-query attention, SwiGLU and tied input/output embeddings follow
the pinned MiniMind architecture. No pretrained MiniMind weights or tokenizer
are included or downloaded. This attribution does not imply upstream endorsement.

## Libraries and learning references

PyTorch, NumPy and Hugging Face Tokenizers remain separately licensed dependencies.
Their installed distributions carry their own license notices; they are not
vendored into this repository.

- Tokenizers: https://github.com/huggingface/tokenizers
- PyTorch: https://github.com/pytorch/pytorch
- NumPy: https://github.com/numpy/numpy

The project plan also cites the following educational references. They are not
bundled as alternative model implementations:

- Sebastian Raschka, LLMs-from-scratch: https://github.com/rasbt/LLMs-from-scratch
- Andrej Karpathy, nanoGPT: https://github.com/karpathy/nanoGPT

The code license does not grant rights to external training data. Each source
must have its own provenance and applicable training-use terms recorded.

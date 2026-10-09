import pytest
import torch
from torch.nn import functional as F

from finllm.model import FinLLM, ModelConfig


def small_model(**overrides):
    config = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, intermediate_size=80,
                  max_position_embeddings=16)
    config.update(overrides)
    torch.manual_seed(11)
    return FinLLM(ModelConfig(**config))


def test_full_model_parameter_count_and_tied_weights():
    model = FinLLM(ModelConfig())
    assert model.num_parameters() == 24_255_744
    assert model.lm_head.weight is model.embed_tokens.weight


def test_future_tokens_cannot_change_past_logits():
    model = small_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 8, 9]])
    with torch.no_grad():
        first_logits, second_logits = model(first), model(second)
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])
    assert not torch.allclose(first_logits[:, 3:], second_logits[:, 3:])


def test_padding_does_not_leak_and_all_padding_is_finite():
    model = small_model().eval()
    ids = torch.tensor([[7, 8, 2, 3, 4], [6, 7, 8, 9, 10]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])
    changed = ids.clone()
    changed[mask == 0] = 15
    with torch.no_grad():
        logits = model(ids, mask)
        altered = model(changed, mask)
        unpadded = model(ids[:1, 2:])
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, altered)
    torch.testing.assert_close(logits[:1, 2:], unpadded, atol=1e-6, rtol=1e-5)


def test_next_token_backward_is_finite_and_updates_all_parameters():
    model = small_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
    logits = model(tokens[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, 64), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_rejects_overlong_context_and_incompatible_heads():
    with pytest.raises(ValueError, match="max_position_embeddings"):
        small_model()(torch.ones(1, 17, dtype=torch.long))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(num_attention_heads=7)
    with pytest.raises(ValueError, match="same shape"):
        small_model()(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 1))

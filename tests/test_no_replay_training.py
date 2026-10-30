"""An AMP skip must not make a one-pass stage silently replay its corpus."""
import copy

import pytest

import finllm.training as training
from finllm.checkpoint import load_checkpoint, sha256_file
from test_training import tiny_config, few_cpu_threads


def test_one_pass_stops_before_any_repeated_microbatch(tiny_config, monkeypatch):
    tiny_config["forbid_epoch_replay"] = True
    seen = []
    original = training._batch

    def track(dataset, indices, *args):
        seen.extend(indices)
        return original(dataset, indices, *args)

    monkeypatch.setattr(training, "_batch", track)
    with pytest.raises(ValueError, match="epoch replay is forbidden"):
        training.run_training(tiny_config)
    checkpoint = load_checkpoint(tiny_config["output_dir"] + "/latest.pt")
    assert len(seen) == len(set(seen)) == 12
    assert checkpoint["progress"]["updates"] == 3
    assert checkpoint["progress"]["epoch"] == 0
    assert checkpoint["progress"]["cursor"] == 12


def test_amp_skip_consumes_data_but_cannot_enable_replay(tiny_config, monkeypatch):
    class SkipFirstAttempt:
        def __init__(self, *args, **kwargs):
            self.attempt, self.factor = 0, 2.0
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def get_scale(self):
            return self.factor
        def step(self, optimizer):
            if self.attempt:
                optimizer.step()
        def update(self):
            if not self.attempt:
                self.factor /= 2
            self.attempt += 1
        def state_dict(self):
            return {"attempt": self.attempt, "factor": self.factor}

    monkeypatch.setattr(training.torch.amp, "GradScaler", SkipFirstAttempt)
    tiny_config["forbid_epoch_replay"] = True
    with pytest.raises(ValueError, match="epoch replay is forbidden"):
        training.run_training(tiny_config)
    checkpoint = load_checkpoint(tiny_config["output_dir"] + "/latest.pt")
    assert checkpoint["progress"]["updates"] == 2
    assert checkpoint["progress"]["attempted_updates"] == 3
    assert checkpoint["progress"]["cursor"] == 12
    assert checkpoint["scheduler_state"]["last_epoch"] == 2


def test_resume_cannot_disable_one_pass_contract(tiny_config):
    tiny_config["forbid_epoch_replay"] = True
    tiny_config["train"]["max_updates"] = 2
    result = training.run_training(tiny_config)
    checkpoint = result["latest_checkpoint"]
    before = sha256_file(checkpoint)
    changed = copy.deepcopy(tiny_config)
    changed.update(resume=checkpoint, forbid_epoch_replay=False)
    changed["train"]["max_updates"] = 3
    with pytest.raises(ValueError, match="provenance mismatch"):
        training.run_training(changed)
    assert sha256_file(checkpoint) == before
    changed["forbid_epoch_replay"] = True
    assert training.run_training(changed)["updates"] == 3

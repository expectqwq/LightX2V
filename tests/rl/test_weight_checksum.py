import hashlib

import torch

from lightx2v.rl.weights import model_state, tensor_checksum


def test_tensor_checksum_supports_scalar_bfloat16():
    tensor = torch.tensor(1.5, dtype=torch.bfloat16)
    expected = hashlib.sha256(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()

    assert tensor_checksum(tensor) == expected


class _StateNode:
    def __init__(self, values):
        self.values = values

    def state_dict(self, destination):
        destination.update(self.values)


class _Model:
    def __init__(self):
        self.pre_weight = _StateNode(
            {
                "model.weight": torch.ones(4),
                "diffusion_model.blocks.norm.diff": torch.tensor(0.0),
                "diffusion_model.blocks.active_norm.diff": torch.ones(4),
            }
        )
        self.transformer_weights = _StateNode({})


def test_model_state_skips_only_inactive_scalar_diff_placeholders():
    state = model_state(_Model())

    assert set(state) == {
        "model.weight",
        "diffusion_model.blocks.active_norm.diff",
    }

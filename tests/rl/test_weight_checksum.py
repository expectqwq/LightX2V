import torch

from lightx2v.rl.weights import model_state


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

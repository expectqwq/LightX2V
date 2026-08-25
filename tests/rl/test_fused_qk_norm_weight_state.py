import torch

from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFusedQKNorm3DRope


def test_fused_qk_norm_exposes_all_source_tensors():
    names = ["q_t", "q_hw", "k_t", "k_hw"]
    module = RMSWeightFusedQKNorm3DRope(*names)
    tensors = {name: torch.randn(8) for name in names}
    module.load(tensors)

    state = module.state_dict()

    assert list(state) == names
    assert all(state[name] is tensors[name] for name in names)

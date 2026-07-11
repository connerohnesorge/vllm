import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _pack_sequential_samples,
)


def test_pack_sequential_samples() -> None:
    target = torch.tensor([[10, 20, 30, 40], [10, 20, 30, 40]])
    drafts = torch.tensor(
        [[0, 10, 99, 30], [0, 10, 20, 30]], dtype=torch.int32
    )

    sampled, lengths = _pack_sequential_samples(target, drafts)

    assert sampled.tolist() == [[10, 20, -1, -1], [10, 20, 30, 40]]
    assert lengths.tolist() == [2, 4]

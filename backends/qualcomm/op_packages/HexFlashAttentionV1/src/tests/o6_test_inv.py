import torch

TEST_OP_ID = 6
IMPL = torch.reciprocal
DTYPE = torch.float32

INPUT_NUM = 1


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [(2 * torch.randn(1, 1, 1, 32, dtype=DTYPE),) for _ in range(10)]

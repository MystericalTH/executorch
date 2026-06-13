import torch

TEST_OP_ID = 0
IMPL = torch.exp2
DTYPE = torch.float16

INPUT_NUM = 1


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [(2 * torch.randn(1, 1, 1, 64, dtype=DTYPE),) for _ in range(10)]

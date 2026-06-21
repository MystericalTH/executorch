import torch

TEST_OP_ID = 4


def transpose23(input: torch.Tensor):
    return torch.transpose(input, 2, 3)


IMPL = transpose23
DTYPE = torch.float16

INPUT_NUM = 1


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [(torch.randn(1, 1, 64, 64, dtype=DTYPE),) for _ in range(10)]

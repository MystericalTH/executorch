import torch

TEST_OP_ID = 51


def matmul_transpose(input0: torch.Tensor, input1: torch.Tensor) -> torch.Tensor:
    return torch.matmul(input0, input1.transpose(2, 3))


IMPL = matmul_transpose
DTYPE = torch.float16

INPUT_NUM = 2


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [
        (
            torch.randn(1, 1, 1, 128, dtype=DTYPE),
            torch.randn(1, 1, 64, 128, dtype=DTYPE),
        )
        for _ in range(10)
    ]

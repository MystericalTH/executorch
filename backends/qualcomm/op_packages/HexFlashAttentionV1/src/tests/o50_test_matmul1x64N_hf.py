import torch

TEST_OP_ID = 50
IMPL = torch.matmul
DTYPE = torch.float16

INPUT_NUM = 2


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [
        (
            torch.randn(1, 1, 1, 128, dtype=DTYPE),
            torch.randn(1, 1, 128, 256, dtype=DTYPE),
        )
        for _ in range(10)
    ]

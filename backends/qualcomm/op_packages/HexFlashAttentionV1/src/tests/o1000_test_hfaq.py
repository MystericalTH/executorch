import math

import torch

TEST_OP_ID = 1000
DTYPE = torch.float32


class TestModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        scale = float(1 / math.sqrt(query.size(3)))
        return torch.ops.hex_flash_test.hfaq.default(
            query,
            key,
            value,
            attn_mask,
            scale=scale,
            enable_gqa=1,
        )


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    HEADS = 32
    KV_SEQ_LEN = 64 * 128
    EMB_LEN = 128
    return [
        (
            torch.randn(1, HEADS, 1, EMB_LEN, dtype=DTYPE),
            torch.randn(1, HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, 1, 1, KV_SEQ_LEN, dtype=DTYPE),
        )
        for _ in range(5)
    ]

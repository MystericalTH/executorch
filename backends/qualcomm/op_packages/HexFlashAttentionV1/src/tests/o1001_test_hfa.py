import math

import torch

TEST_OP_ID = 1001
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
        return torch.ops.hex_flash_test.hfa.default(
            query,
            key,
            value,
            attn_mask,
            is_causal=False,
            enable_gqa=True,
            scale=scale,
        )


def generate_sample_inputs(n=8) -> list[tuple[torch.Tensor]]:
    Q_SEQ_LEN = 1
    Q_HEADS = 32
    KV_HEADS = 8
    KV_SEQ_LEN = 64 * n
    EMB_LEN = 128
    return [
        (
            torch.randn(1, Q_HEADS, Q_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, KV_HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, KV_HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, 1, Q_SEQ_LEN, KV_SEQ_LEN, dtype=DTYPE),
        )
        for _ in range(1)
    ]

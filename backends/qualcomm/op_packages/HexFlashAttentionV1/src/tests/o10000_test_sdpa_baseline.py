import math

import torch

TEST_OP_ID = 10000
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
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            is_causal=False,
            enable_gqa=True,
            scale=scale,
        )


def generate_sample_inputs(n=8) -> list[tuple[torch.Tensor]]:
    Q_HEADS = 32
    KV_HEADS = 8
    KV_SEQ_LEN = 64 * n
    EMB_LEN = 128
    return [
        (
            torch.randn(1, Q_HEADS, 1, EMB_LEN, dtype=DTYPE),
            torch.randn(1, KV_HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, KV_HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, 1, 1, KV_SEQ_LEN, dtype=DTYPE),
        )
        for _ in range(10)
    ]

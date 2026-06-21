import math

import torch

TEST_OP_ID = 100
DTYPE = torch.float32


def mock_hfaq_local(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    scale = scale if scale else float(1 / math.sqrt(query.size(3)))
    att = scale * (query @ key.transpose(2, 3)) + attn_mask
    head_max = torch.max(att, dim=3, keepdim=True).values
    att = torch.exp(att - head_max)
    head_l = torch.sum(att, dim=3, keepdim=True)
    return torch.concat([att @ value, head_max, head_l], dim=3).transpose(1, 3)


class TestModel(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        scale = float(1 / math.sqrt(query.size(3)))
        return torch.ops.hex_flash_test.hfaq_local.default(
            query, key, value, attn_mask, scale=scale
        )


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    HEADS = 32
    KV_SEQ_LEN = 1024
    EMB_LEN = 128
    return [
        (
            torch.randn(1, HEADS, 1, EMB_LEN, dtype=DTYPE),
            torch.randn(1, HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, HEADS, KV_SEQ_LEN, EMB_LEN, dtype=DTYPE),
            torch.randn(1, 1, 1, KV_SEQ_LEN, dtype=DTYPE),
        )
        for _ in range(1)
    ]

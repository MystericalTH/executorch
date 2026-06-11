import torch

TEST_OP_ID = 2
DTYPE = torch.float32


def mock_hfaq_merge(
    local_0: torch.Tensor,
    local_1: torch.Tensor,
) -> torch.Tensor:
    assert local_0.shape == local_1.shape

    heads = local_0.size(3)
    emb_len = local_0.size(1)

    out = local_0.clone()

    local_max_0 = local_0[:, -2, :, :]
    local_l_0 = local_0[:, -1, :, :]

    local_max_1 = local_1[:, -2, :, :]
    local_l_1 = local_1[:, -1, :, :]

    merged_max = torch.maximum(local_max_0, local_max_1)
    adj_0 = torch.exp2(local_max_0 - merged_max)
    adj_1 = torch.exp2(local_max_1 - merged_max)
    out[:, -2, :, :] = merged_max
    out[:, -1, :, :] = local_l_0 * adj_0 + local_l_1 * adj_1

    out[:, :-2, :, :] = local_0[:, :-2, :, :] * adj_0.expand(
        1, emb_len - 2, 1, heads
    ) + local_1[:, :-2, :, :] * adj_1.expand(1, emb_len - 2, 1, heads)

    return out


class TestModel(torch.nn.Module):
    def forward(
        self,
        local_0: torch.Tensor,
        local_1: torch.Tensor,
    ):
        return torch.ops.hex_flash_test.hfaq_merge.default(local_0, local_1)


def generate_sample_inputs() -> list[tuple[torch.Tensor]]:
    return [
        (
            torch.randn(1, 130, 1, 32, dtype=DTYPE),
            torch.randn(1, 130, 1, 32, dtype=DTYPE),
        )
        for _ in range(10)
    ]

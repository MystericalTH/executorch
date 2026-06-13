import torch

from executorch.backends.qualcomm.op_packages.HexFlashAttentionV1.src.tests import (
    o0_test_exp2_hf,
    o1000_test_hfaq,
    o100_test_hfaq_local,
    o101_test_hfaq_merge,
    o1_test_exp2,
    o2_test_mat_transpose32x32,
    o3_test_mat_transpose32Mx32N,
    o4_test_mat_transpose64x64_hf,
    o50_test_matmul1x64N_hf,
    o5_test_mat_transpose64Mx64N_hf,
    o6_test_inv,
)
from torch.library import impl, Library

test_modules = [
    o0_test_exp2_hf,
    o1_test_exp2,
    o2_test_mat_transpose32x32,
    o3_test_mat_transpose32Mx32N,
    o4_test_mat_transpose64x64_hf,
    o5_test_mat_transpose64Mx64N_hf,
    o6_test_inv,
    o50_test_matmul1x64N_hf,
    o100_test_hfaq_local,
    o101_test_hfaq_merge,
    o1000_test_hfaq,
]


def get_test_op_module(op_id: int):
    for module in test_modules:
        if op_id == module.TEST_OP_ID:
            return module
    raise Exception("unknown op id")


test_lib = Library("hex_flash_test", "DEF")

test_lib.define(
    """
    test_op(
        Tensor input_0,
        int op_id
    ) -> Tensor
"""
)

test_lib.define(
    """
    test_op.out(
        Tensor input_0,
        int op_id,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
"""
)


@impl(test_lib, "test_op", dispatch_key="CompositeExplicitAutograd")
def test_op_impl(input_0: torch.Tensor, op_id: int) -> torch.Tensor:
    return get_test_op_module(op_id).IMPL(input_0)


@impl(
    test_lib,
    "test_op.out",
    dispatch_key="CompositeExplicitAutograd",
)
def test_op_out_impl(
    input_0: torch.Tensor, op_id: int, *, out: torch.Tensor = None
) -> torch.Tensor:
    result = test_op_impl(input_0, op_id)
    out.copy_(result)
    return out


test_lib.define(
    """
    test_op_2_input(
        Tensor input_0,
        Tensor input_1,
        int op_id
    ) -> Tensor
"""
)

test_lib.define(
    """
    test_op_2_input.out(
        Tensor input_0,
        Tensor input_1,
        int op_id,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
"""
)


@impl(test_lib, "test_op_2_input", dispatch_key="CompositeExplicitAutograd")
def test_op_2_input_impl(
    input_0: torch.Tensor,
    input_1: torch.Tensor,
    op_id: int,
) -> torch.Tensor:
    return get_test_op_module(op_id).IMPL(input_0, input_1)


@impl(
    test_lib,
    "test_op_2_input.out",
    dispatch_key="CompositeExplicitAutograd",
)
def test_op_2_input_out_impl(
    input_0: torch.Tensor,
    input_1: torch.Tensor,
    op_id: int,
    *,
    out: torch.Tensor = None,
) -> torch.Tensor:
    result = test_op_impl(input_0, input_1, op_id)
    out.copy_(result)
    return out


test_lib.define(
    """
    hfaq_merge(
        Tensor local_0,
        Tensor local_1
    ) -> Tensor
"""
)

test_lib.define(
    """
    hfaq_merge.out(
        Tensor local_0,
        Tensor local_1,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
"""
)


@impl(test_lib, "hfaq_merge", dispatch_key="CompositeExplicitAutograd")
def hfaq_merge_impl(
    local_0: torch.Tensor,
    local_1: torch.Tensor,
) -> torch.Tensor:
    return o101_test_hfaq_merge.mock_hfaq_merge(local_0, local_1)


@impl(
    test_lib,
    "hfaq_merge.out",
    dispatch_key="CompositeExplicitAutograd",
)
def hfaq_merge_out_impl(
    local_0: torch.Tensor,
    local_1: torch.Tensor,
    *,
    out: torch.Tensor = None,
) -> torch.Tensor:
    result = hfaq_merge_impl(local_0, local_1)
    out.copy_(result)
    return out


test_lib.define(
    """
    hfaq_local(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor attn_mask,
        float? scale=None
    ) -> Tensor
"""
)

test_lib.define(
    """
    hfaq_local.out(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor attn_mask,
        float? scale=None,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
"""
)


@impl(test_lib, "hfaq_local", dispatch_key="CompositeExplicitAutograd")
def hfaq_local_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    return o100_test_hfaq_local.mock_hfaq_local(query, key, value, attn_mask, scale)


@impl(
    test_lib,
    "hfaq_local.out",
    dispatch_key="CompositeExplicitAutograd",
)
def hfaq_local_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None,
    *,
    out: torch.Tensor = None,
) -> torch.Tensor:
    result = hfaq_local_impl(query, key, value, attn_mask, scale)
    out.copy_(result)
    return out


test_lib.define(
    """
    hfaq(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor attn_mask,
        float? scale=None,
        int? enable_gqa=0
    ) -> Tensor
"""
)

test_lib.define(
    """
    hfaq.out(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor attn_mask,
        float? scale=None,
        int? enable_gqa=0,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
"""
)


@impl(test_lib, "hfaq", dispatch_key="CompositeExplicitAutograd")
def hfaq_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None,
    enable_gqa: int = 0,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask, scale=scale, enable_gqa=bool(enable_gqa)
    )


@impl(
    test_lib,
    "hfaq.out",
    dispatch_key="CompositeExplicitAutograd",
)
def hfaq_out_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float = None,
    enable_gqa: int = 0,
    *,
    out: torch.Tensor = None,
) -> torch.Tensor:
    result = hfaq_impl(query, key, value, attn_mask, scale, enable_gqa)
    out.copy_(result)
    return out

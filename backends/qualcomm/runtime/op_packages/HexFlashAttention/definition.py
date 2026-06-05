import os

import torch
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from torch.library import impl, Library

hex_flash_lib = Library("hex_flash", "DEF")

hex_flash_lib.define(
    """
    flash_attention(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor? attn_mask=None,
        float? scale=None
    ) -> Tensor
    """
)


@impl(hex_flash_lib, "flash_attention", dispatch_key="CompositeExplicitAutograd")
def flash_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    scale=None,
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        scale=scale,
    )


hex_flash_lib.define(
    """
    flash_attention.out(
        Tensor query,
        Tensor key,
        Tensor value,
        Tensor? attn_mask=None,
        float? scale=None,
        *,
        Tensor(a!) output
    ) -> Tensor(a!)
    """
)


@impl(
    hex_flash_lib,
    "flash_attention.out",
    dispatch_key="CompositeExplicitAutograd",
)
def flash_attention_out_impl(
    query,
    key,
    value,
    attn_mask=None,
    scale=None,
    *,
    out,
) -> torch.Tensor:
    result = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        scale=scale,
    )
    out.copy_(result)
    return out


def get_hex_flash_op_package_config(
    op_package_dir: str = "backends/qualcomm/runtime/op_packages/HexFlashAttention",
    arch: str = "79",
):
    xml_path = f"{op_package_dir}/config/HexFlashAttention.xml"
    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={
            "FlashAttention": torch.ops.hex_flash.flash_attention.default
        },
    )
    lib_name = f"libQnn{op_package_config.op_package_name}"
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.HTP,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=os.path.abspath(
            f"{op_package_dir}/build/hexagon-v{arch}/{lib_name}_HTP.so"
        ),
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=os.path.abspath(
            f"{op_package_dir}/build/hexagon-v{arch}/{lib_name}.so"
        ),
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=os.path.abspath(
            f"{op_package_dir}/build/x86_64-linux-clang/{lib_name}.so"
        ),
    )
    return op_package_config

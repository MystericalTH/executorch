import os

import torch
from torch.library import Library, impl

from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)

my_op_lib = Library("my_ops", "DEF")

# registering an operator that multiplies input tensor by 3 and returns it.
my_op_lib.define("mul3(Tensor input) -> Tensor")


@impl(my_op_lib, "mul3", dispatch_key="CompositeExplicitAutograd")
def mul3_impl(a: torch.Tensor) -> torch.Tensor:
    return a * 3


# registering the out variant.
my_op_lib.define("mul3.out(Tensor input, *, Tensor(a!) output) -> Tensor(a!)")


@impl(my_op_lib, "mul3.out", dispatch_key="CompositeExplicitAutograd")
def mul3_out_impl(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    out.copy_(a)
    out.mul_(3)
    return out


def register_op(op_package_dir, workspace, xml_path):
    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={"ExampleCustomOp": torch.ops.my_ops.mul3.default},
    )
    lib_name = f"libQnn{op_package_config.op_package_name}"

    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.HTP,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{workspace}/{lib_name}_HTP.so",
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=f"{workspace}/{lib_name}.so",
    )
    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=os.path.abspath(
            f"{op_package_dir}/build/x86_64-linux-clang/{lib_name}.so"
        ),
    )
    return op_package_config, lib_name

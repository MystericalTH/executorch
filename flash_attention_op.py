# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of registering attention custom operator through torch library API."""

import json
import os
import subprocess
import sys
from multiprocessing.connection import Client

import numpy as np
import torch
from torch.library import Library, impl

from executorch.backends.qualcomm.custom_op.annotator import (
    CustomOpsQuantAnnotator,
    IOQuantConfig,
)
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.export_utils import (
    QnnConfig,
    SimpleADB,
    build_executorch_binary,
    generate_inputs,
    get_backend_type,
    make_quantizer,
    setup_common_args_and_variables,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    get_ptq_per_channel_quant_config,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.serialization.qc_schema import (
    QcomChipset,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
    _soc_info_table,
)
from executorch.examples.qualcomm.utils import make_output_dir

# ── Custom op registration ────────────────────────────────────────────────────
lib = Library("my_ns", "DEF")

lib.define("attention_fwd(Tensor Q, Tensor K, Tensor V) -> Tensor")
lib.define(
    "attention_fwd.out(Tensor Q, Tensor K, Tensor V, *, Tensor(a!) output) -> Tensor(a!)"
)


@impl(lib, "attention_fwd", "CompositeExplicitAutograd")
def attention_fwd_impl(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    scale = Q.shape[-1] ** -0.5
    attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    return attn @ V


@impl(lib, "attention_fwd.out", "CompositeExplicitAutograd")
def attention_fwd_out_impl(Q, K, V, *, output):
    output.copy_(attention_fwd_impl(Q, K, V))
    return output


# ─────────────────────────────────────────────────────────────────────────────


class Model(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.ops.my_ns.attention_fwd.default(q, k, v)


def _run(cmd, cwd=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, check=True)


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    if args.build_op_package:
        if "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable HEXAGON_SDK_ROOT must be set")
        print(f"HEXAGON_SDK_ROOT={os.getenv('HEXAGON_SDK_ROOT')}")
        if "ANDROID_NDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable ANDROID_NDK_ROOT must be set")
        print(f"ANDROID_NDK_ROOT={os.getenv('ANDROID_NDK_ROOT')}")

    os.makedirs(args.artifact, exist_ok=True)

    instance = Model()
    pte_filename = "custom_qnn"

    # ── Sample inputs — adjust shapes to match your actual attention kernel ──
    # (batch, heads, seq_len, head_dim)
    batch, heads, seq_len, head_dim = 1, 8, 128, 64
    sample_q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    sample_k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    sample_v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
    sample_inputs = (sample_q, sample_k, sample_v)
    # ─────────────────────────────────────────────────────────────────────────

    # On-device workspace — all pushed files land here
    workspace = f"/data/local/tmp/executorch/{pte_filename}"

    soc_info = _soc_info_table[getattr(QcomChipset, args.soc_model)]
    arch = soc_info.htp_info.htp_arch  # e.g. 79 for SM8750 (S25)

    # ── Op package setup ─────────────────────────────────────────────────────
    xml_path = f"{args.op_package_dir}/config/attention_op_package_htp.xml"
    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        # XML <Name> value → PyTorch op target.
        # Important: export/lowering uses the out overload in the final graph.
        torch_op_name_map={"AttentionFwdOp": torch.ops.my_ns.attention_fwd.default},
    )
    lib_name = f"libQnn{op_package_config.op_package_name}"

    if args.build_op_package:
        _run(["rm", "-rf", "build"], cwd=args.op_package_dir)
        _run(["make", f"htp_v{arch}"], cwd=args.op_package_dir)
        # Rename to _HTP.so convention expected by QNN backend
        _run(
            [
                "cp",
                f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}.so",
                f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}_HTP.so",
            ]
        )

    # On-device path for the op package .so
    # This is what QNN runtime will dlopen on the device
    op_package_device_path = f"{workspace}/{lib_name}_HTP.so"

    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.HTP,
        platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
        op_package_path=op_package_device_path,
    )
    op_package_options = op_package_config.get_op_package_options()

    # ── Files to push to device ───────────────────────────────────────────────
    # Both .so files must land in workspace so the dlopen in
    # AttentionOpPackageInterface.cpp (which uses resolveKernelPath()) finds
    # libattention_fwd_kernel.so next to libQnnAttentionOpPackage_HTP.so
    op_package_paths = [
        # The QNN op package wrapper (registered above)
        os.path.abspath(
            f"{args.op_package_dir}/build/hexagon-v{arch}/{lib_name}_HTP.so"
        ),
        # The hexagon-mlir compiled Triton kernel — must be in the same
        # directory as the op package so dladdr-based resolveKernelPath() works
        os.path.abspath(
            f"{args.op_package_dir}/build/hexagon-v{arch}/libattention_fwd_kernel.so"
        ),
    ]
    # ─────────────────────────────────────────────────────────────────────────

    # ── Quantization ─────────────────────────────────────────────────────────
    quant_dtype = QuantDtype.use_8a8w
    if args.use_fp16:
        quantizer = None
    else:
        quant_cfg = get_ptq_per_channel_quant_config()
        custom_quant_annotator = CustomOpsQuantAnnotator()
        # Register annotation for YOUR attention op, not mul3
        custom_quant_annotator.register_annotation(
            torch.ops.my_ns.attention_fwd.default,
            IOQuantConfig(
                input_quant_specs={
                    0: quant_cfg.input_activation,  # Q
                    1: quant_cfg.input_activation,  # K
                    2: quant_cfg.input_activation,  # V
                },
                output_quant_specs={
                    0: quant_cfg.output_activation,
                },
            ),
        )
        custom_quant_annotator.register_annotation(
            torch.ops.my_ns.attention_fwd.out,
            IOQuantConfig(
                input_quant_specs={
                    0: quant_cfg.input_activation,  # Q
                    1: quant_cfg.input_activation,  # K
                    2: quant_cfg.input_activation,  # V
                },
                output_quant_specs={
                    0: quant_cfg.output_activation,
                },
            ),
        )
        annotate_fn = custom_quant_annotator.build_annotation_fn()
        quantizer = make_quantizer(
            quant_dtype=quant_dtype,
            custom_annotations=(annotate_fn,),
            backend=get_backend_type(args.backend),
            soc_model=args.soc_model,
        )
    # ─────────────────────────────────────────────────────────────────────────

    build_executorch_binary(
        model=instance,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=[sample_inputs],  # list of input tuples for calibration
        op_package_options=op_package_options,
        quant_dtype=quant_dtype,
        custom_quantizer=quantizer,
    )

    # ── Output collection ─────────────────────────────────────────────────────
    output_data_folder = f"{args.artifact}/outputs"
    make_output_dir(output_data_folder)

    if args.enable_x86_64:
        input_list_filename = "input_list.txt"
        generate_inputs(args.artifact, input_list_filename, sample_inputs)
        qnn_sdk = os.getenv("QNN_SDK_ROOT")
        assert qnn_sdk, "QNN_SDK_ROOT was not found in environment variable"
        target = "x86_64-linux-clang"
        build_folder = os.path.abspath(args.build_folder)
        artifact = os.path.abspath(args.artifact)

        runner_cmd = " ".join(
            [
                f"export LD_LIBRARY_PATH={qnn_sdk}/lib/{target}/:{build_folder}/lib &&",
                f"{build_folder}/examples/qualcomm/executor_runner/qnn_executor_runner",
                f"--model_path {artifact}/{pte_filename}.pte",
                f"--input_list_path {artifact}/{input_list_filename}",
                f"--output_folder_path {artifact}/outputs",
            ]
        )
        subprocess.run(
            runner_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=artifact,
        )
    else:
        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=f"{args.artifact}/{pte_filename}.pte",
            workspace=workspace,
        )
        # push inputs + both .so files to workspace on device
        adb.push(inputs=sample_inputs, files=op_package_paths)

        if args.debug:
            adb.execute(custom_runner_cmd="logcat -c")
            adb.execute(
                custom_runner_cmd=f"echo 0x1f > {workspace}/qnn_executor_runner.farf"
            )

        adb.execute()

        if args.debug:
            adb.execute(
                custom_runner_cmd=(
                    f"logcat -d -v time >{workspace}/outputs/debug_logs.txt"
                )
            )
        adb.pull(host_output_path=args.artifact)
    # ─────────────────────────────────────────────────────────────────────────

    # ── Verify output ─────────────────────────────────────────────────────────
    x86_golden = instance(*sample_inputs)
    device_output = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"), dtype=np.float32
        )
    ).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-2)).tolist()

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(json.dumps({"is_close": result}))
    else:
        print(f"is_close? {result}")
        if not result:
            print(f"x86_golden {x86_golden}")
            print(f"device_out {device_output}")
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. Default ./custom_op",
        default="./custom_op",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--op_package_dir",
        help="Path to operator package directory (contains config/ and build/).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-F",
        "--use_fp16",
        help="If specified, will run in fp16 precision and discard ptq setting.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--build_op_package",
        help=(
            "Build op package based on op_package_dir. "
            "Requires HEXAGON_SDK_ROOT and ANDROID_NDK_ROOT to be set, "
            "and clang in PATH."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        help="Enable device logging via logcat.",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if args.ip and args.port != -1:
            with Client((args.ip, args.port)) as conn:
                conn.send(json.dumps({"Error": str(e)}))
        else:
            raise

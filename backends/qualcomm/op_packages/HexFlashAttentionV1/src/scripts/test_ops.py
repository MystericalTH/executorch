import json
import os
import subprocess
import sys
from multiprocessing.connection import Client

import numpy as np
import torch

from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.export_utils import (
    build_executorch_binary,
    generate_inputs,
    QnnConfig,
    setup_common_args_and_variables,
    SimpleADB,
)

from executorch.backends.qualcomm.op_packages.HexFlashAttentionV1.src.tests.test_lib_definition import (
    get_test_op_module,
)

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from executorch.examples.qualcomm.utils import make_output_dir


class BaseTestModel(torch.nn.Module):
    op_id: int

    def __init__(self, op_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_id = op_id

    def forward(self, input_0: torch.Tensor):
        return torch.ops.hex_flash_test.test_op.default(input_0, self.op_id)


class Base2InputTestModel(torch.nn.Module):
    op_id: int

    def __init__(self, op_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_id = op_id

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        return torch.ops.hex_flash_test.test_op_2_input.default(
            input_0, input_1, self.op_id
        )


def get_test_op_package_config(android_workspace: str = None):
    op_package_dir = "backends/qualcomm/op_packages/HexFlashAttentionV1"
    xml_path = f"{op_package_dir}/config/HexFlashAttentionV1.xml"

    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={
            "TestOp": torch.ops.hex_flash_test.test_op.default,
            "TestOp2Input": torch.ops.hex_flash_test.test_op_2_input.default,
            "HexFlashAttentionQMerge": torch.ops.hex_flash_test.hfaq_merge.default,
            "HexFlashAttentionQLocal": torch.ops.hex_flash_test.hfaq_local.default,
            "HexFlashAttentionQ": torch.ops.hex_flash_test.hfaq.default,
        },
    )

    lib_name = f"libQnn{op_package_config.op_package_name}"

    if android_workspace:
        op_package_config.register_implementation(
            target=QnnExecuTorchOpPackageTarget.HTP,
            platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
            op_package_path=f"{android_workspace}/{lib_name}_HTP.so",
        )
        op_package_config.register_implementation(
            target=QnnExecuTorchOpPackageTarget.CPU,
            platform=QnnExecuTorchOpPackagePlatform.AARCH64_ANDROID,
            op_package_path=f"{android_workspace}/{lib_name}.so",
        )

    op_package_config.register_implementation(
        target=QnnExecuTorchOpPackageTarget.CPU,
        platform=QnnExecuTorchOpPackagePlatform.X86_64,
        op_package_path=os.path.abspath(
            f"{op_package_dir}/build/x86_64-linux-clang/{lib_name}.so"
        ),
    )

    return op_package_config


def build_op_package(lib_name: str, args):
    make_list = ["make", "htp_x86"]

    if not args.enable_x86_64:
        make_list.append(["htp_aarch64", f"htp_v{args.arch}"])

    _run(["rm", "-rf", "build"], cwd=args.op_package_dir)
    _run(make_list, cwd=args.op_package_dir)

    if not args.enable_x86_64:
        _run(
            [
                "cp",
                f"{args.op_package_dir}/build/hexagon-v{args.arch}/{lib_name}.so",
                f"{args.op_package_dir}/build/hexagon-v{args.arch}/{lib_name}_HTP.so",
            ]
        )


def get_model_instance(test_module):
    try:
        instance = test_module.TestModel()
        print("[Test] found and using custom test model")
    except:
        print("[Test] using base test model")
        if test_module.INPUT_NUM == 2:
            instance = Base2InputTestModel(test_module.TEST_OP_ID)
        else:
            instance = BaseTestModel(test_module.TEST_OP_ID)
    return instance


def _run(cmd, cwd=None):
    subprocess.run(cmd, stdout=sys.stdout, cwd=cwd, check=True)


def main(args):
    qnn_config = QnnConfig.load_config(args.config_file if args.config_file else args)

    use_android_device = not args.enable_x86_64

    if args.build_op_package:
        if "HEXAGON_SDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable HEXAGON_SDK_ROOT must be set")
        print(f"HEXAGON_SDK_ROOT={os.getenv('HEXAGON_SDK_ROOT')}")

        if "ANDROID_NDK_ROOT" not in os.environ:
            raise RuntimeError("Environment variable ANDROID_NDK_ROOT must be set")
        print(f"ANDROID_NDK_ROOT={os.getenv('ANDROID_NDK_ROOT')}")

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    test_module = get_test_op_module(int(args.op_id))
    print("[Test] Testing module:", test_module.__name__)

    instance = get_model_instance(test_module)

    pte_filename = "hex_flash_attention_op_" + args.op_id
    workspace = (
        f"/data/local/tmp/executorch/{pte_filename}" if use_android_device else None
    )

    # op package setup
    op_package_config = get_test_op_package_config(workspace)
    lib_name = f"libQnn{op_package_config.op_package_name}"

    if args.build_op_package:
        build_op_package(lib_name, args)

    op_package_options = op_package_config.get_op_package_options()
    op_package_paths = [
        f"{args.op_package_dir}/build/hexagon-v{args.arch}/{lib_name}_HTP.so",
        f"{args.op_package_dir}/build/aarch64-android/{lib_name}.so",
    ]

    sample_inputs = test_module.generate_sample_inputs()

    build_executorch_binary(
        model=instance,
        qnn_config=qnn_config,
        file_name=f"{args.artifact}/{pte_filename}",
        dataset=sample_inputs,
        op_package_options=op_package_options,
    )

    # collect output data
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
        # setup required params accordingly
        # qnn_config    : QnnConfig that saves config info
        # device_id     : serial number of android device
        # workspace     : folder for storing artifacts on android device
        adb = SimpleADB(
            qnn_config=qnn_config,
            pte_path=f"{args.artifact}/{pte_filename}.pte",
            workspace=workspace,
        )
        adb.push(inputs=sample_inputs, files=op_package_paths)
        if args.debug:
            adb.execute(custom_runner_cmd="logcat -c")
            adb.execute(
                custom_runner_cmd=f"echo 0x1f > {workspace}/qnn_executor_runner.farf"
            )

        adb.execute()
        if args.debug:
            adb.execute(
                custom_runner_cmd=f"logcat -d -v time >{workspace}/outputs/debug_logs.txt"
            )
        adb.pull(host_output_path=args.artifact)

    x86_golden = instance(*sample_inputs[0])

    torch_to_numpy = {torch.float16: np.float16, torch.float32: np.float32}

    device_output = torch.from_numpy(
        np.fromfile(
            os.path.join(output_data_folder, "output_0_0.raw"),
            dtype=torch_to_numpy[test_module.DTYPE],
        ),
    ).reshape(x86_golden.size())
    result = torch.all(torch.isclose(x86_golden, device_output, atol=1e-4)).tolist()

    glob_pct_error = (
        torch.sum(torch.abs(device_output - x86_golden).to(torch.float64))
        / torch.sum(torch.abs(x86_golden).to(torch.float64))
    ).item() * 100

    if args.ip and args.port != -1:
        with Client((args.ip, args.port)) as conn:
            conn.send(
                json.dumps(
                    {
                        "is_close": result,
                    }
                )
            )
    else:
        print(f"is_close? {result}")
        if not result:
            print(f"x86_golden {x86_golden.shape}\n{x86_golden}")
            print(f"device_out {device_output.shape}\n{device_output}")

    print(f"({args.op_id}) Test Module Name:", test_module.__name__)

    print("Max Abs Error:", torch.abs(device_output - x86_golden).max().item())
    print("Mean Abs Error", torch.abs(device_output - x86_golden).mean().item())
    print("Abs Pct Error:", glob_pct_error, "%")
    print("No. of nan:", torch.isnan(device_output).sum().item())


if __name__ == "__main__":
    parser = setup_common_args_and_variables()

    parser.add_argument(
        "--arch",
        help="hexagon architecture version",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--op_id",
        help="Op ID to test",
        required=True,
    )

    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts.",
        default="backends/qualcomm/op_packages/HexFlashAttentionV1/artifact",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--op_package_dir",
        help="Path to operator package generated from QNN.",
        default="backends/qualcomm/op_packages/HexFlashAttentionV1",
        type=str,
    )

    parser.add_argument(
        "--build_op_package",
        help="Build op package based on op_package_dir. Please set up "
        "`HEXAGON_SDK_ROOT` and `ANDROID_NDK_ROOT` environment variable. "
        "And add clang compiler into `PATH`. Please refer to Qualcomm AI Engine "
        "Direct SDK document to get more details",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--debug",
        help="Enable device logging",
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
            raise Exception(e)

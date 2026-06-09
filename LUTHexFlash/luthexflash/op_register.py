import os

import torch
from executorch.backends.qualcomm.custom_op.interface import QnnCustomOpPackageBuilder
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)
from torch.library import Library, impl

from luthexflash.utils.weights import unpack_weights

tman_lib = Library("tman", "DEF")

# TMAN Linear

tman_lib.define(
    "linear(Tensor x, Tensor qweight, Tensor scales, Tensor qzeros, Tensor g_idx, "
    "Tensor wf_unsqueeze_zero, Tensor wf_unsqueeze_neg_one, int group_size, int bits, "
    "bool symmetric, bool gptq_v2) -> Tensor"
)

tman_lib.define(
    "linear.out(Tensor x, Tensor qweight, Tensor scales, Tensor qzeros, Tensor g_idx, "
    "Tensor wf_unsqueeze_zero, Tensor wf_unsqueeze_neg_one, int group_size, int bits, "
    "bool symmetric, bool gptq_v2, *, Tensor(a!) out) -> Tensor(a!)"
)

tman_lib.define(
    "linear.meta(Tensor x, Tensor qweight, Tensor scales, Tensor qzeros, Tensor g_idx, "
    "Tensor wf_unsqueeze_zero, Tensor wf_unsqueeze_neg_one, int group_size, int bits, "
    "bool symmetric, bool gptq_v2) -> Tensor"
)

# Bitnet Linear

tman_lib.define("bitnet_linear(Tensor x, Tensor weight, Tensor weight_scale) -> Tensor")

tman_lib.define(
    "bitnet_linear.out(Tensor x, Tensor weight, Tensor weight_scale, *, Tensor(a!) out) -> Tensor(a!)"
)

tman_lib.define(
    "bitnet_linear.meta(Tensor x, Tensor weight, Tensor weight_scale) -> Tensor"
)

# ─────────────────────────────────────────────────────────────────────────────


# Helper for weight recovery (kept identical to your math template)
def _dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Based on dequantize_weights in gptqmodel/nn_modules/qlinear/__init__.py
    """
    import torch as t

    num_itr = 1  # desc_act=False
    assert qweight.dtype == t.int32 and qzeros.dtype == t.int32
    pack_factor = 32 // bits
    dequant_dtype = t.int16 if bits == 8 else t.int8
    maxq = 2**bits - 1

    if bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(qzeros, 2).expand(-1, -1, pack_factor),
            wf_unsqueeze_zero,  # wf.unsqueeze(0),
        ).to(dequant_dtype)
        zeros = t.bitwise_and(zeros, maxq).reshape(scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
                wf_unsqueeze_neg_one,  # wf.unsqueeze(-1)
            ).to(dequant_dtype),
            maxq,
        )
    elif bits == 3:
        zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> wf_unsqueeze_zero  # wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(scales.shape)

        weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> wf_unsqueeze_neg_one) & 0x7  # wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if num_itr == 1:
        weights = scales[g_idx.long()] * (weight - zeros[g_idx.long()])
    else:
        num_dim = g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = scales[:, i * num_dim : (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
            g_idx_i = g_idx[i * num_dim : (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights = t.cat(weights, dim=1)

    return weights


# Implementation: Functional variant
@impl(tman_lib, "linear", dispatch_key="CompositeExplicitAutograd")
def tman_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)

    # Materialize full weight layout using your unpacking logic
    weights = _dequantize_weight(
        qweight, scales, qzeros, g_idx, wf_unsqueeze_zero, wf_unsqueeze_neg_one, bits
    ).to(x.dtype)

    # Compute dot products
    x_flat = x.reshape(-1, x.shape[-1])
    out = torch.matmul(x_flat, weights).reshape(out_shape)
    return out.to(x.dtype)


# Implementation: Out variant (Crucial for memory allocator backends)
@impl(tman_lib, "linear.out", dispatch_key="CompositeExplicitAutograd")
def tman_linear_out_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    # Compute using standard logic block
    res = tman_linear(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        group_size,
        bits,
        symmetric,
        gptq_v2,
    )
    # Safely move data into the pre-allocated buffer memory pool
    out.copy_(res)
    return out


# Implementation: Meta variant (Ensures compilation doesn't require actual data execution)
@impl(tman_lib, "linear.meta", dispatch_key="Meta")
def tman_linear_meta(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    # Return an abstract tensor shell to let tracing calculate shapes/strides
    return torch.empty(out_shape, device="meta", dtype=x.dtype)


# ─────────────────────────────────────────────────────────────────────────────


# Implementation: Functional variant
@impl(tman_lib, "bitnet_linear", dispatch_key="CompositeExplicitAutograd")
def tman_bitnet_linear_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # Unpack binarized/ternarized weights
    w_quant = unpack_weights(weight, dtype=x.dtype)

    # Activation quantization (per-token INT8 scaling)
    num_bits = 8
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1

    scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * scale).round().clamp(Qn, Qp)
    input_quant, input_scale = result.to(torch.int8), scale

    # Compute the core linear projection dot products
    y = torch.nn.functional.linear(input_quant.to(x.dtype), w_quant)

    # Rescale back to original precision layer boundaries
    y = y / input_scale * weight_scale
    return y


@impl(tman_lib, "bitnet_linear.out", dispatch_key="CompositeExplicitAutograd")
def tman_bitnet_linear_out_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    res = tman_bitnet_linear_impl(x, weight, weight_scale)
    out.copy_(res)
    return out


@impl(tman_lib, "bitnet_linear.meta", dispatch_key="Meta")
def tman_bitnet_linear_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # BitNet weights are packed, so out_features typically tracks weight.shape[0]
    # or a modified dimension depending on your pack layout.
    # Matching your template functional output configuration:
    out_features = weight.shape[0]
    out_shape = x.shape[:-1] + (out_features,)

    return torch.empty(out_shape, device="meta", dtype=x.dtype)


def register_op(op_package_dir, workspace, xml_path=None):
    if xml_path is None:
        _xml_path_dir = os.path.join(op_package_dir, "config")

        files = [
            os.path.join(_xml_path_dir, f)
            for f in os.listdir(_xml_path_dir)
            if os.path.isfile(os.path.join(_xml_path_dir, f))
        ]

        if files:
            xml_path = files[0]
        else:
            raise FileNotFoundError(f"No configuration files found in {_xml_path_dir}")

    op_package_config = QnnCustomOpPackageBuilder(
        xml_path=xml_path,
        torch_op_name_map={
            "TMANLinear": torch.ops.tman.linear.default,
            # "TMANPrecompute": torch.ops.tman.precompute.default,
        },
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
    op_package_options = op_package_config.get_op_package_options()
    return op_package_options, lib_name

import torch

# from executorch.backends.qualcomm.quantizer.annotator import register_annotator
from torch.fx import Node

from executorch.backends.qualcomm.quantizer.quantizer import (
    QuantizationConfig,
)
from executorch.backends.qualcomm.quantizer.rules import (
    _is_annotated,
    annotate_in_out_obs_sharing_op,
    annotate_single_in_single_out,
)


def annotate_split_with_sizes(
    node: Node, quantization_config: QuantizationConfig
) -> None:
    annotate_in_out_obs_sharing_op(node, quantization_config)
    if not _is_annotated([node]):
        annotate_single_in_single_out(node, quantization_config)


def annotate_tman_linear(node: Node, quantization_config: QuantizationConfig) -> None:
    if _is_annotated([node]):
        return
    annotate_single_in_single_out(node, quantization_config)

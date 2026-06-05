import warnings
from typing import Dict

import executorch.backends.qualcomm.python.PyQnnManagerAdaptor as PyQnnManager
import numpy as np

import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA

from .node_visitor import NodeVisitor
from .node_visitor_manager import register_node_visitor
from .qnn_constants import OpFlashAttention, QNN_OP_PACKAGE_NAME_HEX_FLASH


@register_node_visitor
class FlashAttention(NodeVisitor):
    target = ["aten.scaled_dot_product_attention.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnManager.TensorWrapper],
    ) -> PyQnnManager.PyQnnOpWrapper:
        if len(node.args) > 4:
            warnings.warn(
                "[QNN Delegate Op Builder]: FlashAttention currently does not support dropout_p, causal, enable_gqa",
                stacklevel=1,
            )
            return

        query_node = self.get_node(node.args[0])
        query_tensor = self.get_tensor(query_node, node)
        query_tensor_wrapper = self.define_tensor(
            query_node,
            node,
            query_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        key_node = self.get_node(node.args[1])
        key_tensor = self.get_tensor(key_node, node)
        key_tensor_wrapper = self.define_tensor(
            key_node,
            node,
            key_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        value_node = self.get_node(node.args[2])
        value_tensor = self.get_tensor(value_node, node)
        value_tensor_wrapper = self.define_tensor(
            value_node,
            node,
            value_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        attn_mask_node = self.get_node(node.args[3])
        attn_mask_tensor = self.get_tensor(attn_mask_node, node)
        attn_mask_tensor = self.define_tensor(
            attn_mask_node,
            node,
            attn_mask_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnManager.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        scale = node.args[6] if len(node.args) > 6 else node.kwargs.get("scale")

        flash_attention_op = PyQnnManager.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_HEX_FLASH,
            OpFlashAttention.op_name,
        )
        flash_attention_op.AddInputTensors(
            [
                query_tensor_wrapper,
                key_tensor_wrapper,
                value_tensor_wrapper,
                attn_mask_tensor,
            ]
        )
        flash_attention_op.AddOutputTensors([output_tensor_wrapper])
        flash_attention_op.AddScalarParam(
            OpFlashAttention.param_scale,
            PyQnnManager.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            {QCOM_DATA: np.float32(scale) if scale else 0},
        )
        return flash_attention_op

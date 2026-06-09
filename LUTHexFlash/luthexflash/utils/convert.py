import torch
from luthexflash.modules.tman import TMANLinear


def convert_linear_to_qlinear(module: torch.nn.Module, qlinear_cls):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    def replace_linear(module: torch.nn.Module):
        attr_strs = dir(module)
        if isinstance(module, torch.nn.ModuleList):
            attr_strs += [str(i) for i in range(len(module))]

        for attr_str in attr_strs:
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.Linear):
                qlinear = qlinear_cls(
                    in_features=target_attr.in_features,
                    out_features=target_attr.out_features,
                    bias=target_attr.bias is not None,
                )
                # The model should have been converted to gptq_v2 in convert_gptq_weights_to_llama.py
                qlinear.qzero_format(2)
                assert isinstance(qlinear, TorchLinear)
                setattr(module, attr_str, qlinear)

        for _, sub_module in module.named_children():
            sub_module = replace_linear(sub_module)
        return module

    return replace_linear(module)


def convert_qlinear_to_linear(module: torch.nn.Module):
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    def replace_qlinear(module: torch.nn.Module):
        attr_strs = dir(module)
        if isinstance(module, torch.nn.ModuleList):
            attr_strs += [str(i) for i in range(len(module))]

        for attr_str in attr_strs:
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, BaseQuantLinear):
                if not isinstance(target_attr, TorchLinear):
                    raise RuntimeError("Only GPTQ TorchLinear backend is supported")
                target_attr.post_init()
                new_attr = torch.nn.Linear(
                    target_attr.in_features, target_attr.out_features
                )
                new_attr.weight = torch.nn.Parameter(
                    target_attr.dequantize_weight().T.detach().to("cpu", torch.float16)
                )
                new_attr.bias = (
                    torch.nn.Parameter(target_attr.bias)
                    if target_attr.bias is not None
                    else None
                )
                setattr(module, attr_str, new_attr)

        for _, sub_module in module.named_children():
            sub_module = replace_qlinear(sub_module)
        return module

    return replace_qlinear(module)


def convert_qlinear_to_tman_linear(module: torch.nn.Module):
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    def replace_qlinear(module: torch.nn.Module):
        attr_strs = dir(module)
        if isinstance(module, torch.nn.ModuleList):
            attr_strs += [str(i) for i in range(len(module))]

        for attr_str in attr_strs:
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, BaseQuantLinear):
                if not isinstance(target_attr, TorchLinear):
                    raise RuntimeError("Only GPTQ TorchLinear backend is supported")
                target_attr.post_init()
                setattr(module, attr_str, TMANLinear(target_attr))

        for _, sub_module in module.named_children():
            sub_module = replace_qlinear(sub_module)
        return module

    return replace_qlinear(module)

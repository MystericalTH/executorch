import torch

from luthexflash.op_register import tman_linear
from luthexflash.utils.weights import unpack_gptqv2


class TMANLinear(torch.nn.Module):
    def __init__(self, qlinear: torch.nn.Module, n_splits: int = 1):
        super().__init__()
        # GPTQv1: AutoGPTQ
        # GPTQv2: GPTQModel
        self.gptq_v2 = qlinear.qzero_format() == 2
        self.in_features = qlinear.in_features
        self.out_features = qlinear.out_features // n_splits

        _, _, _, bits, group_size, symmetric = unpack_gptqv2(
            qlinear.qweight.detach().numpy(),
            qlinear.scales.detach().numpy(),
            qlinear.qzeros.detach().numpy(),
            self.gptq_v2,
        )

        if n_splits == 1:
            self.qweight = torch.nn.Parameter(qlinear.qweight, requires_grad=False)
            self.scales = torch.nn.Parameter(qlinear.scales, requires_grad=False)
            self.qzeros = torch.nn.Parameter(qlinear.qzeros, requires_grad=False)
        else:
            self.qweight = torch.nn.Parameter(
                qlinear.qweight.new_zeros(
                    (qlinear.qweight.shape[0], qlinear.qweight.shape[1] // n_splits)
                ),
                requires_grad=False,
            )
            self.scales = torch.nn.Parameter(
                qlinear.scales.new_zeros(
                    (qlinear.scales.shape[0], qlinear.scales.shape[1] // n_splits)
                ),
                requires_grad=False,
            )
            self.qzeros = torch.nn.Parameter(
                qlinear.qzeros.new_zeros(
                    (qlinear.qzeros.shape[0], qlinear.qzeros.shape[1] // n_splits)
                ),
                requires_grad=False,
            )

        self.g_idx = torch.nn.Parameter(qlinear.g_idx, requires_grad=False)
        self.wf_unsqueeze_zero = torch.nn.Parameter(
            qlinear.wf_unsqueeze_zero, requires_grad=False
        )
        self.wf_unsqueeze_neg_one = torch.nn.Parameter(
            qlinear.wf_unsqueeze_neg_one, requires_grad=False
        )

        self.group_size = group_size
        self.bits = bits
        self.symmetric = symmetric

    def forward(self, x):
        return tman_linear(
            x,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.wf_unsqueeze_zero,
            self.wf_unsqueeze_neg_one,
            self.group_size,
            self.bits,
            self.symmetric,
            self.gptq_v2,
        )

    def extra_repr(self):
        s = (
            "{in_features}, {out_features}, group_size={group_size}, bits={bits}"
            ", symmetric={symmetric}"
        )
        return s.format(**self.__dict__)

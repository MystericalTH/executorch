from dataclasses import dataclass

QNN_OP_PACKAGE_NAME_TMAN = "TMANOpPackage"
QNN_OP_PACKAGE_NAME_QTI_AISW = "qti.aisw"


@dataclass(init=False, frozen=True)
class OpConvert:
    op_name: str = "Convert"


@dataclass(init=False, frozen=True)
class OpTMANLinear:
    op_name: str = "TMANLinear"
    param_group_size: str = "group_size"
    param_bits: str = "bits"
    param_symmetric: str = "symmetric"


@dataclass(init=False, frozen=True)
class OpTMANPrecompute:
    op_name: str = "TMANPrecompute"
    param_group_size: str = "group_size"
    param_bits: str = "bits"
    param_symmetric: str = "symmetric"


@dataclass(init=False, frozen=True)
class OpTMANFinalize:
    op_name: str = "TMANFinalize"
    param_group_size: str = "group_size"
    param_bits: str = "bits"
    param_symmetric: str = "symmetric"

from typing import Union, Tuple

from odl.space.base_tensors import Tensor

__all__ = ('astype', 'can_cast', 'finfo', 'iinfo', 'isdtype', 'result_type')

def astype(x: Tensor, dtype: Tensor.dtype) -> Tensor:
    """Copies an array to a specified data type irrespective of Type Promotion
    Rules rules."""
    raise NotImplementedError


def can_cast(from_: Union[Tensor.dtype, Tensor], to: Tensor.dtype) -> bool:
    """Determines if one data type can be cast to another data type according
    Type Promotion Rules rules."""
    raise NotImplementedError


def finfo(type: Union[Tensor.dtype, Tensor]) -> bool:
    """Machine limits for floating-point data types."""
    return type.array_namespace.finfo(type)


def iinfo(type: Union[Tensor.dtype, Tensor]) -> bool:
    """Machine limits for integer data types."""
    raise NotImplementedError


def isdtype(
    dtype: Tensor.dtype, kind: Union[str, Tensor.dtype, Tuple[Union[str, Tensor.dtype]]]
) -> bool:
    """Returns a boolean indicating whether a provided dtype is of a specified
    data type "kind"."""
    raise NotImplementedError


def result_type(*arrays_and_dtypes: Union[Tensor, Tensor.dtype]) -> Tensor.dtype:
    """Returns the dtype that results from applying the type promotion rules
    (see Type Promotion Rules) to the arguments."""
    raise NotImplementedError

from typing import Union

__all__ = ('astype', 'can_cast', 'finfo', 'iinfo', 'isdtype', 'result_type')

def astype(x, dtype):
    """Copies an array to a specified data type irrespective of Type Promotion
    Rules rules."""
    raise NotImplementedError


def can_cast(from_, to) -> bool:
    """Determines if one data type can be cast to another data type according
    Type Promotion Rules rules."""
    raise NotImplementedError


def finfo(type) -> bool:
    """Machine limits for floating-point data types."""
    return type.array_namespace.finfo(type)


def iinfo(type) -> bool:
    """Machine limits for integer data types."""
    raise NotImplementedError


def isdtype(dtype, kind) -> bool:
    """Returns a boolean indicating whether a provided dtype is of a specified
    data type "kind"."""
    raise NotImplementedError


def result_type(*arrays_and_dtypes):
    """Returns the dtype that results from applying the type promotion rules
    (see Type Promotion Rules) to the arguments."""
    raise NotImplementedError

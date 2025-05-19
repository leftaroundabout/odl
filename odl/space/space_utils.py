# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utility functions for space implementations."""

from __future__ import print_function, division, absolute_import

from odl.util.utility import AVAILABLE_DTYPES
from odl.space.base_tensors import TensorSpace
from odl.space.npy_tensors import NumpyTensorSpace
from odl.space.pytorch_tensors import PytorchTensorSpace

TENSOR_SPACE_IMPLS = {
    'numpy': NumpyTensorSpace,
    'pytorch': PytorchTensorSpace
    }

__all__ = ("tensor_space", )

def tensor_space(
    shape, dtype="float32", impl="numpy", device="cpu", **kwargs
) -> TensorSpace:
    """Return a tensor space with arbitrary scalar data type.

    Parameters
    ----------
    shape : positive int or sequence of positive ints
        Number of entries per axis for elements in this space. A
        single integer results in a space with 1 axis.
    dtype : str, optional
        Data type of each element. Must be provided as a string. (cf 
        AVAILABLE_DTYPES in odl/util/utility.py)
        For ``None``, the `TensorSpace.default_dtype` of the
        created space is used.
    impl : str, optional
        Implementation back-end for the space. See
        `odl.space.entry_points.tensor_space_impl_names` for available
        options.
    device: str, optional
        Implementation of the device for the space. Defaults to 'cpu',
        but can be 'cuda:i' for pytorch tensors on the GPU.
    kwargs :
        Extra keyword arguments passed to the space constructor.

    Returns
    -------
    space : `TensorSpace`

    Examples
    --------
    Space of 3-tuples with ``uint64`` entries (although not strictly a
    vector space):

    >>> odl.tensor_space(3, dtype='uint64')
    tensor_space(3, dtype='uint64')

    2x3 tensors with same data type:

    >>> odl.tensor_space((2, 3), dtype='uint64')
    tensor_space((2, 3), dtype='uint64')

    The default data type depends on the implementation. For
    ``impl='numpy'``, it is ``'float32'``:

    >>> ts = odl.tensor_space((2, 3))
    >>> ts
    rn((2, 3))
    >>> ts.dtype
    dtype('float64')

    See Also
    --------
    rn, cn : Constructors for real and complex spaces
    """
    # Check the dtype argument
    assert (
        dtype in AVAILABLE_DTYPES
    ), f"The dtype must be in {AVAILABLE_DTYPES}, but {dtype} was provided"
    # Check the impl argument
    assert (
        impl in TENSOR_SPACE_IMPLS.keys()
    ), f"The only supported impls are {TENSOR_SPACE_IMPLS.keys()}, but {impl} was provided"

    # Use args by keyword since the constructor may take other arguments
    # by position
    return TENSOR_SPACE_IMPLS[impl](shape=shape, dtype=dtype, device=device, **kwargs)

if __name__ == "__main__":
    from odl.util.testutils import run_doctests

    run_doctests()

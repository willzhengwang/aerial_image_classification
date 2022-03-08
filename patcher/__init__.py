#!/usr/bin/env python

"""
Purpose: split a large (image) array into patches and merge small patches to a large array
Author: Zheng (Will) Wang
Website: https://willzhengwang.github.io/homepage/

MIT Licence
Copyright (c) 2022 Will Zheng WANG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided
import numbers


def make_patches(arr_in: np.ndarray, patch_size: Tuple, step=1,
                 do_pad: bool = False, mode: str = 'constant', **kwargs) -> np.ndarray:
    """
    Split an N-dimensional numpy array into small patches.

    Args:
        arr_in (np.ndarray): input array
        patch_size (Tuple): window shape
        step (int/Tuple, optional): the step size between patches. A single integer, or a tuple that has the same as patch_size. Defaults to 1.
        do_pad (bool, optional): padding option. Defaults to False.
        mode (str, optional): the mode used in numpy.pad(). Defaults to 'constant'.
        kwargs: dict. Any keyword arguments the numpy.pad() requires.
    """
    return view_as_windows(arr_in, patch_size, step, do_pad=do_pad, mode=mode, **kwargs)


def merge_patches(patches: np.ndarray, out_shape: Tuple = None, step: int = 1) -> np.ndarray:
    """
    Merge small patches into a large array.

    Args:
        patches (np.ndarray): _description_
        out_shape (Tuple): target out array shape.
                            Defaults to None: (n_patches - 1) * step + max(patch_size, step) along each axis
                            out_shape is larger than the default size: padding zeros to the end
                            out_shape is less than the default size: cutting downing
        step (int/Tuple, optional): the step size between patches.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """

    # -- basic checks on arguments

    ndim = len(patches.shape) // 2

    if out_shape is None:
        out_shape = (None,) * ndim
    elif len(out_shape) != ndim:
        raise ValueError("`patches` is incompatible with `out_shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim

    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `out_shape`")

    while np.any(np.array(list(patches.shape)[:ndim]) != 1):
        for axis in range(ndim):
            patches = merge_along_axis(patches, axis, step[axis])

    assert np.all(np.array(patches.shape)[:ndim] == 1)
    for _ in np.arange(ndim):
        patches = np.squeeze(patches, axis=0)

    # deal with None out_shape
    if np.any(np.array(out_shape) != None):
        # `patches` is the merged array
        for axis in range(ndim):
            if out_shape[axis] > patches.shape[axis]:
                patches = patches[dynamic_slicing(patches, axis, 0, out_shape[axis])]
                # add zeros to the end
                add_shape = list(patches.shape)
                add_shape[axis] = out_shape[axis] - patches.shape[axis]
                patches = np.concatenate((patches, np.zeros(tuple(add_shape), dtype=patches.dtype)), axis=axis)
            elif out_shape[axis] < patches.shape[axis]:
                patches = patches[dynamic_slicing(patches, axis + ndim, 0, out_shape[axis])]

    return patches


def view_as_windows(arr_in: np.ndarray, win_shape: Tuple, step=1, do_pad=False, mode='constant', **kwargs):
    """
    # The code is developed on top of https://github.com/scikit-image/scikit-image/blob/main/skimage/util/shape.py

    # Copyright (C) 2011, the scikit-image team All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without modification,
    # are permitted provided that the following conditions are met:
    #
    # Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    # Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    # in the documentation and/or other materials provided with the distribution.
    # Neither the name of skimage nor the names of its contributors may be used to endorse or promote products derived from this software
    # without specific prior written permission.
    # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
    # THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Args:
        arr_in (np.ndarray): input array
        win_shape (Tuple): window shape
        step (int/Tuple, optional): the step size between patches. A single integer, or a tuple that has the same as patch_size. Defaults to 1.
        do_pad (bool, optional): padding option. Defaults to False.
        mode (str, optional): the mode used in numpy.pad(). Defaults to 'constant'.
        kwargs: dict. Any keyword arguments the numpy.pad() requires.

    Returns:
        _type_: _description_
    """

    # -- basic checks on arguments
    if not isinstance(arr_in, np.ndarray):
        raise TypeError("`arr_in` must be a numpy ndarray")

    ndim = arr_in.ndim

    if isinstance(win_shape, numbers.Number):
        win_shape = (win_shape,) * ndim
    if not (len(win_shape) == ndim):
        raise ValueError("`win_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim

    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    win_shape = np.array(win_shape, dtype=arr_shape.dtype)

    if ((arr_shape - win_shape) < 0).any():
        raise ValueError("`win_shape` is too large")

    if ((win_shape - 1) < 0).any():
        raise ValueError("`win_shape` is too small")

    win_indices_shape = (np.array(arr_in.shape) - np.max(np.array([np.array(win_shape), np.array(step)]),
                                                         axis=0)) // np.array(step) + 1

    if do_pad:
        win_remainders = np.remainder(
            (np.array(arr_in.shape) - np.max(np.array([np.array(win_shape), np.array(step)]), axis=0)), np.array(step))
        if np.any(win_remainders):
            npad = [(0, 0)] * ndim
            for i, remainder in enumerate(win_remainders):
                if remainder != 0:
                    npad[i] = (0, win_indices_shape[i] * step[i] + win_shape[i] - arr_in.shape[i])
                    win_indices_shape[i] += 1
            arr_in = np.pad(arr_in, pad_width=npad, mode=mode, **kwargs)

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    new_shape = tuple(list(win_indices_shape) + list(win_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def dynamic_slicing(arr: np.ndarray, axis: int, start: int = None, end: int = None):
    """
    Slice an array from the start to the end along a dynamic axis

    Args:
        arr (np.ndarray): _description_
        axis (int): _description_
        start (int, optional): _description_. Defaults to None.
        end (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if start is None or end is None:
        # get all indices along this axis
        start = 0
        end = arr.shape[axis]
    return (slice(None),) * (axis % arr.ndim) + (slice(start, end, 1),)


def merge_along_axis(patches: np.ndarray, axis: int, step: int) -> np.ndarray:
    """
    Unpatch/merge small patches along a specific axis.

    Assuming the final mosaic shape: [H, W, D]; the single patch shape: [p_h, p_w, p_d]; and the patch indices shape [s_h, s_w, s_d].
    This function will achieve the following:
    input patches: [s_h, s_w, s_d, p_h, p_w, p_d]; along the axis=0; --> output patches [1, s_w, s_d, H, p_w, p_d]
    input patches: [s_h, s_w, s_d, p_h, p_w, p_d]; along the axis=1; --> output patches [s_h, 1, s_d, p_h, W, p_d]

    Args:
        patches (np.ndarray): _description_
        axis (int): _description_
        step (int): _description_

    Returns:
        np.ndarray: patches merged along the specific axis
    """

    assert isinstance(step, numbers.Number), "`step` must be a single number"
    assert len(patches.shape) % 2 == 0, "the shape of `patches` is incorrect"
    ndim = len(patches.shape) // 2

    win_indices_shape = tuple(list(patches.shape)[:ndim])
    win_shape = tuple(list(patches.shape)[-ndim:])

    # along this axis
    n_patches = win_indices_shape[axis]

    if n_patches == 1:
        return patches

    win_size = win_shape[axis]
    ovl_size = win_shape[axis] - step
    mosaic_size = (n_patches - 1) * step + max(win_size, step)

    # mosaic results along this axis
    mosaic_shape = list(patches.shape)
    mosaic_shape[axis], mosaic_shape[axis + ndim] = 1, mosaic_size
    mosaic = np.zeros(tuple(mosaic_shape), dtype=patches.dtype)

    for j in range(n_patches):

        if ovl_size > 0:
            # for the unique section
            uni_section = (j * step + ovl_size, (j + 1) * step)
            uni_inds = dynamic_slicing(mosaic, axis + ndim, uni_section[0], uni_section[1])
            strip_patches = patches[dynamic_slicing(patches, axis, j, j + 1)]
            mosaic[uni_inds] = strip_patches[dynamic_slicing(strip_patches, axis + ndim, ovl_size, step)]

            # for the overlaps
            if j == 0:
                # the first item
                ovl_section = (0, ovl_size)
                ovl_inds = dynamic_slicing(mosaic, axis + ndim, ovl_section[0], ovl_section[1])
                mosaic[ovl_inds] = strip_patches[dynamic_slicing(strip_patches, axis + ndim, 0, ovl_size)]
            else:
                # intermediate items
                ovl_section = (j * step, j * step + ovl_size)
                ovl_inds = dynamic_slicing(mosaic, axis + ndim, ovl_section[0], ovl_section[1])
                strip_patches_pre = patches[dynamic_slicing(patches, axis, j - 1, j)]
                mosaic[ovl_inds] = np.mean(
                    np.array([strip_patches[dynamic_slicing(strip_patches, axis + ndim, 0, ovl_size)],
                              strip_patches_pre[
                                  dynamic_slicing(strip_patches_pre, axis + ndim, win_size - ovl_size, win_size)]]),
                    axis=0)
            if j == n_patches - 1:
                # the last item
                ovl_section = (mosaic_size - ovl_size, mosaic_size)
                ovl_inds = dynamic_slicing(mosaic, axis + ndim, ovl_section[0], ovl_section[1])
                mosaic[ovl_inds] = strip_patches[
                    dynamic_slicing(strip_patches, axis + ndim, win_size - ovl_size, win_size)]
        else:
            # for the unique section
            uni_section = (j * step, j * step + win_size)
            uni_inds = dynamic_slicing(mosaic, axis + ndim, uni_section[0], uni_section[1])
            strip_patches = patches[dynamic_slicing(patches, axis, j, j + 1)]
            mosaic[uni_inds] = strip_patches[dynamic_slicing(strip_patches, axis + ndim, 0, win_size)]

    return mosaic


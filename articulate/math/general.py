r"""
    General math utils.
"""


__all__ = ['lerp', 'normalize_tensor', 'clip_tensor_by_norm', 'refine_r6d', 'append_value',
    'append_zero', 'append_one', 'vector_cross_matrix', 'vector_cross_matrix_np',
    'block_diagonal_matrix_np', 'calc_acc_jitter', 'calc_jitter', 'resample']


import numpy as np
import torch
from functools import partial
from typing import Union
from scipy import interpolate as interp


def lerp(a, b, t):
    r"""
    Linear interpolation (unclamped).

    :param a: Begin value.
    :param b: End value.
    :param t: Lerp weight. t = 0 will return a; t = 1 will return b.
    :return: The linear interpolation value.
    """
    return a * (1 - t) + b * t


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def clip_tensor_by_norm(x:torch.Tensor, dim:int=-1,
    min:Union[float, torch.Tensor]=None, max:Union[float, torch.Tensor]=None):
    ''' Clip a tensor along a dimension if the norm of some part of it excceed the max value.
        The tensor direction will not change.
    '''
    res = x.clone()
    if max is not None:
        norm = res.norm(dim=dim, keepdim=True).clip(min=max)
        res = res * max / norm
    if min is not None:
        norm = res.norm(dim=dim, keepdim=True).clip(max=min)
        res = res * min / norm
    return res


def refine_r6d(r6d: torch.Tensor):
    ''' Find the closest r6d tensor to the given one.
        i.e. the two axes must be unit vector and must be perpendicular to each other.
    args:
        r6d: torch.Tensor, could be reshaped to [-1, 6].
    return:
        torch.Tensor, the refined r6d vector.
    '''
    r6d = r6d.view(-1, 6)
    a, b = normalize_tensor(r6d[:, :3]), normalize_tensor(r6d[:, 3:])
    x, y = normalize_tensor(b-a), normalize_tensor(a+b)
    a, b = (1/2**0.5) * (y-x), (1/2**0.5) * (x+y)
    return torch.cat([a, b], dim=1)


def append_value(x: torch.Tensor, value: float, dim=-1):
    r"""
    Append a value to a tensor in a specific dimension. (torch)

    e.g. append_value(torch.zeros(3, 3, 3), 1, dim=1) will result in a tensor of shape [3, 4, 3] where the extra
         part of the original tensor are all 1.

    :param x: Tensor in any shape.
    :param value: The value to be appended to the tensor.
    :param dim: The dimension to be expanded.
    :return: Tensor in the same shape except for the expanded dimension which is 1 larger.
    """
    app = torch.ones_like(x.index_select(dim, torch.tensor([0], device=x.device))) * value
    x = torch.cat((x, app), dim=dim)
    return x


append_zero = partial(append_value, value=0)
append_one = partial(append_value, value=1)


def vector_cross_matrix(x: torch.Tensor):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for each vector3 `v`. (torch, batch)

    :param x: Tensor that can reshape to [batch_size, 3].
    :return: The skew-symmetric matrix in shape [batch_size, 3, 3].
    """
    x = x.view(-1, 3)
    zeros = torch.zeros(x.shape[0], device=x.device)
    return torch.stack((zeros, -x[:, 2], x[:, 1],
                        x[:, 2], zeros, -x[:, 0],
                        -x[:, 1], x[:, 0], zeros), dim=1).view(-1, 3, 3)


def vector_cross_matrix_np(x):
    r"""
    Get the skew-symmetric matrix :math:`[v]_\times\in so(3)` for vector3 `v`. (numpy, single)

    :param x: Vector3 in shape [3].
    :return: The skew-symmetric matrix in shape [3, 3].
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]], dtype=float)


def block_diagonal_matrix_np(matrix2d_list):
    r"""
    Generate a block diagonal 2d matrix using a series of 2d matrices. (numpy, single)

    :param matrix2d_list: A list of matrices (2darray).
    :return: The block diagonal matrix.
    """
    ret = np.zeros(sum([np.array(m.shape) for m in matrix2d_list]))
    r, c = 0, 0
    for m in matrix2d_list:
        lr, lc = m.shape
        ret[r:r+lr, c:c+lc] = m
        r += lr
        c += lc
    return ret


def calc_acc_jitter(acc:torch.Tensor, fs:float=60) -> float:
    ''' Calculate the motion jitter of acceleration, in m/s^3.
    '''
    jitter = torch.mean(torch.abs(torch.diff(acc, dim=0)) * fs)
    return jitter.item()


def calc_jitter(data:torch.Tensor, dim:int=0, fs:float=60) -> float:
    ''' Calculate jitter of any data.
    '''
    # take third derivatives
    jitter = data.diff(n=3, dim=dim) * (fs**3)
    jitter = torch.mean(torch.abs(jitter))
    return jitter.item()


def resample(data:np.ndarray, axis:int, ratio:float, kind='quadratic') -> np.ndarray:
    ''' Resample data by spline interpolation.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the samping axis.
        ratio: float, the number of resampled data points over
            the number of original data points.
        kind: str, the interpolation time.
    returns:
        np.ndarray, the resampled data, dtype = the original dtype.
    '''
    N = data.shape[axis]
    M = int(np.round(N * ratio))
    t = np.arange(N, dtype=np.float64)
    x = np.linspace(0, N-1, num=M, endpoint=True)
    interp_func = interp.interp1d(t, data, kind=kind, axis=axis)
    return interp_func(x)


if __name__ == '__main__':
    acc = torch.randn(100, 3)
    print(calc_acc_jitter(acc))

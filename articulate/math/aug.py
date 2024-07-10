''' Traditional data augmentation algorithms.
'''


__all__ = ['jitter', 'scale_params', 'scale', 'zoom_params', 'zoom', 'magnitude_warp_params',
    'magnitude_warp', 'time_warp_params', 'time_warp']


import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import interpolate as interp


def jitter(data: torch.Tensor, std:float) -> torch.Tensor:
    ''' Added gaussian noise to data.
    args:
        data: torch.Tensor.
        std: float, the standard deviation of the Gaussion noise.
    '''
    return data + torch.randn_like(data) * std


def scale_params(std:float=0.01, low:float=0.0, high:float=2.0) -> float:
    ''' Generate random params that Scale needs.
    args: see scale().
    '''
    s = np.clip(1.0 + np.random.randn() * std, a_min=low, a_max=high)
    return s


def scale(data:torch.Tensor, std:float=0.01, low:float=0.0,
    high:float=2.0, params:float=None) -> np.ndarray:
    ''' Scale the data by a random factor s ~ N(1, std^2).
    args:
        data: torch.Tensor.
        std: float, the std deviation of the scaling factor.
        low and high: float, the lower and higher bounds of the random factor.
        params: float, if not None, use the params to augment data.
    '''
    if params is not None:
        s = params
    else: s = np.clip(1.0 + np.random.randn() * std, a_min=low, a_max=high)
    return data * s


def zoom_params(low:float=0.9, high:float=1.0) -> float:
    ''' Generate random params that Zoom needs.
    args: see zoom().
    '''
    s = np.random.rand() * (high-low) + low
    return s


def zoom(data:torch.Tensor, dim:int, low:float=0.9,
    high:float=1.0, params:float=None) -> torch.Tensor:
    ''' Zoom data in time axis, by a random factor s in [low, high].
    args:
        data: torch.Tensor.
        dim: int, the index of the time series axis.
        low: float, the lower bound of the range of s, default = 0.9.
        high: float, the higher bound of the range of s, default = 1.0.
        params: float, if not None, use the params to augment data.
    '''
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    N = data.shape[dim]
    if params is not None:
        s = params
    else: s = np.random.rand() * (high-low) + low
    start = 0.5 * (1-s) * (N-1)
    end = start + s * (N-1)
    t = np.linspace(start, end, num=N)
    f = interp.interp1d(np.arange(N), data, kind='quadratic',
        axis=dim, fill_value=0, bounds_error=False)
    return torch.from_numpy(f(t))


def magnitude_warp_params(n_knots:int=6, std:float=0.01,
    low:float=0.0, high:float=2.0) -> np.ndarray:
    ''' Generate random params that Magnitude Warp needs.
    args: see magnitude_warp().
    '''
    knots = np.ones(n_knots) + np.random.randn(n_knots) * std
    return knots.clip(low, high)


def magnitude_warp(data:torch.Tensor, dim:int, n_knots:int=6, std:float=0.01,
        low:float=0.0, high:float=2.0, params:np.ndarray=None) -> np.ndarray:
    ''' Warp the data magnitude by a smooth curve along the time axis.
    args:
        data: np.ndarray, with any shape and dtype.
        dim: int, the index of time axis.
        n_knots: int, the number of random knots on the random curve.
        std: float, the standard deviations of knots.
        low and high: float, the lower and higher bounds of the random knots.
        params: np.ndarray, if not None, use the params to augment data.
    returns:
        np.ndarray, with the same shape as data.
    '''
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    N = data.shape[dim]
    if params is not None:
        knots = params
    else:
        knots = np.ones(n_knots) + np.random.randn(n_knots) * std
        knots = knots.clip(low, high)
    x_knots = np.linspace(0, 1, num=knots.shape[0], endpoint=True)
    tck = interp.splrep(x_knots, knots, s=0, per=False)
    xs = np.linspace(0, 1, num=N, endpoint=True)
    magnitude = interp.splev(xs, tck, der=0)
    slices = [None] * data.ndim
    slices[dim] = slice(None)
    return torch.from_numpy(data * magnitude[tuple(slices)]).to(torch.float32)


def time_warp_params(n_knots:int=6, std:float=0.01, low:float=0.0, high:float=2.0) -> np.ndarray:
    ''' Generate random params that Time Warp needs.
    args: see time_warp().
    '''
    knots = np.ones(n_knots) + np.random.randn(n_knots) * std
    return knots.clip(low, high)


def time_warp(data:torch.Tensor, dim:int, n_knots:int=6, std:float=0.01,
    low:float=0.0, high:float=2.0, params:np.ndarray=None) -> np.ndarray:
    ''' Warp the timestamps by a smooth curve.
        Version 2: Version 2: warps timestamps uniformly.
    args:
        data: np.ndarray, with any shape and dtype.
        axis: int, the index of time axis.
        n_knots: int, the number of random knots on the random curve.
            To be unified with magnitude warping, n_knots includes the start
            and end timestamps, though they are not warped, which means,
            if n_knots == 2, nothing will happen.
        std: float, the standard deviations of knots, acceptable range: [0, 0.30].
        low and high: float, the lower and higher bounds of the random knots.
        params: np.ndarray, if not None, use the params to augment data.
    '''
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    N = data.shape[dim]
    if params is not None:
        knots = params
    else:
        knots = np.ones(n_knots) + np.random.randn(n_knots) * std
        knots = knots.clip(low, high)
    x_knots = np.linspace(0, 1, num=knots.shape[0], endpoint=True)
    tck = interp.splrep(x_knots, knots, s=0, per=False)
    xs = np.linspace(0, 1, num=N-1, endpoint=True)
    t = interp.splev(xs, tck, der=0)
    t = np.concatenate([[0.0], np.cumsum(t)])
    t = ((N-1) * t / t[-1]).clip(0, N-1)
    f = interp.interp1d(np.arange(N), data, axis=dim)
    return torch.from_numpy(f(t)).to(torch.float32)

    
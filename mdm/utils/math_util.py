import numpy as np
from scipy import interpolate as interp


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

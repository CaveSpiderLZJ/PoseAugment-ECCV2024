import numpy as np
from scipy import signal
from typing import Union, Tuple


class Butterworth:
    
    
    def __init__(self, fs:float, cut:Union[float,Tuple[float,float]],
            mode:str='lowpass', order:int=4) -> None:
        ''' Init the parameters of the Butterworth filter.
        args:
            fs: float, the sampling frequency in Hz.
            cut: Union[float,Tuple[float,float]], the cut-off frequencies of the
                transition band. Use low and high cut-off frequencies or both of
                them depending on the filter mode.
            mode: str, in {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
                Default is 'lowpass'.
            order: int, the order of the Butterworth filter.
        '''
        nyq = 0.5 * fs
        if mode == 'lowpass' or mode == 'highpass':
            self.sos = signal.butter(order, cut/nyq, btype=mode, output='sos')
        elif mode == 'bandpass' or mode == 'bandstop':
            self.sos = signal.butter(order, [cut[0]/nyq, cut[1]/nyq], btype=mode, output='sos')
        else: raise Exception(f'Wrong filter mode: {mode}.')
    
    
    def filt(self, data:np.ndarray, axis:int=-1) -> np.ndarray:
        ''' Filter signals.
        args:
            data: np.ndarray, of any shape and any dtype.
            axis: int, the axis to perform 1d filtering. Default is -1.
        returns:
            np.ndarray, with the same shape as data, the filtered signals.
        '''
        return signal.sosfiltfilt(self.sos, data, axis=axis)

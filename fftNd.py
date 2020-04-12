import torch
import torch.nn as nn
import torch.nn.functional as F

# define names for 
names = 'abcdefghijklmnopqrstuvwxyz'

def __fftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=(), is_rfft=False, is_inverse=False):
    
    # Pointer to function to use
    fft_func = torch.fft
    if is_inverse:
            fft_func = torch.ifft

    # Collect arguments in dictionary for final call
    args = {'input':input, 'signal_ndim':signal_ndim, 'normalized':normalized}

    # Check if real to complex fft selected and asign arguments
    if is_rfft:
        args['onesided'] = onesided
        fft_func = torch.rfft
        if is_inverse:
            fft_func = torch.irfft
            args['signal_sizes'] = signal_sizes
    # If less or equal to 3 dimensions, use pytorch implementation
    if signal_ndim<=3:
        return fft_func(**args)

    # If fft with more than 3 dimensions requested continue
    # remaining_dims = signal_ndims-3
    # dimension_names = list(names[0:input.ndim])
    # named_tensor = tensor.refine_names(*dimension_names)
    return 0



### Specific functions, visible to users
def fftNd(input, signal_ndim=1, normalized=False):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)

def ifftNd(input, signal_ndim=1, normalized=False, signal_sizes=()):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, signal_sizes=signal_sizes, is_inverse=True)

def rfftNd(input, signal_ndim=1, normalized=False, onesided=True):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, is_rfft=True)

def irfftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=()):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, signal_sizes=signal_sizes, is_rfft=True, is_inverse=True)

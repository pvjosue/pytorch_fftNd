import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

letters = 'abcdefghijklmnopqrst'

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


    dimension_names = ['batch','chan']
    dimension_names.extend([letters[i] for i in range(input.ndim-2)])
    if is_rfft is False:
        dimension_names[-1] = 'complex'
    input = input.refine_names(*dimension_names)
    
    original_size = input.shape
    dims_ids = [n for n in range(len(original_size))]

    out_result = input
    args['signal_ndim'] = 1

    last_dim = 0
    if is_rfft is False:
        last_dim = 1

    for nDim in range(2,len(original_size)-last_dim):
        curr_char = dimension_names[nDim]
        # 1D fft of every dimension indivisually, so atach every other into batch dimension
        new_size = [(dimension_names[i]) for i in range(2,len(original_size)) if dimension_names[i]!=curr_char and dimension_names[i]!='complex'] 
        new_size = ['batch'] + new_size + ['chan',curr_char]

        extra_dims = 2
        if is_rfft is False or nDim>2:
            new_size.extend(['complex'])
            extra_dims = 3

        middle_result = out_result.align_to(*new_size)

        middle_size = middle_result.shape
        batch_size = [middle_result.shape[i] for i in range(middle_result.ndim-extra_dims)]
        batch_size = np.prod(batch_size)

        view_size = [batch_size, original_size[1],original_size[nDim]]
        if is_rfft is False or nDim>2:
            view_size.extend([2])
        middle_result = middle_result.contiguous().rename(None).view(view_size)

        # apply 1D fft
        args['input'] = middle_result
        middle_result = fft_func(**args)

        out_result = middle_result.view(middle_size).refine_names(*new_size)
        out_result = out_result.align_as(input)

    return out_result.rename(None)



### Specific functions, visible to users
def fftNd(input, signal_ndim=1, normalized=False):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)

def ifftNd(input, signal_ndim=1, normalized=False, signal_sizes=()):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, signal_sizes=signal_sizes, is_inverse=True)

def rfftNd(input, signal_ndim=1, normalized=False, onesided=True):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, is_rfft=True)

def irfftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=()):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, signal_sizes=signal_sizes, is_rfft=True, is_inverse=True)

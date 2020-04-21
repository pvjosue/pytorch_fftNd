import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

letters = 'abcdefghijklmnopqrst'

def __fftNd(input, signal_ndim=1, normalized=False, onesided=True, is_rfft=False, is_inverse=False):
    
    # Collect arguments in dictionary for final call
    args = {'input':input, 'signal_ndim':signal_ndim, 'normalized':normalized}
    # Pointer to function to use
    fft_func = torch.fft
    if is_inverse:
        fft_func = torch.ifft

    # If less or equal to 3 dimensions, use pytorch implementation
    if signal_ndim<=3:
        return fft_func(**args)

    # Assign names to dimensions for easier permuting
    dimension_names = ['batch','chan']
    dimension_names.extend([letters[i] for i in range(input.ndim-2)])
    dimension_names[-1] = 'complex'
    input = input.refine_names(*dimension_names)
    
    original_size = input.shape
    dims_ids = [n for n in range(len(original_size))]

    # Set signal dimention to 1, as nD fourier is performed by n sucesive 1D ffts
    out_result = input
    args['signal_ndim'] = 1
    last_dim = 1

    # Iterate dimensions
    for nDim in range(2,len(original_size)-last_dim):
        curr_char = dimension_names[nDim]
        # 1D fft of every dimension indivisually, so atach every other into batch dimension
        new_size = [(dimension_names[i]) for i in range(2,len(original_size)) if dimension_names[i]!=curr_char and dimension_names[i]!='complex'] 
        new_size = ['batch'] + new_size + ['chan',curr_char,'complex']

        # Permute such that all dimensions are stacked to the batch dimension, except the nDim
        middle_result = out_result.align_to(*new_size)

        # Compute view shape to run fft 1D on
        middle_size = list(middle_result.shape)
        batch_size = [middle_result.shape[i] for i in range(middle_result.ndim-2)]
        batch_size = np.prod(batch_size)
        view_size = [batch_size, original_size[1],original_size[nDim], 2]
        # And reshape
        middle_result = middle_result.contiguous().rename(None).view(view_size)

        # Update arguments for fft
        args['input'] = middle_result

        # Check if it is irfft for last dimension
        if is_inverse and is_rfft and nDim == len(original_size)-last_dim-1:
            fft_func = torch.irfft
            # Remove complex dimension
            middle_size = middle_size[:-1]
            new_size = new_size[:-1]
            args['onesided'] = onesided
            if onesided:
                middle_size[-1] += middle_size[-1]//2
        if is_inverse == False and is_rfft and nDim == len(original_size)-last_dim-1:
            fft_func = torch.irfft
            # Remove complex dimension
            middle_size = middle_size[:-1]
            new_size = new_size[:-1]
            args['onesided'] = False

        # Run fft_func
        middle_result = fft_func(**args)

        # Get back to original shape
        out_result = middle_result.view(middle_size).refine_names(*new_size)
        out_result = out_result.align_as(input)
    # Remove dimention names and return result
    return out_result.rename(None)



### Specific functions, visible to users
def fftNd(input, signal_ndim=1, normalized=False):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)

def ifftNd(input, signal_ndim=1, normalized=False, signal_sizes=()):
    return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, is_inverse=True)

def rfftNd(input, signal_ndim=1, normalized=False, onesided=True):
    # Simulate rfft with ffts
    dims = input.ndim * [1]
    dims.extend([2])
    input = input.unsqueeze(input.ndim).repeat(dims)
    input[...,1] = 0
    result = __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)
    if onesided:
        result = result[...,:result.shape[-2]//2+1,:]
    return result

def irfftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=()):
    result = __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, is_rfft=True, is_inverse=True)
    return result[...,0]

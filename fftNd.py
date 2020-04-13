import torch
import torch.nn as nn
import torch.nn.functional as F

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

    # If fft with more than 3 dimensions requested continue
    remaining_dims = signal_ndim-3
    dimension_names = ['batch','chan']
    dimension_names.extend([letters[i] for i in range(input.ndim-2)])
    if is_rfft is False:
        dimension_names[-1] = 'complex'
    input = input.refine_names(*dimension_names)

    if remaining_dims<3:
        # Merge some dimensions in the batch dimension
        input2 = input.align_to('batch','a','chan','...')
        original_size = input2.shape

        newSize = list([input2.shape[0]*input2.shape[1]])
        newSize.extend(input2.shape[2:])
        temp_names = input2.names
        input2 = input2.rename(None).view(*newSize) 

        args['input'] = input2
        args['signal_ndim'] = 3
        result1 = fft_func(**args)
        result1 = result1.view(*original_size).refine_names(*temp_names)
        result1 = result1.align_as(input)

        input2 = result1.align_to('batch','b','c','d','chan','...')
        original_size = input2.shape
        newSize = list([input2.shape[0]*input2.shape[1]*input2.shape[2]*input2.shape[3]])
        newSize.extend(input2.shape[4:])
        temp_names = input2.names
        input2 = input2.rename(None).view(*newSize) 

        args['input'] = input2
        args['signal_ndim'] = remaining_dims
        result1 = fft_func(**args)
        result1 = result1.view(*original_size).refine_names(*temp_names)
        result1 = result1.align_as(input)

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

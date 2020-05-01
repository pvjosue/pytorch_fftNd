# fftNd in Pytorch
An n-dimensional implementation of the Fast Fourier Transform and its inverse in Pytorch.
The included functions are:
* `fftNd(input, signal_ndim=1, normalized=False)`
* `ifftNd(input, signal_ndim=1, normalized=False, signal_sizes=())`
* `rfftNd(input, signal_ndim=1, normalized=False, onesided=True)`
* `irfftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=())`
These functions are working with all 

## Functionality
The nD fourier transform is performed by applying _n_ 1D FFTs in a batch matter. This functions use Pytorch named tensors for aranging the dimensions in each 1D FFT.

## Examples
The main.py contains a comparison between each fft function against its numpy conterpart. Also a simple nD Fourier convolution is used for evaluation.

## Usage
```python
import torch
from fftNd import *

device = "cpu"
# Create input complex tensor
x = torch.rand(1,1,5,5,2) 
# dimensions: [batch, channels, dim1, dim2, ... , dimN, complex] note that the last dimension is shape two, to allocate the real and imaginary parts.

# Test wrappers with 2D
y = rfftNd(x,2)
y = ifftNd(y,2)
y = rfftNd(x[:,:,:,:,0],2)
y = irfftNd(y,2)

# Test 6D fft
nDims = 6
dim_sizes = [1,1]
# each dimension has size 5
dim_sizes.extend(nDims*[5])
# add complex dimension
dim_sizes.extend([2])
# Create complex input tensor
x = torch.rand(*dim_sizes).to(device)

with torch.no_grad():
    # Run proposed implementation
    torch.cuda.synchronize()
    start = time.time()
    y = fftNd(x,nDims)
    torch.cuda.synchronize()
    end = time.time()
    print("time: " + str(end-start))
```

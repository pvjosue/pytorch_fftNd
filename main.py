import torch
from fftNd import *
import numpy as np
import time
# torch.manual_seed(1230)

def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return s, axes

def irfftn(a, s=None, axes=None, norm=None):
    a = np.asarray(a)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = np.fft.ifft(a, s[ii], axes[ii], norm)
    a = np.fft.irfft(a, s[-1], axes[-1], norm)
    return a


device = "cpu"
# Create input complex tensor
x = torch.rand(1,1,5,5,2)


# Test wrappers with 2D
# y = rfftNd(x,2)
# y = ifftNd(y,2)
# y = rfftNd(x[:,:,:,:,0],2)
# y = irfftNd(y,2)

# test 4D
nDims = 4
dim_sizes = [1,1]
dim_sizes.extend(nDims*[33])
dim_sizes.extend([2])
x = torch.rand(*dim_sizes).to(device)

torch.cuda.synchronize()
start = time.time()
y = fftNd(x,nDims)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

start = time.time()
yNumpy = np.fft.fftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

Y = torch.zeros_like(x)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

diff = abs(y-Y).sum().item()
# print(y)
# print(Y)
print('\tdiff: ' + str(diff))

# IFFT
torch.cuda.synchronize()
start = time.time()
y = ifftNd(x,nDims)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

start = time.time()
yNumpy = np.fft.ifftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

Y = torch.zeros_like(x)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

diff = abs(y-Y).sum().item()
# print(y)
# print(Y)
print('\tdiff: ' + str(diff))


# RFFT
torch.cuda.synchronize()
start = time.time()
y = rfftNd(x[...,0],nDims,onesided=True)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

xNumpy = x[0,0,:,...,0].cpu().numpy()

start = time.time()
yNumpy = np.fft.rfftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

Y = torch.zeros_like(y)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

diff = abs(y-Y).sum().item()
# print(y)
# print(Y)
print('\tdiff: ' + str(diff))


# irfft
torch.cuda.synchronize()
start = time.time()
y = irfftNd(x,nDims, onesided=False)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

start = time.time()
yNumpy = irfftn(xNumpy, xNumpy.shape)#[...,:xNumpy.shape[-1]]
end = time.time()
print("time: " + str(end-start))

Y = torch.zeros_like(y)
Y[0,0,:,...] = torch.from_numpy(np.real(yNumpy))

diff = abs(y-Y).sum().item()
# print(y)
# print(Y)
print('\tdiff: ' + str(diff))
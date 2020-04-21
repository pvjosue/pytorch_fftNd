import torch
from fftNd import *
import numpy as np
import time
torch.manual_seed(1230)

device = "cuda"
# Create input complex tensor
x = torch.rand(1,1,5,5,2)


# Test wrappers with 2D
y = rfftNd(x,2)
y = ifftNd(y,2)
y = rfftNd(x[:,:,:,:,0],2)
y = irfftNd(y,2)

############# test 4D convs
nDims = 6
dim_sizes = [1,1]
dim_sizes.extend(nDims*[5])
dim_sizes.extend([2])
# Create complex input tensor
x = torch.rand(*dim_sizes).to(device)

######### fftNd
# Run proposed implementation
torch.cuda.synchronize()
start = time.time()
y = fftNd(x,nDims)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

# Create Numpy tensor
xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

# Run numpy implementation
start = time.time()
yNumpy = np.fft.fftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

# Copy result to tensor
Y = torch.zeros_like(x)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

# Compute error
diff = abs(y-Y).sum().item()
print('\tfftNd diff: ' + str(diff))

########## ifftNd
torch.cuda.synchronize()
start = time.time()
y = ifftNd(x,nDims)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

# Create Numpy tensor
xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

# Run numpy implementation
start = time.time()
yNumpy = np.fft.ifftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

# Copy result to tensor
Y = torch.zeros_like(x)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

# Compute error
diff = abs(y-Y).sum().item()
print('\tifftNd diff: ' + str(diff))


########## rfftNd
torch.cuda.synchronize()
start = time.time()
y = rfftNd(x[...,0],nDims,onesided=True)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

# Create Numpy tensor
xNumpy = x[0,0,:,...,0].cpu().numpy()

# Run numpy implementation
start = time.time()
yNumpy = np.fft.rfftn(xNumpy)
end = time.time()
print("time: " + str(end-start))

# Copy result to tensor
Y = torch.zeros_like(y)
Y[0,0,:,...,0] = torch.from_numpy(np.real(yNumpy))
Y[0,0,:,...,1] = torch.from_numpy(np.imag(yNumpy))

# Compute error
diff = abs(y-Y).sum().item()
print('\trfftNd diff: ' + str(diff))


########## irfftNd
torch.cuda.synchronize()
start = time.time()
y = irfftNd(x,nDims, onesided=False)
torch.cuda.synchronize()
end = time.time()
print("time: " + str(end-start))

# Create Numpy tensor
xNumpy = x[0,0,:,...,0].cpu().numpy() + x[0,0,:,...,1].cpu().numpy()* 1j

# Run numpy implementation
start = time.time()
yNumpy = np.fft.irfftn(xNumpy, xNumpy.shape)
end = time.time()
print("time: " + str(end-start))

# Copy result to tensor
Y = torch.zeros_like(y)
Y[0,0,:,...] = torch.from_numpy(np.real(yNumpy))

# Compute error
diff = abs(y-Y).sum().item()
print('\tirfftNd diff: ' + str(diff))



######### convNd with fourier
def mulComplex(x, other):
    outR = torch.sub(torch.mul(x[...,0], other[...,0]), torch.mul(x[...,1], other[...,1]))
    outI = torch.add(torch.mul(x[...,0], other[...,1]), torch.mul(x[...,1], other[...,0]))
    out = torch.cat((outR.unsqueeze(-1), outI.unsqueeze(-1)), outR.ndimension())
    return out

def fft_conv(A,B):
    nDims = A.ndim-2
    padSize = (torch.tensor(A.shape[2:])-torch.tensor(B.shape[2:]))
    padSizes = torch.zeros(2*nDims,dtype=int)
    padSizes[0::2] = torch.floor(padSize/2.0)
    padSizes[1::2] = torch.ceil(padSize/2.0)
    padSizes = list(padSizes.numpy()[::-1])
    B = F.pad(B,padSizes)
    return ifftNd( mulComplex(rfftNd(A, nDims,onesided=False),rfftNd(B, nDims,onesided=False)), nDims)[...,0]

def np_fftconvolve(A, B):
    nDims = A.ndim-2
    padSize = (torch.tensor(A.shape[2:])-torch.tensor(B.shape[2:]))
    padSizes = torch.zeros(2*nDims,dtype=int)
    padSizes[0::2] = torch.floor(padSize/2.0)
    padSizes[1::2] = torch.ceil(padSize/2.0)
    padSizes = list(padSizes.numpy()[::-1])
    B = F.pad(B,padSizes)
    A = A[0,0,...].numpy()
    B = B[0,0,...].numpy()
    return np.real(np.fft.ifftn(np.fft.fftn(A)*np.fft.fftn(B, s=A.shape)))

# Create image and kernel
A = torch.rand(1,1,5,6,7,8)
B = torch.rand(1,1,2,3,4,5)

# Run proposed implementation
start = time.time()
C = fft_conv(A,B)
end = time.time()
print("time: " + str(end-start))

# Create Numpy tensors
Anumpy = A[0,0,...].numpy()
Bnumpy = B[0,0,...].numpy()

# Run numpy implementation
start = time.time()
Cnumpy = np_fftconvolve(A,B)
end = time.time()
print("time: " + str(end-start))
Cnumpy = torch.from_numpy(Cnumpy).unsqueeze(0).unsqueeze(0)

# Compute error
diff = abs(C-Cnumpy).sum().item()
print('\t Fourier Conv. diff: ' + str(diff))
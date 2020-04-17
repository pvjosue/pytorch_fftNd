import torch
from fftNd import *
import numpy as np
import time
torch.manual_seed(1230)

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
dim_sizes.extend(nDims*[5])
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
print('diff: ' + str(diff))
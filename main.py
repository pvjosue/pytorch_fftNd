import torch
from fftNd import *

# Create input complex tensor
x = torch.rand(1,1,5,5,2)


# Test wrappers with 2D
y = fftNd(x,2)
y = ifftNd(y,2)
y = rfftNd(x[:,:,:,:,0],2)
y = irfftNd(y,2)


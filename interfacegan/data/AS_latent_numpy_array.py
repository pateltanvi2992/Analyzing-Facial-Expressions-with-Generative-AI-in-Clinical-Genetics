import torch
import numpy

a = torch.load('/data/patelt6/encoder4editing/AS_and_Unaffected_inversion/latents.pt')
 
sampleTensor = torch.tensor(a)
#sampleTensor = torch.flatten(sampleTensor)
print(sampleTensor.ndim)
sampleTensor = torch.flatten(sampleTensor, start_dim=1)
print(sampleTensor.ndim)
y = sampleTensor.data.cpu().numpy()

y = numpy.array(y, dtype=numpy.ndarray)

print(y.shape)

numpy.save('/data/patelt6/encoder4editing/AS_and_Unaffected_inversion/AS_latents.npy', y)
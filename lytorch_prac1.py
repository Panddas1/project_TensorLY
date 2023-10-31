from tltorch import *
import torch



from torch import nn
import numpy as np

def generate_random_complex_matrix(rows, cols):
    real_part = np.random.randn(rows, cols)
    imag_part = np.random.randn(rows, cols)
    complex_matrix = real_part + 1j * imag_part
    return torch.tensor(complex_matrix, dtype=torch.complex64,requires_grad=True)


shape=[10,10,10]
tucker_tensor = FactorizedTensor.new(shape, rank=2, factorization='cp')
tucker_tensor=tucker_tensor.normal_(0,0.2)

print("--------------------------------------")

#TRL
input_shape = (4, 5,6)
output_shape = (6,)
batch_size = 2

device = 'cpu'

input = torch.randn((batch_size,) + input_shape,
                dtype=torch.complex64, device=device)

trl = TRL(input_shape, output_shape, rank=2,dtype=torch.complex64)
result = trl(input)
for p in trl.parameters():
   print(p)

tcl=TCL(input_shape=input_shape,rank=[3,4,5],dtype=torch.complex64)

# for p in tcl.parameters():
#     p.requires_grad=False

result2 = tcl(input)

s1=torch.sigmoid(result2)

pass


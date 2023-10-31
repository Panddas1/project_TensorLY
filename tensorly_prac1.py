from tensorly import tenalg
import tensorly
import numpy as np
import torch

# tensor linear algebra
x1=np.random.randn(2,3)
x2=np.random.randn(3,3)
x3=tenalg.khatri_rao([x1,x2])
x4=tenalg.kronecker([x1,x2])
x5=tenalg.proximal.soft_thresholding(x1, 1)

x6=np.random.randn(3)
x7=np.random.randn(7)
x8=tenalg.outer([x6,x7])

print("--------------------------------")

u1=np.random.randn(2,3)
X=tensorly.cp_tensor.cp_to_tensor()




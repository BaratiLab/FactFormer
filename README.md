# FactFormer
Official implementation of Scalable Transformer for PDE surrogate modeling

### Getting started

The code is tested under PyTorch 1.8.2 and CUDA 11, later versions should also work. Other packages needed are Numpy/einops/Matplotlib/tqdm

Use a layer of factorized attention:

```python
import torch
from libs.factorization_module import FABlock3D

fa_layer = FABlock3D(dim,                   # input dimension
                     dim_head,              # dimension in each attention head, will be expanded by the kernel_multiplier when computing kernel: d = dim_head * kernel_multiplier
                     latent_dim,            # the output dimension of the projection operator
                     heads,                 # attention heads
                     dim_out,               # output dimension
                     kernel_multiplier,     # use more function bases to computer kernel: k(x_i, x_j)=\sum_{c}^dq_c(x_i)k_c(x_j)    
                     use_rope,              # use rotary positional encoding or not, by default True
                     scaling_factor         # use scaling factor to modulate the kernel, an example is 1/ sqrt(d) like scaled-dot product attention, by default is: 1
                    )
```

### Dataset



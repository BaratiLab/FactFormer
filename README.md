# FactFormer
Official implementation of Scalable Transformer for PDE surrogate modeling

### Getting started

The code is tested under PyTorch 1.8.2 and CUDA 11, later versions should also work. Other packages needed are Numpy/einops/Matplotlib/tqdm

Use a layer of 3D factorized attention:

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
# random input
z = torch.randn(((4, 64, 64, 64, dim))
# axial coords
pos_x = torch.linspace(0, 1, 64)
pos_y = torch.linspace(0, 1, 64)
pos_z = torch.linspace(0, 1, 64)

z = fa_layer(z, [pos_x, pos_y, pos_z])
```

For running experiments on the problems discussed in the paper, please refer to the ```examples``` directory. 

For example, training a model for Darcy flow:

```python darcy2d_fact.py --config darcy2d_fact.yml```

### Dataset

We provided our generated dataset at: 
* Kmflow 2D: https://drive.google.com/file/d/1QkiF6ClryURuiBMLqXc50Fvf7lJI7Ajm/view?usp=sharing
* Isotropic 3D: https://drive.google.com/drive/folders/1oTtA8k56R6zL6xgjBg9eFTr1v_fwtN4C?usp=sharing
* Smoke 3D: https://drive.google.com/drive/folders/14E3cYd_V06jU-i040FVerHYhbFpmWyXv?usp=sharing

The Darcy dataset can be obtained from FNO's [repo](https://github.com/neuraloperator/neuraloperator/tree/master) .

Please refer to the scripts under ```dataset``` to generate data or customize the dataset.



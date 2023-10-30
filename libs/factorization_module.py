import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Union, Tuple, List, Optional
from libs.positional_encoding_module import RotaryEmbedding, apply_rotary_pos_emb, SirenNet
from libs.basics import PreNorm, PostNorm, GeAct, MLP, masked_instance_norm
from libs.attention import LowRankKernel


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP([hidden_dim, hidden_dim, out_dim], GeAct(nn.GELU())))

    def forward(self, x):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.out_ffn(x)
        return x  # b nx c


class FABlock2D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny c -> b ny nx c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)

        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_head * heads),
            nn.Linear(dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u, pos_lst):
        # x: b c h w
        u = self.in_norm(u)
        v = self.to_v(u)
        u = self.to_in(u)

        u_x = self.to_x(u)
        u_y = self.to_y(u)

        pos_x, pos_y = pos_lst
        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)

        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_y, u_phi)
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)
        return self.to_out(u_phi)


class FABlock3D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny nz c -> b ny nx nz c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_z = nn.Sequential(
            Rearrange('b nx ny nz c -> b nz nx ny c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_z = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)

        self.to_out = nn.Sequential(
            nn.InstanceNorm3d(dim_head * heads),
            nn.Linear(dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u, pos_lst):
        # x: b h w d c
        u = self.in_norm(u)
        v = self.to_v(u)
        u = self.to_in(u)

        u_x = self.to_x(u)
        u_y = self.to_y(u)
        u_z = self.to_z(u)
        pos_x, pos_y, pos_z = pos_lst

        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)
        k_z = self.low_rank_kernel_z(u_z, pos_x=pos_z)

        u_phi = rearrange(v, 'b i l r (h c) -> b h i l r c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmsc->bhimsc', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhimsc->bhilsc', k_y, u_phi)
        u_phi = torch.einsum('bhrs,bhilsc->bhilrc', k_z, u_phi)
        u_phi = rearrange(u_phi, 'b h i l r c -> b i l r (h c)', h=self.heads)

        return self.to_out(u_phi)

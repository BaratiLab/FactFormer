import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from .positional_encoding_module import RotaryEmbedding, GaussianFourierFeatureTransform, \
    apply_rotary_pos_emb, apply_2d_rotary_pos_emb, SirenNet, apply_3d_rotary_pos_emb
from .basics import PreNorm, PostNorm, MLP, GeAct, masked_instance_norm
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class AsymetricKernel(nn.Module):
    # position encoded linear attention with Galerkin style normalization
    # OFormer: https://arxiv.org/abs/2205.13671
    # Galerkin Transformer: https://arxiv.org/abs/2105.14995
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 positional_embedding='rotary',
                 pos_dim=2,
                 project_query=True,
                 to_out_fn=None,
                 kernel_multiplier=1,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim_head = dim_head
        self.kernel_dim = kernel_multiplier * dim_head

        self.project_query = project_query
        if project_query:
            self.to_q = nn.Linear(dim, kernel_multiplier * dim_head * heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(dim, kernel_multiplier * dim_head * heads, bias=False)

        self.k_norm = nn.InstanceNorm1d(kernel_multiplier * dim_head, affine=False)
        self.v_norm = nn.InstanceNorm1d(dim_head, affine=False)

        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        if to_out_fn is None:
            self.to_out = nn.Linear(dim_head * heads, dim) if dim_head * heads != dim else nn.Identity()
        else:
            self.to_out = to_out_fn

        assert positional_embedding in ['rff', 'rotary', 'learnable', 'none']
        self.positional_embedding = positional_embedding
        self.pos_dim = pos_dim

        if positional_embedding == 'rff':
            self.pos_emb = GaussianFourierFeatureTransform(pos_dim, self.kernel_dim, scale=8,
                                                           learnable=False, num_heads=heads)
        elif positional_embedding == 'rotary':
            self.pos_emb = RotaryEmbedding(self.kernel_dim//self.pos_dim, min_freq=1/64)
        elif positional_embedding == 'learnable':
            self.pos_emb = nn.Sequential(
                GaussianFourierFeatureTransform(pos_dim, dim_head * heads // 2, scale=8,
                                                learnable=True),
                nn.Linear(dim_head * heads // 2, dim_head*heads, bias=False),
                nn.GELU(),
                nn.Linear(dim_head*heads, self.kernel_dim*heads, bias=False))
        else:
            pass
        self.init_gain = 1 / np.sqrt(dim_head)
        self.diagonal_weight = self.init_gain
        # self.diagonal_weight = nn.Parameter(1 / np.sqrt(dim_head) *
        #                                     torch.ones(heads, 1, 1), requires_grad=True)
        self.initialize_qkv_weights()


    def init_weight(self, weight, inif_fn):
        for param in weight.parameters():
            if param.ndim > 1:
                dim_head = param.size(0) // self.heads
                for h in range(self.heads):
                    inif_fn(param[h * dim_head:(h + 1) * dim_head, :], gain=self.init_gain)
                    #
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def initialize_qkv_weights(self):
        init_fn = xavier_uniform_

        if self.project_query:
            self.init_weight(self.to_q, init_fn)
        self.init_weight(self.to_k, init_fn)
        self.init_weight(self.to_v, init_fn)


    def normalize_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, u_x, u_y=None, pos_x=None, pos_y=None,
                associate=True,
                get_attention=False):
        # u_x, u_y: b n c
        # u_x is from the first source
        # u_y is from the second source
        # pos: b n d
        if u_y is None:
            u_y = u_x

        if get_attention and associate:
            raise Exception('Cannot get attention when associate is True')

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)
        v = self.to_v(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # q = self.q_norm(q)
        # galerkin-style normalization
        k = self.normalize_wrt_domain(k, self.k_norm)
        v = self.normalize_wrt_domain(v, self.v_norm)

        if self.positional_embedding != 'none' and pos_x is None:
            raise ValueError('positional embedding is not none but pos is None')

        if self.positional_embedding != 'rotary' and\
                self.positional_embedding != 'none' and\
                self.positional_embedding != 'rff':
            pos_x_emb = self.pos_emb(pos_x)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y)
            q = q * pos_x_emb
            k = k * pos_y_emb
        elif self.positional_embedding == 'rff':

            pos_x_emb = self.pos_emb(pos_x, unfold_head=True)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y, unfold_head=True)

            # duplicate q, k
            q_ = torch.cat((q, q), dim=-1)
            k_ = torch.cat((k, k), dim=-1)
            q = q_ * pos_x_emb
            k = k_ * pos_y_emb

        elif self.positional_embedding == 'rotary':
            if self.pos_dim == 3:
                assert pos_x.shape[-1] == 3
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_z = self.pos_emb.forward(pos_x[..., 2], q.device)
                q_freqs_x = repeat(q_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])
                q_freqs_z = repeat(q_freqs_z, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                    k_freqs_z = q_freqs_z
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_z = self.pos_emb.forward(pos_y[..., 2], k.device)
                    k_freqs_x = repeat(k_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])
                    k_freqs_z = repeat(k_freqs_z, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])

                q = apply_3d_rotary_pos_emb(q, q_freqs_x, q_freqs_y, q_freqs_z)
                k = apply_3d_rotary_pos_emb(k, k_freqs_x, k_freqs_y, k_freqs_z)

            elif self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_x = repeat(q_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_x = repeat(k_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])

                q = apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = self.pos_emb.forward(pos_x[..., 0], q.device).unsqueeze(0)
                q_freqs = repeat(q_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = self.pos_emb.forward(pos_y[..., 0], k.device).unsqueeze(0)
                    k_freqs = repeat(k_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 3 dimensions')
        else:     # do nothing
            pass

        if associate:
            # dots = torch.matmul(k.transpose(-1, -2), v)
            # u = torch.matmul(q, dots) / n
            dots = torch.einsum('bhic,bhid->bhcd', k, v)
            u = torch.einsum('bhic,bhcd->bhid', q, dots) / n

        else:
            # this is more efficient when n<<c
            # score = torch.matmul(q, k.transpose(-1, -2))
            # u = torch.matmul(score, v) / n
            score = torch.einsum('bhid,bhjd->bhij', q, k)
            u = torch.einsum('bhij,bhjd->bhid', score, v) / n

        u = rearrange(u, 'b h n d -> b n (h d)')
        u = self.to_out(u)
        if get_attention:
            return u, score
        return u


class LowRankKernel(nn.Module):
    # low rank kernel, ideally operates only on one dimension
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 positional_embedding='rotary',
                 pos_dim=1,
                 normalize=False,
                 softmax=False,
                 residual=True,
                 dropout=0,
                 scaling=1,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim_head = dim_head
        self.heads = heads
        self.normalize = normalize
        self.residual = residual
        if dropout > 1e-6:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.to_q = nn.Linear(dim, dim_head*heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head*heads, bias=False)

        assert positional_embedding in ['rff', 'rotary', 'learnable', 'none']
        self.positional_embedding = positional_embedding
        self.pos_dim = pos_dim

        if positional_embedding == 'rff':
            self.pos_emb = GaussianFourierFeatureTransform(pos_dim, dim_head, scale=1,
                                                           learnable=False, num_heads=heads)
        elif positional_embedding == 'rotary':
            self.pos_emb = RotaryEmbedding(dim_head//self.pos_dim, min_freq=1/64)
        elif positional_embedding == 'learnable':
            self.pos_emb = nn.Sequential(
                GaussianFourierFeatureTransform(pos_dim, dim_head * heads // 2, scale=1,
                                                learnable=True),
                nn.Linear(dim_head * heads // 2, dim_head*heads, bias=False),
                nn.GELU(),
                nn.Linear(dim_head*heads, dim_head*heads, bias=False))
        else:
            pass
        self.init_gain = 0.02   # 1 / np.sqrt(dim_head)
        # self.diagonal_weight = nn.Parameter(1 / np.sqrt(dim_head) *
        #                                     torch.ones(heads, 1, 1), requires_grad=True)
        self.initialize_qk_weights()
        self.softmax = softmax

        self.residual = residual
        if self.residual:
            self.gamma = nn.Parameter(torch.tensor(1 / np.sqrt(dim_head)), requires_grad=True)
        else:
            self.gamma = 0
        self.scaling = scaling

    def initialize_qk_weights(self):
        xavier_uniform_(self.to_q.weight, gain=self.init_gain)
        xavier_uniform_(self.to_k.weight, gain=self.init_gain)
        # torch.nn.init.normal_(self.to_q.weight, std=self.init_gain)
        # torch.nn.init.normal_(self.to_k.weight, std=self.init_gain)

    def normalize_wrt_domain(self, x):
        x = (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True) + 1e-5)
        return x

    def forward(self, u_x, u_y=None, pos_x=None, pos_y=None):
        # u_x, u_y: b n c
        # u_x is from the first source
        # u_y is from the second source
        # pos: b n d
        if u_y is None:
            u_y = u_x

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        if self.normalize:
            q = self.normalize_wrt_domain(q)
            k = self.normalize_wrt_domain(k)

        if self.positional_embedding != 'none' and pos_x is None:
            raise ValueError('positional embedding is not none but pos is None')

        if self.positional_embedding != 'rotary' and \
                self.positional_embedding != 'none' and \
                self.positional_embedding != 'rff':
            pos_x_emb = self.pos_emb(pos_x)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y)
            q = q * pos_x_emb
            k = k * pos_y_emb
        elif self.positional_embedding == 'rff':

            pos_x_emb = self.pos_emb(pos_x, unfold_head=True)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y, unfold_head=True)

            # duplicate q, k
            q_ = torch.cat((q, q), dim=-1)
            k_ = torch.cat((k, k), dim=-1)
            q = q_ * pos_x_emb
            k = k_ * pos_y_emb

        elif self.positional_embedding == 'rotary':
            if self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_x = repeat(q_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_x = repeat(k_freqs_x, 'b n d -> b h n d', h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, 'b n d -> b h n d', h=k.shape[1])

                q = apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = self.pos_emb.forward(pos_x[..., 0], q.device).unsqueeze(0)
                q_freqs = repeat(q_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = self.pos_emb.forward(pos_y[..., 0], k.device).unsqueeze(0)
                    k_freqs = repeat(k_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        else:  # do nothing
            pass

        K = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling  # if not on uniform grid, need to consider quadrature weights
        K = self.dropout(K)
        if self.softmax:
            K = F.softmax(K, dim=-1)
        if self.residual:
            K = K + self.gamma * torch.eye(n).to(q.device).view(1, 1, n, n) / n
        return K












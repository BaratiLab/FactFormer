import glob

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import gc
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader, TensorDataset
import logging, pickle, h5py

from libs.positional_encoding_module import GaussianFourierFeatureTransform
from libs.factorization_module import FABlock3D
from libs.basics import PreNorm, MLP
from utils import Trainer, dict2namespace, index_points, load_checkpoint, save_checkpoint, ensure_dir
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from loss_fn import rel_l2_loss

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil
from collections import OrderedDict
from train_utils import CurriculumSampler
import random


torch.backends.cudnn.benchmark = True


class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(3, dim // 2, 8),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock3D(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, nz, c = u.shape  # just want to make sure its shape
        nx, ny, nz = pos_lst[0].shape[0], pos_lst[1].shape[0], pos_lst[2].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1),
                                          pos_lst[1].squeeze(-1),
                                          pos_lst[2].squeeze(-1)]
                                         ), dim=-1).reshape(-1, 3)

        for l, (pos_enc, attn_layer) in enumerate(self.layers):
            u += rearrange(pos_enc(pos), '1 (nx ny nz) c -> 1 nx ny nz c', nx=nx, ny=ny, nz=nz)
            u = attn_layer(u, pos_lst) + u
        return u


class FactFormer3D(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()
        self.config = config

        self.in_dim = config.in_dim
        self.in_tw = config.in_time_window
        self.out_dim = config.out_dim
        self.out_tw = config.out_time_window

        self.dim = config.dim                 # dimension of the transformer
        self.depth = config.depth           # depth of the encoder transformer
        self.dim_head = config.dim_head

        self.heads = config.heads

        self.pos_in_dim = config.pos_in_dim
        self.pos_out_dim = config.pos_out_dim

        self.kernel_multiplier = config.kernel_multiplier
        self.latent_mutliplier = config.latent_mutliplier
        self.latent_dim = int(self.dim * self.latent_mutliplier)
        self.max_latent_steps = config.max_latent_steps

        # flatten time window
        self.to_in = nn.Sequential(
            nn.Conv2d(self.in_dim, self.dim // 2, kernel_size=1, stride=1, padding=0, groups=self.in_dim),
            nn.GELU(),
            nn.Conv2d(self.dim // 2, self.dim, kernel_size=(self.in_tw, 1), stride=1, padding=0, bias=False),
        )

        # assume input is b c t h w d
        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier,)

        self.expand_latent = nn.Linear(self.dim, self.latent_dim, bias=False)
        self.latent_time_emb = nn.Parameter(torch.randn(1, self.max_latent_steps,
                                                        1, 1, 1, self.latent_dim)*0.02,
                                                        requires_grad=True)

        # simple propagation
        self.propagator = PreNorm(self.latent_dim,
                                  MLP([self.latent_dim, self.dim, self.latent_dim], act_fn=nn.GELU()))
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny nz c -> b c (nx ny nz)'),
            nn.GroupNorm(num_groups=8, num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, self.dim, kernel_size=1, stride=1, padding=0,
                      groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim // 2, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,
                u,
                pos_lst,
                latent_steps,
                ):
        # u: b c t h w d
        # each core has the shape of b c t r1 n r2
        # out: b c t n
        b, c, t, nx, ny, nz = u.shape
        u = rearrange(u, 'b c t nx ny nz -> b c t (nx ny nz)')
        u = self.to_in(u)
        u = rearrange(u, 'b c 1 (nx ny nz) -> b nx ny nz c', nx=nx, ny=ny, nz=nz)

        u = self.encoder(u, pos_lst)
        u = self.expand_latent(u)
        u_lst = []
        for l_t in range(latent_steps):
            # plus time embedding
            u = u + self.latent_time_emb[:, l_t, ...]
            u = self.propagator(u) + u
            u_lst.append(u)
        u = torch.cat(u_lst, dim=0)
        u = self.simple_to_out(u)
        u = rearrange(u, '(t b) c (nx ny nz) -> b t nx ny nz c', nx=nx, ny=ny, t=latent_steps)
        return u

# def loss_wrapper(x, y, denormalize_fn, lam=0.1):
def loss_wrapper(x, y):
    # x: b t nx ny nz c
    # break down the loss for each channel
    l2norm = 0.
    loss_dict = {}
    vel_loss = rel_l2_loss(x[..., 1:], y[..., 1:], dim=(1, 2, 3, 4), eps=1e-5, reduction='sum')
    loss_dict['u'] = vel_loss.detach().cpu().clone().numpy()
    prs_loss = rel_l2_loss(x[..., 0], y[..., 0], dim=(1, 2, 3, 4), eps=1e-5, reduction='sum')
    loss_dict['p'] = prs_loss.detach().cpu().clone().numpy()

    l2norm = vel_loss + prs_loss
    return l2norm, loss_dict


def test_loss_wrapper(x, y):
    # y b t nx ny nz c
    loss_dict = {}
    vel_loss = rel_l2_loss(x[..., 1:], y[..., 1:], dim=(1, 2, 3, 4), eps=1e-5, reduction='sum')
    loss_dict['u'] = vel_loss.detach().cpu().clone().numpy()
    prs_loss = rel_l2_loss(x[..., 0], y[..., 0], dim=(1, 2, 3, 4), eps=1e-5, reduction='sum')
    loss_dict['p'] = prs_loss.detach().cpu().clone().numpy()
    return loss_dict


class Fluid3DData(Dataset):
    # will contain five fields,
    def __init__(self, config, train=True, max_len=50):
        self.data_dir = config.data.data_dir
        self.resolution = config.data.resolution
        self.skip = 60 // self.resolution
        self.max_latent_steps = config.model.max_latent_steps
        self.f_lst = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        self.tw = config.model.in_time_window
        self.out_tw = config.model.out_time_window

        print(f'There are {len(self.f_lst)} files in {self.data_dir}')
        self.train = train
        self.train_idxs = list(range(0, config.data.train_num))
        self.test_idxs = list(range(len(self.f_lst)-config.data.test_num, len(self.f_lst)-0, 1))

        if self.train:
            self.idxs = self.train_idxs
        else:
            self.idxs = self.test_idxs

        self.cache = OrderedDict()
        self.stats = {}

        if os.path.exists(config.data.dataset_stat):
            print('Loading dataset stats from', config.data.dataset_stat)
            stats = np.load(config.data.dataset_stat, allow_pickle=True)
            # npz to dict
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data()
            print('Saving dataset stats to', config.data.dataset_stat)
            self.dump_stats(config.data.dataset_stat)
        print(f'Dataset stats: {self.stats}')
        self.max_len = max_len
        # self.load_all_data()

    def prepare_data(self):
        # load all training data and then calculate the mean/std online
        self.stats['prs_mean'] = 0
        self.stats['prs_std'] = 1
        self.stats['vel_mean'] = 0
        self.stats['vel_std'] = 1

        for idx in tqdm(self.train_idxs):
            prs = np.load(os.path.join(self.f_lst[idx], 'prs.npy'))
            vel = np.concatenate([np.load(os.path.join(self.f_lst[idx], 'vx.npy')),
                                  np.load(os.path.join(self.f_lst[idx], 'vy.npy')),
                                  np.load(os.path.join(self.f_lst[idx], 'vz.npy'))], axis=-1)
            self.stats['prs_mean'] += prs.mean()
            self.stats['prs_std'] += prs.std()
            self.stats['vel_mean'] += vel.mean()
            self.stats['vel_std'] += vel.std()

        self.stats['prs_mean'] /= len(self.train_idxs)
        self.stats['prs_std'] /= len(self.train_idxs)
        self.stats['vel_mean'] /= len(self.train_idxs)
        self.stats['vel_std'] /= len(self.train_idxs)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: b n n n c, assume the order is prs, vel
        prs_mean = torch.tensor(self.stats['prs_mean'], device=x.device, dtype=x.dtype)
        prs_std = torch.tensor(self.stats['prs_std'], device=x.device, dtype=x.dtype)
        vel_mean = torch.tensor(self.stats['vel_mean'], device=x.device, dtype=x.dtype)
        vel_std = torch.tensor(self.stats['vel_std'], device=x.device, dtype=x.dtype)
        x[..., 0:1] = x[..., 0:1] * prs_std + prs_mean
        x[..., 1:] = x[..., 1:] * vel_std + vel_mean
        return x

    def dump_stats(self, f):
        np.savez(f, **self.stats)

    def load_data(self, idx, start, end):
        # load file
        prs = np.load(os.path.join(self.f_lst[idx], 'prs.npy'), 'r')
        prs = prs[start:end, ::self.skip, ::self.skip, ::self.skip].copy()
        vx = np.load(os.path.join(self.f_lst[idx], 'vx.npy'), 'r')
        vy = np.load(os.path.join(self.f_lst[idx], 'vy.npy'), 'r')
        vz = np.load(os.path.join(self.f_lst[idx], 'vz.npy'), 'r')
        vx = vx[start:end, ::self.skip, ::self.skip, ::self.skip].copy()
        vy = vy[start:end, ::self.skip, ::self.skip, ::self.skip].copy()
        vz = vz[start:end, ::self.skip, ::self.skip, ::self.skip].copy()
        prs = (prs - self.stats['prs_mean']) / (self.stats['prs_std'] + 1e-6)
        vx = (vx - self.stats['vel_mean']) / (self.stats['vel_std'] + 1e-6)
        vy = (vy - self.stats['vel_mean']) / (self.stats['vel_std'] + 1e-6)
        vz = (vz - self.stats['vel_mean']) / (self.stats['vel_std'] + 1e-6)

        # flatten the spatial dimensions
        # cat the channel dimensions
        prs = prs.reshape(1, -1, self.resolution, self.resolution, self.resolution)  # [1, 21, 60, 60*60]
        vx = vx.reshape(1, -1, self.resolution, self.resolution, self.resolution)
        vy = vy.reshape(1, -1, self.resolution, self.resolution, self.resolution)
        vz = vz.reshape(1, -1, self.resolution, self.resolution, self.resolution)

        feat = np.concatenate([prs, vx, vy, vz], axis=0)  # [4, 20, 60, 60, 60]
        return feat

    def load_all_data(self):
        for idx in tqdm(self.idxs):
            feat = self.load_data(idx, 0, self.max_len)
            self.cache[idx] = feat

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]

        # randomly sample a start time step
        if self.train:
            if self.max_len - self.tw - self.max_latent_steps > 0:
                start = np.random.randint(0, self.max_len - self.tw - self.max_latent_steps)
            else:
                start = 0
            out_tw = self.max_latent_steps
        else:
            start = 0
            out_tw = self.out_tw
        # feat = self.cache[idx]
        feat = self.load_data(idx, start, start+self.tw+out_tw)
        feat_ = feat[:, :self.tw, ...]     # [c, t, n, n, n]
        feat_tsr = torch.from_numpy(feat_).float()
        timestep = np.arange(start, start+out_tw).reshape(-1)
        pred_ = feat[:, self.tw:self.tw+out_tw, ...]           # [c, n, n, n]
        pred_tsr = torch.from_numpy(pred_).float()
        pred_tsr = rearrange(pred_tsr, 'c t h w d -> t h w d c')

        t = timestep / self.out_tw  # training mode its [b,], test mode its [b, t]
        t = np.array(t) if self.train else t
        return feat_tsr, pred_tsr, t.astype(np.float32)

    def add_pushforward_steps(self, pushforward=1):
        # add the pushforward steps to the latent steps
        self.max_latent_steps *= (1 + pushforward)
        print(f'New max latent steps: {self.max_latent_steps}')


class NS3DTrainer(Trainer):
    def __init__(self,
                 config,
                 args,

                 ):
        super().__init__(config, args)

        # which device available?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build the model with config
        self.model = FactFormer3D(config.model).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        self.max_latent_steps = config.model.max_latent_steps
        self.curriculum_scheduler = CurriculumSampler(1,
                                                       self.max_latent_steps,
                                                       config.training.curriculum_length)

        # build the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.training.lr, weight_decay=1e-4)

        # build the dataset
        self.batch_size = config.data.batch_size

        self.train_dataset = Fluid3DData(config, train=True, max_len=20)
        self.test_dataset = Fluid3DData(config, train=False, max_len=20)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=10, shuffle=True, num_workers=0,
                                      pin_memory=False, )
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size, shuffle=True, num_workers=1,
                                       pin_memory=True, persistent_workers=True)

        self.denormalize = self.train_dataset.denormalize

        # build the loss function
        self.loss_fn = loss_wrapper
        self.test_loss_fn = test_loss_wrapper

        # self.xy_grid = torch.from_numpy(self.train_dataset.coords['pos']).float().to(self.device)
        if not self.args.testing:
            self.scheduler = OneCycleLR(self.optimizer,
                                        config.training.lr,
                                        epochs=config.training.epochs,
                                        steps_per_epoch=int(len(self.train_dataset) // config.data.batch_size),
                                        div_factor=config.training.lr_div_factor,
                                        pct_start=0.2,
                                        final_div_factor=config.training.lr_div_factor*10,
                                        )
            self.total_iter = int(len(self.train_dataset) // config.data.batch_size) * config.training.epochs
            print(f'Total training iterations:'
                  f' {self.total_iter}')
            self.n_iter = 0
            self.pushforward_after = config.training.pushforward_after
            self.pushforward_every = config.training.pushforward_every
            self.pushforward_flag = False

        # create logger and ensure log directory
        self.log_dir = config.log_dir
        # ensure the directory to log
        if os.path.exists(self.log_dir) \
                and (args.resume is False and args.testing is False):  # make sure not deleting exist log
            shutil.rmtree(self.log_dir)
            print('Log directory exists, removing it.')
        ensure_dir(self.log_dir)

        self.logger = logging.getLogger("LOG")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (self.log_dir, 'logging_info'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        if args.comment is not None:
            self.logger.info(args.comment)

        # dump the config
        with open(self.log_dir + '/config.yaml', 'w') as f:
            yaml.dump(config, f)

        # cache the current training script
        filename = os.path.basename(__file__)
        shutil.copy(__file__, self.log_dir + f'/{filename}')
        if not self.args.testing:

            # ensure the directory to save the model
            ensure_dir(self.log_dir + '/model')
            if self.config.training.dump_visualization:
                ensure_dir(self.log_dir + '/visualization')
            self.logger.info('=====================================')
            self.logger.info(f'Number of trainable parameters: {num_params}')
            self.logger.info(f'Total training iterations: {self.total_iter}')
            # print and log the info of curriculum and replay buffer
            self.logger.info(f'Max latent steps: {self.max_latent_steps}')
            self.logger.info(f'Curriculum length: {config.training.curriculum_length}')
            self.logger.info(f'Pushforward after: {config.training.pushforward_after}')
            self.logger.info('=====================================')

    def load_checkpoint(self, checkpoint_path):
        print(f'Resuming checkpoint from: {checkpoint_path}')
        self.logger.info(f'Resuming checkpoint from: {checkpoint_path}')

        ckpt = load_checkpoint(checkpoint_path)  # custom method for loading last checkpoint
        self.model.load_state_dict(ckpt['model'])
        if not self.args.testing:

            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])

            # TODO: set all scheduler to maximum, hardcoded for now
            self.pushforward_flag = True
            self.curriculum_scheduler._step = self.curriculum_scheduler.curriculum_length
            self.update_dataset()

            self.n_iter = ckpt['n_iter']
        print(f'Loaded checkpoint from: {checkpoint_path}')

    def save_checkpoint(self, postfix):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'n_iter': self.n_iter,
        }
        checkpoint_path = f'{self.log_dir}/model/checkpoint_{postfix}.pth'
        save_checkpoint(ckpt, checkpoint_path)

    def make_image_grid(self,
                        image: torch.Tensor,
                        out_path):
        # side by side comparison between original and reconstructed
        b, t, h, w = image.shape
        image = image.detach().cpu().numpy()
        image = image.reshape((b*t, h, w))
        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(b, t),  # creates 2x2 grid of axes
                         )

        for ax, im_no in zip(grid, np.arange(b*t)):
            # Iterating over the grid returns the Axes.
            ax.imshow(
                image[im_no].reshape((h, w)),
                cmap='twilight',
            )

            ax.axis('off')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    # put input/output to the model here
    def step_fn(self, data,
                train=True,
                current_latent_steps=None,
                pushforward=False):
        # torch.cuda.empty_cache()
        x, y, t_coord = data
        x = x.to(self.device)
        y = y.to(self.device)
        t_coord = t_coord.to(self.device)

        pos_x = torch.linspace(0, 2*np.pi, self.config.data.resolution).float().to(self.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 2*np.pi, self.config.data.resolution).float().to(self.device).unsqueeze(-1)
        pos_z = torch.linspace(0, 2*np.pi, self.config.data.resolution).float().to(self.device).unsqueeze(-1)
        pos_lst = [pos_x, pos_y, pos_z]

        # loss
        if train:
            b = x.shape[0]

            # t_step = self.curriculum_scheduler.get_value()
            if current_latent_steps is None:
                y_hat = self.model(x, pos_lst, 1)
                y = y[:, 0:1]
            elif not pushforward:
                y_hat = self.model(x, pos_lst, current_latent_steps)
                y = y[:, 0:current_latent_steps]
            else:
                with torch.no_grad():
                    x_hat = self.model(x, pos_lst, current_latent_steps)
                    x = torch.cat([x[:, :, current_latent_steps:],
                                   rearrange(
                                       x_hat,
                                       'b t h w d c -> b c t h w d')], dim=2)
                y_hat = self.model(x.detach(), pos_lst, current_latent_steps)
                y = y[:, current_latent_steps:current_latent_steps * 2]
            # denormalize
            y_hat = self.denormalize(y_hat)
            y = self.denormalize(y)
            loss, loss_dict = self.loss_fn(y_hat, y)

            return loss, loss_dict
        else:
            if y.shape[1] % self.max_latent_steps != 0:
                print(f'Warning: length of ground truth: '
                      f'{y.shape[1]} is not divisible by'
                      f'latent steps: {self.max_latent_steps}')
            y_hat = torch.zeros_like(y)  # b, t, h, w, c
            for i in range(y.shape[1] // self.max_latent_steps):
                y_hat[:, i * self.max_latent_steps:(i + 1) * self.max_latent_steps] =\
                    self.model.forward(x, pos_lst, latent_steps=self.max_latent_steps)
                x = torch.cat([x[:, :, self.max_latent_steps:],
                               rearrange(
                                   y_hat[:, i * self.max_latent_steps:(i + 1) * self.max_latent_steps],
                                   'b t h w d c -> b c t h w d'
                               )], dim=2)

            # denormalize
            y_hat = self.denormalize(y_hat)
            y = self.denormalize(y)
            test_loss = self.test_loss_fn(y_hat, y)
            return test_loss, y_hat, y

    @torch.no_grad()
    def testing_loop(self, epoch):
        self.model.eval()
        visualization_cache = {
            'output': [],
            'target': [],
        }
        losses = {}
        picked = 0
        for i, data in enumerate(tqdm(self.test_loader)):
            # sample test input always with a fixed ratio
            test_loss_dict, y_hat, y = self.step_fn(data, train=False)

            # loss
            for k, v in test_loss_dict.items():
                if k not in losses:
                    losses[k] = []
                losses[k].append(v)

            if picked < 4:
                idx = np.arange(0, min(4 - picked, y.shape[0]))
                # set masked values to np.nan
                # y = self.denormalize(y)
                # y_hat = self.denormalize(y_hat)

                # randomly pick a batch
                y = y[idx, ::2, y.shape[2]//2, ..., 1]
                y_hat = y_hat[idx, ::2, y.shape[2]//2, ..., 1]

                picked += y.shape[0]

                visualization_cache['output'] += [y_hat]
                visualization_cache['target'] += [y]

        # average loss
        test_loss = {k: np.mean(v) for k, v in losses.items()}
        if self.config.training.dump_visualization:
            # save visualization
            # denormalize
            visualization_cache = {k: torch.cat(v, dim=0) for k, v in visualization_cache.items()}
            # self.make_image_grid(visualization_cache['input'],
            #                      out_path=f'{self.log_dir}/visualization/input_{epoch}.png',
            #                      nrow=visualization_cache['input'].shape[1])
            self.make_image_grid(visualization_cache['output'],
                                 out_path=f'{self.log_dir}/visualization/output_{epoch}.png')
            self.make_image_grid(visualization_cache['target'],
                                 out_path=f'{self.log_dir}/visualization/target_{epoch}.png')

        return test_loss

    def training_loop(self):

        # train the model
        self.model.train()

        # tqdm progress bar
        with tqdm(range(self.config.training.epochs)) as pbar:
            if self.n_iter > 0:
                pbar.update(self.n_iter//len(self.train_loader))
            while True:
                if pbar.n % self.config.training.test_every == 0:
                    test_loss = self.testing_loop(pbar.n)
                    for k, v in test_loss.items():
                        print(f'Epoch {pbar.n}: {k} test loss {v:.4f}')
                        self.logger.info(f'Epoch {pbar.n} - {k} test loss {v:.4f}')
                if pbar.n % self.config.training.save_every == 0:
                    self.save_checkpoint(pbar.n)
                if pbar.n >= self.config.training.epochs or \
                    self.n_iter >= self.scheduler.total_steps:
                    return
                for data in self.train_loader:
                    # get the data
                    if self.n_iter > self.pushforward_after and self.n_iter % self.pushforward_every == 0:
                        loss, loss_dict = self.step_fn(data, train=True,
                                                       current_latent_steps=self.curriculum_scheduler.get_value(),
                                                       pushforward=True
                                                       )
                    else:
                        loss, loss_dict = self.step_fn(data, train=True,
                                                       current_latent_steps=self.curriculum_scheduler.get_value(),
                                                       pushforward=False
                                                       )

                    self.optimizer.zero_grad()
                    # self.emb_optimizer.zero_grad()

                    # backward
                    loss.backward()

                    # clip gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    self.optimizer.step()
                    # self.emb_optimizer.step()
                    self.scheduler.step()
                    self.curriculum_scheduler.step()

                    # update progress bar
                    desc = f'Epoch {pbar.n}/{self.config.training.epochs} ' + \
                           f' - Loss: {loss.item():.4f}' + \
                           f' - LR (1e-4): {self.optimizer.param_groups[0]["lr"]*1e4:.3f}' + \
                           f' - Latent steps: {self.curriculum_scheduler.get_value()}'
                    # add loss dict to desc
                    for k, v in loss_dict.items():
                        desc += f' - {k}: {v.item():.4f}'
                    pbar.set_description(desc)
                    self.n_iter += 1
                    if not self.pushforward_flag and self.n_iter > self.pushforward_after:
                        break
                if not self.pushforward_flag and self.n_iter > self.pushforward_after:
                    self.update_dataset()

                pbar.update(1)

    def update_dataset(self):
        self.pushforward_flag = True
        self.logger.info(f'Pushforward after {self.pushforward_after} iterations')
        self.logger.info(f'Updating the dataset')
        self.train_dataset.add_pushforward_steps(1)
        del self.train_loader
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True, persistent_workers=True)



def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--testing', action='store_true', help='Testing mode')
    parser.add_argument('--ckpt_to_resume', type=str, default=None, help='Checkpoint to resume')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    return args, config


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args, config = parse_args_and_config()
    set_random_seed(args.seed)

    if not (args.resume or args.testing):  # train from scratch
        trainer = NS3DTrainer(config, args)
        trainer.training_loop()
    elif args.resume:  # resume training
        print('Resuming training')
        assert args.ckpt_to_resume is not None, 'Please specify the checkpoint to resume'
        # check if the checkpoint exists
        assert os.path.exists(args.ckpt_to_resume), f'Checkpoint {args.ckpt_to_resume} does not exist'

        trainer = NS3DTrainer(config, args)
        trainer.load_checkpoint(args.ckpt_to_resume)
        trainer.training_loop()

    print('Running finished...')
    exit()
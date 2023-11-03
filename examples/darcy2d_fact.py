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

from libs.factorization_module import FABlock2D
from libs.positional_encoding_module import GaussianFourierFeatureTransform
from libs.basics import PreNorm, MLP, masked_instance_norm
from utils import Trainer, dict2namespace, index_points, load_checkpoint, save_checkpoint, ensure_dir
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from loss_fn import rel_l2_loss

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil
from collections import OrderedDict
import random
from scipy.io import loadmat


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
                GaussianFourierFeatureTransform(2, dim // 2, 1),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True,
                                   **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1)
        for pos_enc, attn_layer in self.layers:
            u += pos_enc(pos).view(1, nx, ny, -1)
            u = attn_layer(u, pos_lst) + u
        return u


class FactFormer2D(nn.Module):
    def __init__(self,
                 config
                 ):
        super().__init__()
        self.config = config
        # self.resolutions = config.resolutions   # hierachical resolutions, [16, 8, 4]
        # self.out_resolution = config.out_resolution

        self.in_dim = config.in_dim
        self.out_dim = config.out_dim

        self.dim = config.dim                 # dimension of the transformer
        self.depth = config.depth           # depth of the encoder transformer
        self.dim_head = config.dim_head
        self.reducer = config.reducer
        self.resolution = config.resolution

        self.heads = config.heads

        self.pos_in_dim = config.pos_in_dim
        self.pos_out_dim = config.pos_out_dim
        self.positional_embedding = config.positional_embedding
        self.kernel_multiplier = config.kernel_multiplier

        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True)

        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier)

        self.down_block = nn.Sequential(
            nn.InstanceNorm2d(self.dim),
            nn.Conv2d(self.dim, self.dim//2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True))

        self.up_block = nn.Sequential(
            nn.Upsample(size=(self.resolution, self.resolution), mode='nearest'),
            nn.Conv2d(self.dim//2, self.dim//2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.dim//2, self.dim, kernel_size=3, stride=1, padding=1, bias=True))

        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=8, num_channels=self.dim*2),
            nn.Conv1d(self.dim*2, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self,
                u,
                pos_lst,
                ):
        b, nx, ny, c = u.shape
        u = self.to_in(u)
        u_last = self.encoder(u, pos_lst)
        u = rearrange(u_last, 'b nx ny c -> b c nx ny')
        u = self.down_block(u)
        u = self.up_block(u)
        u = rearrange(u, 'b c nx ny -> b nx ny c')
        u = torch.cat([u, u_last], dim=-1)
        u = self.simple_to_out(u)
        u = rearrange(u, 'b c (nx ny) -> b nx ny c', nx=nx, ny=ny)
        return u


# def loss_wrapper(x, y, denormalize_fn, lam=0.1):
def loss_wrapper(x, y, lam=0.1):
    # x: b n n c
    # break down the loss for each channel
    l2norm = 0.
    loss_dict = {}
    l2norm = rel_l2_loss(x[..., 0], y[..., 0], dim=(1, 2), reduction='sum')
    loss_dict['error'] = l2norm.detach().clone()
    return l2norm, loss_dict


def test_loss_wrapper(x, y):

    return rel_l2_loss(x[..., 0], y[..., 0], dim=(1, 2), reduction='sum')


class Darcy2DData(Dataset):

    def __init__(self, config, train=True):

        self.resolution = config.data.resolution
        self.skip = int((421-1) // (self.resolution-1))   # maximum resolution: 421

        self.train = train
        self.train_num = config.data.train_num
        self.test_num = config.data.test_num
        if self.train:
            self.data_dir = config.data.train_data_dir
        else:
            self.data_dir = config.data.test_data_dir

        self.stats = {}
        self.load_all_data(self.data_dir)
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
        # print(f'Dataset stats: {self.stats}')

    def prepare_data(self):
        # load all training data and then calculate the mean/std online
        self.stats['x_mean'] = self.data_x.mean(axis=0)
        self.stats['x_std'] = self.data_x.std(axis=0) + 1e-5
        self.stats['y_mean'] = self.data_y.mean(axis=0)
        self.stats['y_std'] = self.data_y.std(axis=0) + 1e-5

    def denormalize(self, y: torch.Tensor, apply_dirichlet=True) -> torch.Tensor:
        # y: b n n c
        b, n, _, c = y.shape
        mean = torch.tensor(self.stats['y_mean'], device=y.device, dtype=y.dtype)
        std = torch.tensor(self.stats['y_std'], device=y.device, dtype=y.dtype)
        y = y * std.reshape(1, n, n, 1) + mean.reshape(1, n, n, 1)

        if apply_dirichlet:
            y = rearrange(y, 'b nx ny c -> b c nx ny')
            y = y[..., 1:-1, 1:-1].contiguous()
            y = F.pad(y, (1, 1, 1, 1), "constant", 0)
            y = rearrange(y, 'b c nx ny -> b nx ny c')
        return y

    def dump_stats(self, f):
        np.savez(f, **self.stats)

    def load_all_data(self, path):
        data = loadmat(path)
        if self.train:
            self.data_x = data['coeff'][:self.train_num, ::self.skip, ::self.skip][:, :self.resolution, :self.resolution]
            self.data_y = data['sol'][:self.train_num, ::self.skip, ::self.skip][:, :self.resolution, :self.resolution]
        else:
            self.data_x = data['coeff'][-self.test_num:, ::self.skip, ::self.skip][:, :self.resolution, :self.resolution]
            self.data_y = data['sol'][-self.test_num:, ::self.skip, ::self.skip][:, :self.resolution, :self.resolution]
        del data
        print('Loaded data from: ', path)
        print(f'Data shape: {self.data_x.shape, self.data_y.shape}')

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, idx):
        feat = self.data_x[idx, ...]  # [n, n]
        pred = self.data_y[idx, ...]  # [n, n]

        # normalize feat
        feat = (feat - self.stats['x_mean']) / self.stats['x_std']

        feat_tsr = torch.from_numpy(feat).float()
        pred_tsr = torch.from_numpy(pred).float()
        return feat_tsr.unsqueeze(-1), pred_tsr.unsqueeze(-1)


class DarcyTrainer(Trainer):
    def __init__(self,
                 config,
                 args,

                 ):
        super().__init__(config, args)

        # which device available?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        # build the model with config
        self.model = FactFormer2D(config.model).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")

        # build the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.training.lr,
                                           weight_decay=1e-4)

        # build the dataset
        self.batch_size = config.data.batch_size
        self.train_dataset = Darcy2DData(config, train=True)
        self.test_dataset = Darcy2DData(config, train=False)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=10, shuffle=False, num_workers=2,
                                      pin_memory=False, )
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size, shuffle=True, num_workers=2,
                                       pin_memory=True, persistent_workers=True)

        self.denormalize = self.train_dataset.denormalize

        # build the loss function
        self.loss_fn = loss_wrapper
        self.test_loss_fn = test_loss_wrapper

        self.n_iter = 0

        self.scheduler = OneCycleLR(self.optimizer,
                                    config.training.lr,
                                    epochs=config.training.epochs,
                                    steps_per_epoch=int(len(self.train_dataset) // config.data.batch_size),
                                    div_factor=config.training.lr_div_factor,
                                    pct_start=0.2,
                                    final_div_factor=config.training.lr_div_factor*10,
                                    )
        total_iters = int(len(self.train_dataset) // config.data.batch_size) * config.training.epochs
        print(f'Total training iterations:'
              f' {total_iters}')

        # create logger and ensure log directory
        self.log_dir = config.log_dir
        # ensure the directory to log
        if os.path.exists(self.log_dir) and not args.resume:
            shutil.rmtree(self.log_dir)
            print('Log directory exists, removing it.')
        ensure_dir(self.log_dir)
        # ensure the directory to save the model
        ensure_dir(self.log_dir + '/model')
        if self.config.training.dump_visualization:
            ensure_dir(self.log_dir + '/visualization')

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
        # copy all the files under nn_module
        shutil.copytree('nn_module', self.log_dir + '/nn_module')

    def load_checkpoint(self, checkpoint_path):
        print(f'Resuming checkpoint from: {checkpoint_path}')
        self.logger.info(f'Resuming checkpoint from: {checkpoint_path}')

        ckpt = load_checkpoint(checkpoint_path)  # custom method for loading last checkpoint
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

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
                        u_pred: torch.Tensor,
                        u_gt: torch.Tensor,
                        out_path,
                        nrow=8):
        b, h, w, c = u_pred.shape  # c = 1
        u_pred = u_pred.detach().cpu().squeeze().numpy()
        u_gt = u_gt.detach().cpu().squeeze().numpy()

        fig = plt.figure(figsize=(8., 8.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(b * 2 // nrow, nrow),  # creates 8x8 grid of axes
                         )

        for ax, im_no in zip(grid, np.arange(b * 2)):
            # Iterating over the grid returns the Axes.
            if im_no % 2 == 0:
                ax.imshow(u_pred[im_no // 2], cmap='coolwarm')
            elif im_no % 2 == 1:
                ax.imshow(u_gt[im_no // 2], cmap='coolwarm')

            ax.axis('equal')
            ax.axis('off')

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

    # put input/output to the model here
    def step_fn(self, data,
                train=True):
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        b, c = x.shape[0], x.shape[-1]

        pos_x = torch.linspace(0, 1, self.config.data.resolution).float().to(self.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 1, self.config.data.resolution).float().to(self.device).unsqueeze(-1)

        pos_lst = [pos_x, pos_y]

        # loss
        if train:
            # forward

            y_hat = self.model.forward(x, pos_lst)
            y_hat = self.denormalize(y_hat)
            loss, loss_dict = self.loss_fn(y_hat, y)

            return loss, loss_dict
        else:
            with torch.no_grad():
                y_hat = self.model.forward(x, pos_lst)
                y_hat = self.denormalize(y_hat)
                test_loss = self.test_loss_fn(y_hat, y)
                return test_loss, y_hat, y

    @torch.no_grad()
    def testing_loop(self, epoch):
        self.model.eval()
        visualization_cache = {
            'output': [],
            'target': [],
        }
        losses = []
        picked = 0
        for i, data in enumerate(tqdm(self.test_loader)):
            # sample test input always with a fixed ratio
            test_loss, y_hat, y = self.step_fn(data, train=False)

            # loss
            losses += [test_loss]

            if picked < 16:
                idx = np.arange(0, min(16 - picked, y.shape[0]))

                # randomly pick a batch
                y = y[idx]
                y_hat = y_hat[idx]

                picked += y.shape[0]

                visualization_cache['output'] += [y_hat]
                visualization_cache['target'] += [y]

        test_loss = torch.stack(losses).mean().item()
        if self.config.training.dump_visualization:
            # save visualization
            # denormalize
            visualization_cache = {k: torch.cat(v, dim=0) for k, v in visualization_cache.items()}

            self.make_image_grid(visualization_cache['output'], visualization_cache['target'],
                                 out_path=f'{self.log_dir}/visualization/output_{epoch}.png')

        return test_loss

    def training_loop(self):

        # train the model
        self.model.train()

        # tqdm progress bar
        with tqdm(range(self.config.training.epochs)) as pbar:
            if self.n_iter > 0:
                pbar.update(self.n_iter)
            while True:
                if pbar.n % self.config.training.test_every == 0:
                    test_loss = self.testing_loop(pbar.n)
                    print(f'Epoch {pbar.n}: test loss {test_loss}')
                    self.logger.info(f'Epoch {pbar.n} - Test loss: {test_loss:.4f}')
                if pbar.n % self.config.training.save_every == 0:
                    self.save_checkpoint(pbar.n)
                if pbar.n >= self.config.training.epochs:
                    self.save_checkpoint('final')
                    break
                for data in self.train_loader:
                    # get the data

                    loss, loss_dict = self.step_fn(data, train=True)

                    self.optimizer.zero_grad()

                    # backward
                    loss.backward()

                    # clip gradient
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    # update progress bar
                    desc = f'Epoch {pbar.n}/{self.config.training.epochs} ' + \
                           f' - Loss: {loss.item():.4f}' + \
                           f' - LR: {self.optimizer.param_groups[0]["lr"]:.6f}'

                    # add loss dict to desc
                    for k, v in loss_dict.items():
                        desc += f' - {k}: {v.item():.4f}'
                    pbar.set_description(desc)
                    self.n_iter += 1

                pbar.update(1)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=666, help='Random seed')
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
    # create the trainer
    if not args.resume:  # train from scratch
        trainer = DarcyTrainer(config, args)
        trainer.training_loop()
    else:  # resume training
        assert args.ckpt_to_resume is not None, 'Please specify the checkpoint to resume'
        # check if the checkpoint exists
        assert os.path.exists(args.ckpt_to_resume), f'Checkpoint {args.ckpt_to_resume} does not exist'

        trainer = DarcyTrainer(config, args)
        trainer.load_checkpoint(args.ckpt_to_resume)
        trainer.training_loop()

    print('Running finished...')
    exit()



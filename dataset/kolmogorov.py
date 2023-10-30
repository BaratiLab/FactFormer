# modified from: https://github.com/neuraloperator/physics_informed/blob/master/solver/kolmogorov_flow.py

import os

import torch
import math
import numpy as np

from random_fields import GaussianRF
from timeit import default_timer
import argparse


class KolmogorovFlow2d(object):

    def __init__(self, w0, Re, n):

        # Grid size

        self.s = w0.size()[-1]

        assert self.s == w0.size()[-2], "Grid must be uniform in both directions."

        assert math.log2(self.s).is_integer(), "Grid size must be power of 2."

        assert n >= 0 and isinstance(n, int), "Forcing number must be non-negative integer."

        assert n < self.s // 2 - 1, "Forcing number too large for grid size."

        # Forcing number
        self.n = n

        assert Re > 0, "Reynolds number must be positive."

        # Reynolds number
        self.Re = Re

        # Device
        self.device = w0.device

        # Current time
        self.time = 0.0

        # Current vorticity in Fourier space
        self.w_h = torch.fft.fft2(w0, norm="backward")

        # Wavenumbers in y and x directions
        self.k_y = torch.cat((torch.arange(start=0, end=self.s // 2, step=1, dtype=torch.float32, device=self.device), \
                              torch.arange(start=-self.s // 2, end=0, step=1, dtype=torch.float32, device=self.device)),
                             0).repeat(self.s, 1)

        self.k_x = self.k_y.clone().transpose(0, 1)

        # Negative inverse Laplacian in Fourier space
        self.inv_lap = (self.k_x ** 2 + self.k_y ** 2)
        self.inv_lap[0, 0] = 1.0
        self.inv_lap = 1.0 / self.inv_lap

        # Negative scaled Laplacian
        self.G = (1.0 / self.Re) * (self.k_x ** 2 + self.k_y ** 2)
        self.linear_term = self.G + 0.1   # drag

        # Dealiasing mask using 2/3 rule
        self.dealias = (self.k_x ** 2 + self.k_y ** 2 <= (self.s / 3.0) ** 2).float()
        # Ensure mean zero
        self.dealias[0, 0] = 0.0

    # Get current vorticity from stream function (Fourier space)
    def vorticity(self, stream_f=None, real_space=True):
        if stream_f is not None:
            w_h = self.Re * self.G * stream_f
        else:
            w_h = self.w_h

        if real_space:
            return torch.fft.irfft2(w_h, s=(self.s, self.s), norm="backward")
        else:
            return w_h

    # Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h=None, real_space=False):
        if w_h is None:
            psi_h = self.w_h.clone()
        else:
            psi_h = w_h.clone()

        # Stream function in Fourier space: solve Poisson equation
        psi_h = self.inv_lap * psi_h

        if real_space:
            return torch.fft.irfft2(psi_h, s=(self.s, self.s), norm="backward")
        else:
            return psi_h

    # Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f=None, real_space=True):
        if stream_f is None:
            stream_f = self.stream_function(real_space=False)

        # Velocity field in x-direction = psi_y
        q_h = stream_f * 1j * self.k_y

        # Velocity field in y-direction = -psi_x
        v_h = stream_f * -1j * self.k_x

        if real_space:
            q = torch.fft.irfft2(q_h, s=(self.s, self.s), norm="backward")
            v = torch.fft.irfft2(v_h, s=(self.s, self.s), norm="backward")
            return q, v
        else:
            return q_h, v_h

    # Compute non-linear term + forcing from given vorticity (Fourier space)
    def explicit_term(self, w_h):
        # Physical space vorticity
        w = torch.fft.ifft2(w_h, s=(self.s, self.s), norm="backward")

        # Velocity field in physical space
        q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        # Compute non-linear term
        t1 = torch.fft.fft2(q * w, s=(self.s, self.s), norm="backward")
        t1 = self.k_x * t1

        t2 = torch.fft.fft2(v * w, s=(self.s, self.s), norm="backward")
        t2 = self.k_y * t2

        nonlin = -1j * (t1 + t2)

        if self.n > 0:
            nonlin[..., 0, self.n] -= (float(self.n) / 2.0) * (self.s ** 2)
            nonlin[..., 0, -self.n] -= (float(self.n) / 2.0) * (self.s ** 2)

        return nonlin

    def implicit_term(self, w_h):
        return self.linear_term * w_h

    def implicit_solve(self, w_h, time_step):
        return 1 / (1 + time_step * self.linear_term) * w_h

    def advance(self, t, delta_t=1e-3):

        # Final time
        T = self.time + t

        # Advance solution in Fourier space
        while self.time < T:

            if self.time + delta_t > T:
                current_delta_t = T - self.time
            else:
                current_delta_t = delta_t

            # Inner-step of Heun's method

            g = self.w_h - 0.5 * current_delta_t * self.implicit_term(self.w_h)
            h1 = self.explicit_term(self.w_h)
            w_h_tilde = self.implicit_solve(g + current_delta_t*h1, 0.5*current_delta_t)

            # Cranck-Nicholson + Heun update

            h2 = 0.5 * (self.explicit_term(w_h_tilde) + h1)
            self.w_h = self.implicit_solve(g + current_delta_t*h2, 0.5*current_delta_t)
            # De-alias
            self.w_h *= self.dealias
            self.time += current_delta_t

            if torch.any(torch.isnan(self.w_h)):
                raise Exception('NaN in voriticlity field')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--re", type=float, default=1000.0)
    opt = parser.parse_args()

    device = torch.device('cuda:0')
    s = 2048
    sub = 8
    n = 8
    bsize = 5
    Re = opt.re

    T_in = 2.0
    T = 10
    t = 32
    dt = 1.0 / t

    print('Re = {}'.format(Re))
    print('dt = {}'.format(dt))
    print('batch size = {}'.format(bsize))

    os.makedirs(f'kf_2d_re{int(opt.re)}', exist_ok=True)

    for seed in range(120):
        np.random.seed(seed*666)
        GRF = GaussianRF(2, s, 2 * math.pi, alpha=2.5, tau=7, device=device)
        u0 = GRF.sample(bsize)  # batch size

        NS = KolmogorovFlow2d(u0, Re, n)
        NS.advance(T_in, delta_t=1e-4)
        print('seed:', seed)
        for i in range(T):
            t1 = default_timer()
            for j in range(t):
                NS.advance(dt, delta_t=1e-4)

                sol = NS.vorticity().cpu().numpy()

                for b in range(sol.shape[0]):
                    if not os.path.exists(f'kf_2d_re{int(opt.re)}/seed{seed*bsize+b}'):
                        os.makedirs(f'kf_2d_re{int(opt.re)}/seed{seed*bsize+b}')
                    np.save(os.path.join(f'kf_2d_re{int(opt.re)}',
                                         f'seed{seed*bsize+b}',
                                         f'sol_t{i}_step{j}'), sol[b, ::sub, ::sub])
            t2 = default_timer()
            print(i, t2 - t1)
        # sol_ini = sol[i, -1, :, :]

    # np.save('NS_fine_Re500_S512_s64_T500_t128.npy', sol)

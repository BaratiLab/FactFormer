import torch


def rel_l2_loss(x, y, dim=-2, eps=1e-5, reduction='sum', reduce_all=True):
    # y is the ground truth
    # first reduce wrt to grid point dimension
    # i.e. mesh weighted
    # return torch.mean(
    #         (x - y) ** 2 / (y ** 2 + 1e-6), dim=(-1, -2)).sqrt().mean()
    reduce_fn = torch.mean if reduction == 'mean' else torch.sum

    y_norm = reduce_fn((y ** 2), dim=dim)
    mask = y_norm < eps
    y_norm[mask] = eps
    diff = reduce_fn((x - y) ** 2, dim=dim)
    diff = diff / y_norm  # [b, c]
    if reduce_all:
        diff = diff.sqrt().mean()    # mean across channels and batch and any other dimensions
    else:
        diff = diff.sqrt()  # do nothing
    return diff


def rel_l1_loss(x, y):
    # y is the ground truth
    return torch.mean(
            torch.abs(x - y) / (torch.abs(y) + 1e-6))


def spatial_grad_fft(x):
    # use fft to calculate spatial gradient
    # assume x in [b t h w]
    nx = x.size(2)
    ny = x.size(3)
    device = x.device

    x_h = torch.fft.fft2(x, dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
        reshape(N, 1).repeat(1, N).reshape(1, 1, N, N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0). \
        reshape(1, N).repeat(N, 1).reshape(1, 1, N, N)
    # Negative Laplacian in Fourier space

    dx_h = 1j * k_x * x_h
    dy_h = 1j * k_y * x_h

    dx = torch.fft.irfft2(dx_h[..., :, :k_max + 1], dim=[2, 3])
    dy = torch.fft.irfft2(dy_h[..., :, :k_max + 1], dim=[2, 3])
    return dx, dy


def fdm_temporal_grad(x, dt=1):
    # assume x in [b t h w]
    return (x[..., 2:, :, :] - x[..., :-2, :, :]) / (2 * dt)



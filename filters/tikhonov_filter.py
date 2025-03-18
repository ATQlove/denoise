import numpy as np
import torch
from torch.nn.functional import conv3d, conv2d, pad
from cv2 import getGaussianKernel

device = torch.device('cuda')


def fft3d(x, s=None):
    if s is not None:
        return torch.fft.fftn(x, s=s, dim=[-3, -2, -1])
    else:
        return torch.fft.fftn(x, dim=[-3, -2, -1])


def ifft3d(x, s=None):
    if s is not None:
        return torch.fft.ifftn(x, s=s, dim=[-3, -2, -1])
    else:
        return torch.fft.ifftn(x, dim=[-3, -2, -1])


def back_diff(data, dim):
    if len(data.shape) == 3:
        if dim == 't': return pad((data[1:, :, :] - data[:-1, :, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 1, 0)).squeeze(1).squeeze(0)
        if dim == 'h': return pad((data[:, 1:, :] - data[:, :-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 1, 0, 0, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, :, 1:] - data[:, :, :-1]).unsqueeze(0).unsqueeze(0), (1, 0, 0, 0, 0, 0)).squeeze(1).squeeze(0)
    elif len(data.shape) == 2:
        if dim == 'h': return pad((data[1:, :] - data[:-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 1, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, 1:] - data[:, :-1]).unsqueeze(0).unsqueeze(0), (1, 0, 0, 0)).squeeze(1).squeeze(0)
    else:
        raise NotImplementedError


def forward_diff(data, dim):
    if len(data.shape) == 3:
        if dim == 't': return pad((data[1:, :, :] - data[:-1, :, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 0, 0, 1)).squeeze(1).squeeze(0)
        if dim == 'h': return pad((data[:, 1:, :] - data[:, :-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 1, 0, 0)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, :, 1:] - data[:, :, :-1]).unsqueeze(0).unsqueeze(0), (0, 1, 0, 0, 0, 0)).squeeze(1).squeeze(0)
    elif len(data.shape) == 2:
        if dim == 'h': return pad((data[1:, :] - data[:-1, :]).unsqueeze(0).unsqueeze(0), (0, 0, 0, 1)).squeeze(1).squeeze(0)
        if dim == 'w': return pad((data[:, 1:] - data[:, :-1]).unsqueeze(0).unsqueeze(0), (0, 1, 0, 0)).squeeze(1).squeeze(0)
    else:
        raise NotImplementedError


def shrink(x, gamma):
    if (isinstance(x, list) or isinstance(x, tuple)) and len(x) >= 2:
        qu = torch.abs(x[0]) ** 2
        for idx in range(1, len(x)):
            qu += torch.abs(x[idx]) ** 2
        norm_ = torch.sqrt(qu + 1e-30)
        temp = torch.nn.ReLU()(norm_ - gamma) / norm_
        return [x[idx] * temp for idx in range(len(x))]
    else:
        return torch.sign(x) * torch.nn.ReLU()(torch.abs(x) - gamma)


def conv(img, kernel, conj=False, back_pad=True):
    ndim = len(img.shape)
    if ndim == 2:
        [H, W] = kernel.shape
        H_, W_ = H - 1, W - 1
        if back_pad:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ - W_ // 2, W_ // 2, H_ - H_ // 2, H_ // 2], mode='replicate')  # circular replicate
        else:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ // 2, W_ - W_ // 2, H_ // 2, H_ - H_ // 2], mode='replicate')  # circular replicate
        if conj:
            kernel_ = kernel.unsqueeze(0).unsqueeze(0)
        else:
            kernel_ = torch.flip(kernel, dims=[0, 1]).unsqueeze(0).unsqueeze(0)
        return conv2d(img_, kernel_, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1).squeeze(0).squeeze(0)
    elif ndim == 3:
        [T, H, W] = kernel.shape
        T_, H_, W_ = T - 1, H - 1, W - 1
        if back_pad:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ - W_ // 2, W_ // 2, H_ - H_ // 2, H_ // 2, T_ - T_ // 2, T_ // 2], mode='replicate')
        else:
            img_ = pad(img.unsqueeze(0).unsqueeze(0), [W_ // 2, W_ - W_ // 2, H_ // 2, H_ - H_ // 2, T_ // 2, T_ - T_ // 2], mode='replicate')
        if conj:
            kernel_ = kernel.unsqueeze(0).unsqueeze(0)
        else:
            kernel_ = torch.flip(kernel, dims=[0, 1, 2]).unsqueeze(0).unsqueeze(0)
        return conv3d(img_, kernel_, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1).squeeze(0).squeeze(0)
    else:
        raise NotImplementedError


def psf2otf(psf, shape):
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[-2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=-2 + axis)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def make_gaussian_kernal(shape, sigma):
    [height, width] = list(shape)
    [height_sigma, width_sigma] = list(sigma)
    h = torch.from_numpy(getGaussianKernel(height, height_sigma))
    w = torch.from_numpy(getGaussianKernel(width, width_sigma))
    kernel = torch.mm(h, w.t()).float()
    kernel /= kernel.sum()
    return kernel


def tikhonov(g, otf=None, psf=None, para_fidelity=400.0, para_smooth_space=1.0, para_smooth_timepoint=2.0, iter_Bregman=100, tol=1e-12):
    # ----------------------------------------
    # data pre process
    # ----------------------------------------
    assert para_smooth_space > 0 or para_smooth_timepoint > 0
    if psf is not None or otf is not None: assert para_smooth_space > 0
    g = g.float().to(device)
    gshape = g.shape
    g = g.squeeze()
    if len(g.shape) == 2:
        g = g.unsqueeze(0)
        para_smooth_timepoint = 0

    assert len(g.shape) == 3, "input image doesnot have two or three vaild dims"
    if otf is None and psf is not None:
        if not isinstance(psf, torch.Tensor): psf = torch.from_numpy(psf)
        assert len(psf.shape) == 2
        psf = psf.float()
        psf /= psf.sum()
        otf = psf2otf(psf.unsqueeze(0).unsqueeze(0), g.shape[-2:]).to(device).squeeze()
    else:
        otf = torch.ones_like(g)

    mean_g = torch.mean(g, dim=[1, 2], keepdim=True)
    if mean_g.min() > 1e-8:
        mean_g = mean_g / g.mean()
    else:
        mean_g = torch.ones_like(mean_g)
    g /= mean_g

    gmax = torch.max(g)
    g /= gmax
    # ----------------------------------------
    # initialize
    # ----------------------------------------
    filter_laplace = torch.Tensor([[1, 1.5, 1], [1.5, -10, 1.5], [1, 1.5, 1]]).reshape(1, 3, 3).to(device)
    denominator = para_smooth_space * torch.abs(fft3d(filter_laplace, s=g.shape)) ** 2 + para_fidelity * torch.abs(otf) ** 2

    if para_smooth_timepoint > 0:
        filter_tt = torch.Tensor([1, -2, 1]).reshape(3, 1, 1).to(device)
        denominator += para_smooth_timepoint * torch.abs(fft3d(filter_tt, s=g.shape)) ** 2

    hg = torch.real(ifft3d(fft3d(g) * torch.conj(otf)))

    f_last = torch.zeros_like(g)

    # ----------------------------------------
    # iteration
    # ----------------------------------------
    numerator = None
    f = None
    for idx in range(iter_Bregman):
        if idx == 0:
            f = g.clone()
        else:
            f = torch.real(ifft3d(fft3d(numerator) / denominator))

            if (f_last - f).sum() ** 2 / f.numel() < tol: break
            f_last = f.clone()

        numerator = para_fidelity * hg


    # ----------------------------------------
    # data post process
    # ----------------------------------------
    f = torch.where(torch.isnan(f), torch.full_like(f, 0), f)
    f[f < 0] = 0
    f *= gmax
    f *= mean_g

    return f.reshape(gshape)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .smoothmax import smoothmax


def blur1d(x, sigma, std_devs=None):
    """1D blur with Gaussian."""
    f = gauss1d(sigma, std_devs)
    f = f.view(1, 1, -1)  # set out x in channel dims
    return conv1d_mono(x, f)


def blur2d_sphere(x, sigma, std_devs=None):
    """2D blur with spherical/isotropic Gaussian."""
    f = gauss1d(sigma, std_devs)
    f_y = f.view(1, 1, -1, 1)  # set out x in channel x h x w dims
    f_x = f_y.permute(0, 1, 3, 2)
    return conv2d_mono_sep(x, f_y, f_x)


def blur2d_diag(x, sigma_y, sigma_x, std_devs=None):
    """2D blur with diagonal/uncorrelated Gaussian."""
    # make 1D gaussians for y, x and shape for height, width respectively
    f_y = gauss1d(sigma_y, std_devs).view(1, 1, -1, 1)
    f_x = gauss1d(sigma_x, std_devs).view(1, 1, 1, -1)
    return conv2d_mono_sep(x, f_y, f_x)


def blur2d_full(x, cov, std_devs=None):
    """2D blur with full covariance Gaussian."""
    # TODO(shelhamer) make separable by whitening warp/unwarp
    f = gauss2d_full(cov, std_devs)
    f = f.view(1, 1, *f.size()[-2:])  # set out x in channel dims
    return conv2d_mono(x, f)


def blur2d_local_sphere(x, sigma, std_devs=None):
    """2D blur with local spherical/isotropic Gaussian at each point.
    Local filtering, not convolution: the filter varies with point."""
    f_1d = gauss1d(sigma.view(-1), std_devs)
    f = f_1d.unsqueeze(-1) @ f_1d.unsqueeze(1)
    f = f.unsqueeze(1)  # add channel dim
    return local2d_mono(x, f)


def blur2d_local_full(x, cov, std_devs=None):
    """2D blur with local full covariance Gaussian at each point.
    Local filtering, not convolution: the filter varies with point."""
    f = gauss2d_full(cov.view(-1, 3), std_devs)
    f = f.unsqueeze(1)  # add channel dim
    return local2d_mono(x, f)


def dog1d(x, sigma_center, sigma_surround, std_devs=None):
    """1D Difference of Gaussians for smaller center and larger surround."""
    assert sigma_center < sigma_surround
    # generate center and surround together to share coordinates and size
    sigmas = torch.stack([sigma_center, sigma_surround])
    f_center, f_surround = torch.unbind(gauss1d(sigmas, std_devs), 0)
    f_center, f_surround = f_center.contiguous(), f_surround.contiguous()
    x_center = conv1d_mono(x, f_center.view(1, 1, -1))
    x_surround = conv1d_mono(x, f_surround.view(1, 1, -1))
    return x_center - x_surround


def dog2d(x, sigma_center, sigma_surround, std_devs=None):
    """2D Difference of Gaussians for spherical Gaussians."""
    assert sigma_center < sigma_surround
    # generate center and surround together to share coordinates and size
    sigmas = torch.stack([sigma_center, sigma_surround])
    f_center, f_surround = torch.unbind(gauss1d(sigmas, std_devs), 0)
    f_center, f_surround = f_center.contiguous(), f_surround.contiguous()
    blurs = []
    for f in (f_center, f_surround):
        f_y = f.view(1, 1, -1, 1)  # set out x in channel x h x w dims
        f_x = f.view(1, 1, 1, -1)
        blurs.append(conv2d_mono_sep(x, f_y, f_x))
    x_center, x_surround = blurs
    return x_center - x_surround


def gauss1d(sigma, std_devs=2):
    std_devs = std_devs or 2
    # determine kernel size to cover +/- `std_devs` sigmas of the density.
    # default of 2 gives ~95% coverage.
    half_size = torch.ceil(sigma.max() * std_devs).clamp(min=1.).int()
    # always make the kernel odd to center coordinates
    kernel_size = half_size*2 + 1
    # calculate unnormalized density then normalize
    x = torch.linspace(-half_size, half_size, steps=kernel_size,
                       dtype=sigma.dtype, device=sigma.device)
    f = torch.exp(-x.view(-1, 1)**2 / (2*sigma**2)).t()
    f_norm = f / f.sum(1).view(-1, 1)
    return f_norm


def gauss2d_sphere(sigma, std_devs=2):
    std_devs = std_devs or 2
    # generate 1D, normalized Gaussian
    f_1d = gauss1d(sigma, std_devs)
    # for spherical/isotropic 2D is product of 1D
    f = f_1d.unsqueeze(-1) @ f_1d.unsqueeze(1)
    return f


def gauss2d_diag(sigma_y, sigma_x, std_devs=2):
    std_devs = std_devs or 2
    # make 1D gaussians for y, x and shape for height, width respectively
    f_y = gauss1d(sigma_y, std_devs)
    f_x = gauss1d(sigma_x, std_devs)
    # for diagonal 2D is product of 1D for y, x
    f = f_y.unsqueeze(-1) @ f_x.unsqueeze(1)
    return f


def gauss2d_full(cov, std_devs=2):
    std_devs = std_devs or 2
    if cov.dim() == 1:
        cov = cov.view(1, 3)
    # make cholesky factors to parameterize covariances and inverses
    Us = cov.new_zeros(cov.size(0), 2, 2)
    Us.view(-1, 4)[:, [0, 1 , 3]] = cov
    Us.diagonal(dim1=1, dim2=2).exp_()
    Us_inv = Us.inverse()
    sigmas = torch.bmm(Us.permute(0, 2, 1), Us)
    sigmas_inv = torch.bmm(Us_inv, Us_inv.permute(0, 2, 1))
    # determine kernel size to cover +/- 2 sigma such that. >95% of density
    # is included, and always make the kernel odd to center coordinates
    sigmax, _ = torch.max(sigmas.diagonal(dim1=-2, dim2=-1)**(0.5), 0)
    half_size = torch.ceil(sigmax * std_devs).clamp(min=1.).int()
    kernel_size = half_size*2 + 1
    # calculate unnormalized density then normalize
    y = torch.linspace(half_size[0], -half_size[0], steps=kernel_size[0],
                        dtype=cov.dtype, device=cov.device)
    x = torch.linspace(-half_size[1], half_size[1], steps=kernel_size[1],
                        dtype=cov.dtype, device=cov.device)
    coords = torch.stack(torch.meshgrid(y, x), dim=-1).view(-1, 2)
    coords = coords.expand(Us.size(0), *coords.size())
    dets = Us.diagonal(dim1=-2, dim2=-1).prod(1)
    dists = (torch.bmm(coords, sigmas_inv) * coords).sum(-1)
    f = dets.view(-1, 1)**-1. * torch.exp(-(0.5) * dists)
    # normalize and shape
    f_norm = f / f.sum(1).view(-1, 1)
    f_norm = f_norm.view(-1, *kernel_size)
    return f_norm


def conv1d_mono(x, f):
    """1D convolution sharing single channel (mono) filter across channels."""
    assert f.size(1) == 1
    # pack input channels into batch to share filter across channels,
    # filter, and then unpack.
    b, k, l = x.size()
    x = x.view(b * k, 1, l)
    x = F.conv1d(x, f, padding=f.size(-1) // 2)
    x = x.view(b, k, l)
    return x


def conv2d_mono(x, f):
    """2D convolution sharing single channel (mono) filter across channels."""
    assert f.size(1) == 1
    # pack input channels into batch to share filter across channels,
    # filter, and then unpack.
    b, k, h, w = x.size()
    x = x.view(b * k, 1, h, w)
    x = F.conv2d(x, f, padding=(f.size(2) // 2, f.size(3) // 2))
    x = x.view(b, k, h, w)
    return x


def conv2d_mono_sep(x, f_y, f_x):
    """Separable 2D convolution sharing 1D mono filters across channels."""
    assert f_y.size(1) == 1 and f_x.size(1) == 1
    # pack input channels into batch to share filter across channels,
    # filter, and then unpack.
    b, k, h, w = x.size()
    x = x.view(b * k, 1, h, w)
    # separable filtering: convolve k x k' as k x 1 then 1 x k'
    x = F.conv2d(x, f_y, padding=(f_y.size(2) // 2, 0))
    x = F.conv2d(x, f_x, padding=(0, f_x.size(3) // 2))
    x = x.view(b, k, h, w)
    return x


def local2d_mono(x, f):
    """Local filtering where the filter varies with batch, height, and width
    but is shared across channels."""
    assert f.size(1) == 1
    kh, kw = f.size()[-2:]
    # pack input channels into batch to share filter across channels,
    # filter, and then unpack.
    b, k, h, w = x.size()
    x = x.view(b * k, 1, h, w)
    # form input matrix by unrolling input, pulling out channel,
    # then stacking everything but the kernel into the batch
    in_mat = F.unfold(x, kernel_size=(kh, kw), padding=(kh // 2, kw // 2))
    in_mat = in_mat.view(b * k, kh * kw, h * w).permute(0, 2, 1)
    in_mat = in_mat.contiguous().view(x.numel(), 1, kh * kw)
    # form weight matrix by unrolling kernel and adding dummy dim for dot
    weight_mat = f.view(-1, kh * kw, 1).repeat(k, 1, 1)
    out_mat = torch.matmul(in_mat, weight_mat)
    return out_mat.view(b, k, h, w)


def logchol2sigma(params):
    """Instantiate sigma (2x2) from log-Cholesky parameters (3)."""
    U = torch.zeros((2, 2))
    U.view(-1)[[0, 1, 3]] = params  # fill upper triangle
    U.diagonal().exp_()
    return U.t() @ U


def sigma2logchol(sigma):
    """Convert sigma to log-Cholesky parameters."""
    sigma_chol = torch.cholesky(sigma, upper=True).contiguous().view(-1)
    logchol = torch.stack((torch.log(sigma_chol[0]),
                           sigma_chol[1],
                           torch.log(sigma_chol[-1])))
    return logchol


class Blur2d(nn.Module):
    """Blur by spherical Gaussian with learned covariance sigma."""

    def __init__(self, sigma):
        super().__init__()
        self.sigma = nn.Parameter(sigma)

    def forward(self, x):
        return blur2d_sphere(x, self.sigma)

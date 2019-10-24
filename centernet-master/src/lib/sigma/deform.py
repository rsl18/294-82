import torch
import torch.nn as nn

from def_conv import DeformConv, deform_conv


# y, x coordinates of the unit circle in left-right, top-down order
# to serve as the default for standard Gaussian coordinates.
UNIT_CIRCLE = torch.Tensor([-0.7071, -0.7071,
                             1, 0,
                             0.7071, 0.7071,
                             0, -1,
                             0, 0,
                             0, 1,
                             0.7071, -0.7071,
                             1, 0,
                             0.7071, 0.7071]).view(-1, 2)


def coords2d_gauss(num_steps, num_angles):
    """Approximate standard Gaussian by coordinates with resolution controlled
    by `num_steps` rings and `num_angles` angles."""
    eps = torch.tensor(0.025)  # two sigma coverage
    # decide resolution of radii and angles for approximation,
    # excluding the zero radii and the duplicate 2pi == 0 angle
    normal = torch.distributions.normal.Normal(torch.tensor(0.), 1.)
    radii = normal.icdf(torch.linspace(0.5, -eps + 1, steps=num_steps + 1))[1:]
    angles = torch.linspace(0., 2*math.pi, steps=num_angles + 1)[:-1]

    # pair up polar coordinates and turn into cartesian coordinates
    rho, theta = torch.meshgrid((radii, angles))
    y = rho * torch.sin(theta)
    x = rho * torch.cos(theta)
    coords = torch.cat((torch.zeros(1, 2),  # include center point
                        torch.stack((y, x), dim=-1).view(-1, 2)))
    return coords


class GaussSphereDeformConvPack(DeformConv):
    """Gaussian deformable convolution with spherical covariance for scale and
    convolution regressor."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, deformable_groups, bias)

        cov_dim = 1  # one scale parameter for spherical covariance
        self.conv_cov = nn.Conv2d(self.in_channels,
                cov_dim * self.deformable_groups, kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding, bias=True)
        self.conv_cov.weight.data.zero_()
        self.conv_cov.bias.data.zero_()

        # standard Gaussian coordinates to warp by covariance
        self.standard_offset = UNIT_CIRCLE.clone().float()
        # shape for scale broadcasting and deform conv argument
        self.standard_offset = self.standard_offset.view(1, -1, 1, 1)

    def forward(self, input):
        b, _, h, w = input.size()
        # make Gaussian coords by transforming standard Gaussian by covariances
        offset = self.standard_offset.to(input.device) * self.conv_cov(input)
        return deform_conv(input, offset, self.weight, self.stride,
                           self.padding, self.dilation, self.groups,
                           self.deformable_groups)


class GaussFullDeformConvPack(DeformConv):
    """Gaussian deformable convolution with full covariance
    for scale/aspect/orientation and convolution regressor."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super().__init__( in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, deformable_groups, bias)

        cov_dim = 3 # 3 covariance parameters for full
        self.conv_cov = nn.Conv2d(self.in_channels,
                cov_dim * self.deformable_groups, kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding, bias=True)
        self.conv_cov.weight.data.zero_()
        self.conv_cov.bias.data.zero_()

        # standard Gaussian coordinates to warp by covariance
        self.standard_offset = UNIT_CIRCLE.clone().float()

    def forward(self, input):
        cov_params = self.conv_cov(input)
        b, _, h, w = cov_params.size()
        cov_params = cov_params.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        # instantiate sigmas through log-Cholesky params
        Us = input.new_zeros(cov_params.size(0), 2, 2)
        Us.view(-1, 4)[:, [0, 1 , 3]] = cov_params
        Us.diagonal(dim1=1, dim2=2).exp_()
        sigmas = torch.bmm(Us.permute(0, 2, 1), Us)
        # transform standard offsets by covariances
        offset = self.standard_offset.to(input.device)
        offset = torch.bmm(offset.repeat(sigmas.size(0), 1, 1), sigmas)
        offset = offset.view(b, -1, h, w)
        return deform_conv(input, offset, self.weight, self.stride,
                           self.padding, self.dilation, self.groups,
                           self.deformable_groups)

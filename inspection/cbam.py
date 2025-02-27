import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """
    A basic convolutional block with optional batch normalization and ReLU activation.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding for the convolution. Defaults to 0.
        dilation (int, optional): Dilation rate for the convolution. Defaults to 1.
        groups (int, optional): Number of groups for the convolution. Defaults to 1.
        relu (bool, optional): Whether to apply ReLU activation. Defaults to True.
        bn (bool, optional): Whether to apply batch normalization. Defaults to True.
        bias (bool, optional): Whether to include bias in the convolution. Defaults to False.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        """
        Forward pass of the BasicConv block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """
    A simple module to flatten the input tensor.
    """
    def forward(self, x):
        """
        Flatten the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened tensor.
        """
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """
    A channel attention gate module.

    Args:
        gate_channels (int): Number of channels in the input tensor.
        reduction_ratio (int, optional): Reduction ratio for the channel attention. Defaults to 16.
        pool_types (list, optional): List of pooling types to use. Defaults to ['avg', 'max'].
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        """
        Forward pass of the ChannelGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying channel attention.
        """
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    """
    Compute the log-sum-exp of a tensor along the last two dimensions.

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Log-sum-exp of the input tensor.
    """
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    results = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return results

class ChannelPool(nn.Module):
    """
    A module to pool the channels of the input tensor.
    """
    def forward(self, x):
        """
        Pool the channels of the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Pooled tensor.
        """
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    """
    A spatial attention gate module.
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        """
        Forward pass of the SpatialGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying spatial attention.
        """
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Args:
        gate_channels (int): Number of channels in the input tensor.
        reduction_ratio (int, optional): Reduction ratio for the channel attention. Defaults to 16.
        pool_types (list, optional): List of pooling types to use. Defaults to ['avg', 'max'].
        no_spatial (bool, optional): Whether to skip spatial attention. Defaults to False.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        """
        Forward pass of the CBAM module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying channel and spatial attention.
        """
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .cbam import ChannelGate, SpatialGate

# Define a convolutional layer with optional split mode
class ConvLayer(torch.nn.Module):
    """
    A convolutional layer with optional split mode for handling multiple input channels.

    Args:
        in_chans (int or list): Number of input channels. Can be a list for split mode.
        out_chans (int or list): Number of output channels. Can be a list for split mode.
        conv_mode (str): Mode of convolution. Options: "normal" or "split".
        kernel_size (int): Size of the convolutional kernel.
    """
    def __init__(self, in_chans=768, out_chans=512, conv_mode="normal", kernel_size=3):
        super().__init__()
        self.conv_mode = conv_mode

        if conv_mode == "normal":
            # Standard convolutional layer
            self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        elif conv_mode == "split":
            # Split convolutional layer for handling multiple input channels
            self.convs = nn.ModuleList()
            for j in range(len(in_chans)):
                conv = nn.Conv2d(in_chans[j], out_chans[j], kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
                self.convs.append(conv)

            # Define cut points for splitting the input channels
            self.cut = [0 for i in range(len(in_chans)+1)]
            self.cut[0] = 0
            for i in range(1, len(in_chans)+1):
                self.cut[i] = self.cut[i - 1] + in_chans[i-1]

    def forward(self, x):
        """
        Forward pass of the ConvLayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after convolution.
        """
        if self.conv_mode == "normal":
            # Apply standard convolution
            x = self.conv(x)
        elif self.conv_mode == "split":
            # Apply split convolution
            results = []
            for j in range(len(self.cut)-1):
                input_map = x[:, self.cut[j]:self.cut[j+1]]
                result_map = self.convs[j](input_map)
                results.append(result_map)
            x = torch.cat(results, dim=1)
        return x


# Define a lightweight attention network (LANet)
class LANet(torch.nn.Module):
    """
    Lightweight Attention Network (LANet) for channel-wise attention.

    Args:
        in_chans (int): Number of input channels.
        reduction_ratio (float): Reduction ratio for the attention mechanism.
    """
    def __init__(self, in_chans=512, reduction_ratio=2.0):
        super().__init__()

        self.in_chans = in_chans
        self.mid_chans = int(self.in_chans / reduction_ratio)

        # Define convolutional layers for attention mechanism
        self.conv1 = nn.Conv2d(self.in_chans, self.mid_chans, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(self.mid_chans, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x):
        """
        Forward pass of the LANet.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying attention.
        """
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


# Define a function for Masked Attention Dropout (MAD)
def MAD(x, p=0.6):
    """
    Masked Attention Dropout (MAD) function.

    Args:
        x (Tensor): Input tensor.
        p (float): Dropout probability.

    Returns:
        Tensor: Output tensor after applying MAD.
    """
    B, C, W, H = x.shape

    # Generate random masks
    mask1 = torch.cat([torch.randperm(C).unsqueeze(dim=0) for j in range(B)], dim=0).cuda()
    mask2 = torch.rand([B, C]).cuda()
    ones = torch.ones([B, C], dtype=torch.float).cuda()
    zeros = torch.zeros([B, C], dtype=torch.float).cuda()
    mask = torch.where(mask1 == 0, zeros, ones)
    mask = torch.where(mask2 < p, mask, ones)

    # Apply masks
    x = x.permute(2, 3, 0, 1)
    x = x.mul(mask)
    x = x.permute(2, 3, 0, 1)
    return x


# Define a network of LANets
class LANets(torch.nn.Module):
    """
    A network of LANets for multi-branch attention.

    Args:
        branch_num (int): Number of branches.
        feature_dim (int): Dimension of the feature space.
        la_reduction_ratio (float): Reduction ratio for LANets.
        MAD (function): MAD function for attention dropout.
    """
    def __init__(self, branch_num=2, feature_dim=512, la_reduction_ratio=2.0, MAD=MAD):
        super().__init__()

        self.LANets = nn.ModuleList()
        for i in range(branch_num):
            self.LANets.append(LANet(in_chans=feature_dim, reduction_ratio=la_reduction_ratio))

        self.MAD = MAD
        self.branch_num = branch_num

    def forward(self, x):
        """
        Forward pass of the LANets.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying multi-branch attention.
        """
        B, C, W, H = x.shape

        results = []
        for lanet in self.LANets:
            result = lanet(x)
            results.append(result)

        LANets_result = torch.cat(results, dim=1)

        if self.MAD and self.branch_num != 1:
            LANets_result = self.MAD(LANets_result)

        mask = torch.max(LANets_result, dim=1).values.reshape(B, 1, W, H)
        x = x.mul(mask)
        return x


# Define a feature attention network (FAM)
class FeatureAttentionNet(torch.nn.Module):
    """
    Feature Attention Network (FAM) for combining convolutional and attention mechanisms.

    Args:
        in_chans (int): Number of input channels.
        feature_dim (int): Dimension of the feature space.
        kernel_size (int): Size of the convolutional kernel.
        conv_shared (bool): Whether to share convolutional layers.
        conv_mode (str): Mode of convolution. Options: "normal" or "split".
        channel_attention (str): Type of channel attention. Options: "CBAM" or None.
        spatial_attention (str): Type of spatial attention. Options: "CBAM", "LANet", or None.
        pooling (str): Type of pooling. Options: "max" or "avg".
        la_branch_num (int): Number of branches for LANet.
    """
    def __init__(self, in_chans=768, feature_dim=512, kernel_size=3,
                 conv_shared=False, conv_mode="normal",
                 channel_attention=None, spatial_attention=None,
                 pooling="max", la_branch_num=2):
        super().__init__()

        self.conv_shared = conv_shared
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

        if not self.conv_shared:
            if conv_mode == "normal":
                self.conv = ConvLayer(in_chans=in_chans, out_chans=feature_dim,
                                      conv_mode="normal", kernel_size=kernel_size)
            elif conv_mode == "split" and in_chans == 2112:
                self.conv = ConvLayer(in_chans=[192, 384, 768, 768], out_chans=[47, 93, 186, 186],
                                      conv_mode="split", kernel_size=kernel_size)

        if self.channel_attention == "CBAM":
            self.channel_attention = ChannelGate(gate_channels=feature_dim)

        if self.spatial_attention == "CBAM":
            self.spatial_attention = SpatialGate()
        elif self.spatial_attention == "LANet":
            self.spatial_attention = LANets(branch_num=la_branch_num, feature_dim=feature_dim)

        if pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(num_features=feature_dim, eps=2e-5)

    def forward(self, x):
        """
        Forward pass of the FAM.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying convolution and attention mechanisms.
        """
        if not self.conv_shared:
            x = self.conv(x)

        if self.channel_attention:
            x = self.channel_attention(x)

        if self.spatial_attention:
            x = self.spatial_attention(x)

        x = self.act(x)
        B, C, _, __ = x.shape
        x = self.pool(x).reshape(B, C)
        x = self.norm(x)
        return x


# Define a feature attention module (FAM) with multiple branches
class FeatureAttentionModule(torch.nn.Module):
    """
    Feature Attention Module (FAM) with multiple branches for different tasks.

    Args:
        branch_num (int): Number of branches.
        in_chans (int): Number of input channels.
        feature_dim (int): Dimension of the feature space.
        conv_shared (bool): Whether to share convolutional layers.
        conv_mode (str): Mode of convolution. Options: "normal" or "split".
        kernel_size (int): Size of the convolutional kernel.
        channel_attention (str): Type of channel attention. Options: "CBAM" or None.
        spatial_attention (str): Type of spatial attention. Options: "CBAM", "LANet", or None.
        la_num_list (list): List of branch numbers for LANet.
        pooling (str): Type of pooling. Options: "max" or "avg".
    """
    def __init__(self, branch_num=11, in_chans=2112, feature_dim=512, conv_shared=False, conv_mode="split", kernel_size=3,
                 channel_attention="CBAM", spatial_attention=None, la_num_list=[2 for j in range(11)], pooling="max"):
        super().__init__()

        self.conv_shared = conv_shared
        if self.conv_shared:
            if conv_mode == "normal":
                self.conv = ConvLayer(in_chans=in_chans, out_chans=feature_dim,
                                      conv_mode="normal", kernel_size=kernel_size)
            elif conv_mode == "split" and in_chans == 2112:
                self.conv = ConvLayer(in_chans=[192, 384, 768, 768], out_chans=[47, 93, 186, 186],
                                      conv_mode="split", kernel_size=kernel_size)

        self.nets = nn.ModuleList()
        for i in range(branch_num):
            net = FeatureAttentionNet(in_chans=in_chans, feature_dim=feature_dim,
                                      conv_shared=conv_shared, conv_mode=conv_mode, kernel_size=kernel_size,
                                      channel_attention=channel_attention, spatial_attention=spatial_attention,
                                      la_branch_num=la_num_list[i], pooling=pooling)
            self.nets.append(net)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for the module.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the FAM.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying feature attention mechanisms.
        """
        if self.conv_shared:
            x = self.conv(x)

        results = []
        for net in self.nets:
            result = net(x).unsqueeze(dim=0)
            results.append(result)
        results = torch.cat(results, dim=0)
        return results


# Define a task-specific subnet
class TaskSpecificSubnet(torch.nn.Module):
    """
    Task-specific subnet for processing features.

    Args:
        feature_dim (int): Dimension of the feature space.
        drop_rate (float): Dropout rate.
    """
    def __init__(self, feature_dim=512, drop_rate=0.5):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        """
        Forward pass of the task-specific subnet.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after processing.
        """
        return self.feature(x)


# Define a collection of task-specific subnets
class TaskSpecificSubnets(torch.nn.Module):
    """
    Collection of task-specific subnets for multiple tasks.

    Args:
        branch_num (int): Number of branches.
    """
    def __init__(self, branch_num=11):
        super().__init__()

        self.branch_num = branch_num
        self.nets = nn.ModuleList()
        for i in range(self.branch_num):
            net = TaskSpecificSubnet(drop_rate=0.5)
            self.nets.append(net)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights for the module.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the task-specific subnets.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after processing.
        """
        results = []
        for i in range(self.branch_num):
            net = self.nets[i]
            result = net(x[i]).unsqueeze(dim=0)
            results.append(result)
        results = torch.cat(results, dim=0)
        return results


# Define a result module for generating final outputs
class resultModule(torch.nn.Module):
    """
    Result module for generating final outputs for different tasks.

    Args:
        feature_dim (int): Dimension of the feature space.
        result_type (str): Type of result. Options: "Dict", "List", "Attribute", or specific task name.
    """
    def __init__(self, feature_dim=512, result_type="Dict"):
        super().__init__()
        self.result_sizes = [[2],
                             [1, 2],
                             [7, 2],
                             [2 for j in range(6)],
                             [2 for j in range(10)],
                             [2 for j in range(5)],
                             [2, 2],
                             [2 for j in range(4)],
                             [2 for j in range(6)],
                             [2, 2],
                             [2, 2]]

        self.result_fcs = nn.ModuleList()
        for i in range(0, len(self.result_sizes)):
            for j in range(len(self.result_sizes[i])):
                result_fc = nn.Linear(feature_dim, self.result_sizes[i][j])
                self.result_fcs.append(result_fc)

        self.task_names = [
            'Age', 'Attractive', 'Blurry', 'Chubby', 'Heavy Makeup', 'Gender', 'Oval Face', 'Pale Skin',
            'Smiling', 'Young',
            'Bald', 'Bangs', 'Black Hair', 'Blond Hair', 'Brown Hair', 'Gray Hair', 'Receding Hairline',
            'Straight Hair', 'Wavy Hair', 'Wearing Hat',
            'Arched Eyebrows', 'Bags Under Eyes', 'Bushy Eyebrows', 'Eyeglasses', 'Narrow Eyes', 'Big Nose',
            'Pointy Nose', 'High Cheekbones', 'Rosy Cheeks', 'Wearing Earrings', 'Sideburns',
            r"Five O'Clock Shadow", 'Big Lips', 'Mouth Slightly Open', 'Mustache',
            'Wearing Lipstick', 'No Beard', 'Double Chin', 'Goatee', 'Wearing Necklace',
            'Wearing Necktie', 'Expression', 'Recognition']  # Total:43

        self.result_type = result_type

        self.apply(self._init_weights)

    def set_result_type(self, result_type):
        """
        Set the type of result.

        Args:
            result_type (str): Type of result.
        """
        self.result_type = result_type

    def _init_weights(self, m):
        """
        Initialize weights for the module.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, embedding):
        """
        Forward pass of the result module.

        Args:
            x (Tensor): Input tensor.
            embedding (Tensor): Embedding tensor.

        Returns:
            Tensor or dict: Output tensor or dictionary of results.
        """
        results = []

        k = 0
        for i in range(0, len(self.result_sizes)):
            for j in range(len(self.result_sizes[i])):
                result_fc = self.result_fcs[k]
                result = result_fc(x[i])
                results.append(result)
                k += 1

        [gender,
         age, young,
         expression, smiling,
         attractive, blurry, chubby, heavy_makeup, oval_face, pale_skin,
         bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline, straight_hair, wavy_hair,
         wearing_hat,
         arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses, narrow_eyes,
         big_nose, pointy_nose,
         high_cheekbones, rosy_cheeks, wearing_earrings, sideburns,
         five_o_clock_shadow, big_lips, mouth_slightly_open, mustache, wearing_lipstick, no_beard,
         double_chin, goatee,
         wearing_necklace, wearing_necktie] = results

        results = [age, attractive, blurry, chubby, heavy_makeup, gender, oval_face, pale_skin, smiling, young,
                   bald, bangs, black_hair, blond_hair, brown_hair, gray_hair, receding_hairline,
                   straight_hair, wavy_hair, wearing_hat,
                   arched_eyebrows, bags_under_eyes, bushy_eyebrows, eyeglasses, narrow_eyes, big_nose,
                   pointy_nose, high_cheekbones, rosy_cheeks, wearing_earrings,
                   sideburns, five_o_clock_shadow, big_lips, mouth_slightly_open, mustache,
                   wearing_lipstick, no_beard, double_chin, goatee, wearing_necklace,
                   wearing_necktie, expression]  # Total:42

        results.append(embedding)

        result = dict()
        for j in range(43):
            result[self.task_names[j]] = results[j]

        if self.result_type == "Dict":
            return result
        elif self.result_type == "List":
            return results
        elif self.result_type == "Attribute":
            return results[1: 41]
        else:
            return result[self.result_type]


# Define a model box for combining different components
class ModelBox(torch.nn.Module):
    """
    Model box for combining different components.

    Args:
        backbone (nn.Module): Backbone network.
        fam (nn.Module): Feature attention module.
        tss (nn.Module): Task-specific subnet.
        om (nn.Module): Output module.
        feature (str): Type of feature. Options: "global", "local", or "all".
        result_type (str): Type of result. Options: "Dict", "List", "Attribute", or specific task name.
    """
    def __init__(self, backbone=None, fam=None, tss=None, om=None,
                 feature="global", result_type="Dict"):
        super().__init__()
        self.backbone = backbone
        self.fam = fam
        self.tss = tss
        self.om = om
        self.result_type = result_type
        if self.om:
            self.om.set_result_type(self.result_type)

        self.feature = feature

    def set_result_type(self, result_type):
        """
        Set the type of result.

        Args:
            result_type (str): Type of result.
        """
        self.result_type = result_type
        if self.om:
            self.om.set_result_type(self.result_type)

    def forward(self, x):
        """
        Forward pass of the model box.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor or dict: Output tensor or dictionary of results.
        """
        local_features, global_features, embedding = self.backbone(x)

        if self.feature == "all":
            x = torch.cat([local_features, global_features], dim=1)
        elif self.feature == "global":
            x = global_features
        elif self.feature == "local":
            x = local_features

        x = self.fam(x)
        x = self.tss(x)

        x = self.om(x, embedding)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = input + res * self.res_scale
        return x

class PDEM(nn.Module):
    def __init__(self, dim):
        super(PDEM, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)  # 降维
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)  # 升维

    def forward(self, x, y, Phi, PhiT):
        x_pixel = self.conv1(x)  # 降维
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32)  # 前向感知变换
        delta = y - Phix  # 误差
        x_pixel = F.conv2d(delta, PhiT, padding=0)  # 反向变换
        x_pixel = nn.PixelShuffle(32)(x_pixel)  # 上采样
        x_delta = self.conv2(x_pixel)  # 升维回通道
        return x_delta

class FGDM(nn.Module):
    def __init__(self, dim):
        super(FGDM, self).__init__()
        self.conv_in = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.res1 = ResidualBlock(dim, dim, kernel_size=3)
        self.res2 = ResidualBlock(dim, dim, kernel_size=3)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv_out(x)
        return x


class DDOM(nn.Module):
    def __init__(self, dim):
        super(DDOM, self).__init__()
        self.pdem = PDEM(dim)
        self.fgdm = FGDM(dim)
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)

    def forward(self, x, y, Phi, PhiT):
        x_delta = self.pdem(x, y, Phi, PhiT) + x  # PDEM 输出 + 原图
        x_feat = self.fgdm(x)                      # FGDM 输出
        x_concat = torch.cat([x_feat, x_delta], dim=1)
        x = self.fusion_conv(x_concat)
        return x




class PMM(nn.Module):
    def __init__(self):
        super(PMM, self).__init__()
        self.soft_thr = nn.Parameter(torch.ones(16))  # 软阈值参数

        # Conv → ResidualBlock → Conv
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(32, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        # Conv → ResidualBlock → Conv
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x_input = x

        # 前半段
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.conv2(x)

        # Soft-thresholding
        std = torch.std(x, dim=[2, 3], keepdim=True) + 1e-6
        adjusted_thr = self.soft_thr.view(1, -1, 1, 1) * std
        x = torch.sign(x) * F.relu(torch.abs(x) - adjusted_thr)

        # 后半段
        x = F.relu(self.conv3(x))
        x = self.res2(x)
        x = self.conv4(x)

        # 最终残差连接
        return x + x_input


class DDOASNet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(DDOASNet, self).__init__()
        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)

        layer1 = []
        layer2 = []
        self.LayerNo = LayerNo
        for i in range(self.LayerNo):
            layer1.append(PMM())
            layer2.append(DDOM(self.base))
        self.fcs1 = nn.ModuleList(layer1)
        self.fcs2 = nn.ModuleList(layer2)

    def forward(self, x):
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32)
        x = F.conv2d(y, PhiT, padding=0)
        x = nn.PixelShuffle(32)(x)
        x = self.conv1(x)

        for i in range(self.LayerNo):
            x = self.fcs2[i](x, y, Phi, PhiT)
            x = self.fcs1[i](x)

        x = self.conv2(x)
        phi_cons = torch.mm(self.Phi, self.Phi.t())
        return x, phi_cons







"""
所有模型基代码
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):
        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = input + res * self.res_scale
        return x


class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        # 第一条路径：残差路径
        self.conv_in = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)  # 输入卷积
        self.conv_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)  # 输出卷积
        self.res1 = ResidualBlock(dim, dim, kernel_size=3)
        self.res2 = ResidualBlock(dim, dim, kernel_size=3)

        # 第二条路径：x_delta
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)  # 降维卷积
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)  # 升维卷积

        # 融合路径
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x, y, Phi, PhiT):
        # 第一条路径：残差路径
        x_pre = self.conv_in(x)  # 输入预处理
        x_res = self.res1(x_pre)
        x_res = self.res2(x_res)
        x_post = self.conv_out(x_res)  # 输出特征精炼

        # 第二条路径：x_delta
        x_pixel = self.conv1(x)  # 降维
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32)  # 感知矩阵变换
        delta = y - Phix  # 残差计算
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0))  # 反变换
        x_delta = self.conv2(x_pixel)  # 升维

        # 增加简单残差连接
        x_delta = x_delta + x  # 残差连接：将输入直接加到 x_delta

        # 通道拼接
        x_concat = torch.cat([x_post, x_delta], dim=1)

        # 融合路径
        x = self.fusion_conv(x_concat)

        return x

class TraBlock(nn.Module):
    def __init__(self):
        super(TraBlock, self).__init__()
        self.soft_thr = nn.Parameter(torch.ones(16))  # 软阈值参数

        # 定义前向路径的卷积权重
        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 16, 3, 3)))  # 输入16，输出32
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))  # 输入32，输出32
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 32, 3, 3)))  # 输入32，输出16

        # 定义反向路径的卷积权重
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 16, 3, 3)))  # 输入16，输出32
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))  # 输入32，输出32
        self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 32, 3, 3)))  # 输入32，输出16

    def forward(self, x):
        x_input = x

        # 前向路径
        x = F.conv2d(x_input, self.conv1_forward, padding=1)  # 输入16，输出32
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_forward, padding=1)  # 输入32，输出32
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv3_forward, padding=1)  # 输入32，输出16

        # 动态调整阈值
        std = torch.std(x_forward, dim=[2, 3], keepdim=True) + 1e-6
        adjusted_thr = self.soft_thr.view(1, -1, 1, 1) * std
        x = torch.sign(x_forward) * F.relu(torch.abs(x_forward) - adjusted_thr)

        # 反向路径
        x = F.conv2d(x, self.conv1_backward, padding=1)  # 输入16，输出32
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_backward, padding=1)  # 输入32，输出32
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv3_backward, padding=1)  # 输入32，输出16

        # 残差连接
        x = x_input + x_backward
        return x




class FSOINet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(FSOINet, self).__init__()
        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)

        layer1 = []
        layer2 = []
        self.LayerNo = LayerNo
        for i in range(self.LayerNo):
            layer1.append(TraBlock())
            layer2.append(GradBlock(self.base))
        self.fcs1 = nn.ModuleList(layer1)
        self.fcs2 = nn.ModuleList(layer2)

    def forward(self, x):
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32)
        x = F.conv2d(y, PhiT, padding=0)
        x = nn.PixelShuffle(32)(x)
        x = self.conv1(x)

        for i in range(self.LayerNo):
            x = self.fcs2[i](x, y, Phi, PhiT)
            x = self.fcs1[i](x)

        x = self.conv2(x)
        phi_cons = torch.mm(self.Phi, self.Phi.t())
        return x, phi_cons
"""



"""
软阈值参数
class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        # 第一条路径：残差路径
        self.conv_in = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)  # 输入卷积
        self.conv_out = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True)  # 输出卷积
        self.res1 = ResidualBlock(32, 32, kernel_size=3)
        self.res2 = ResidualBlock(32, 32, kernel_size=3)

        # 第二条路径：x_delta
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)  # 降维卷积
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)  # 升维卷积

        # 融合路径
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x, y, Phi, PhiT):
        # 第一条路径：多通道处理
        x_pre = self.conv_in(x)  # 输入预处理
        x_res = self.res1(x_pre)
        x_res = self.res2(x_res)
        x_post = self.conv_out(x_res)  # 输出特征精炼

        # 第二条路径：单通道处理
        x_pixel = self.conv1(x)  # 降维
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32)  # 感知矩阵变换
        delta = y - Phix  # 残差计算
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0))  # 反变换
        x_delta = self.conv2(x_pixel)  # 升维

        # 增加简单残差连接
        x_delta = x_delta + x  # 残差连接：将输入直接加到 x_delta

        # 通道拼接
        x_concat = torch.cat([x_post, x_delta], dim=1)

        # 融合路径
        x = self.fusion_conv(x_concat)

        return x

class TraBlock(nn.Module):
    def __init__(self):
        super(TraBlock, self).__init__()
        self.soft_thr = nn.Parameter(torch.ones(16))
        self.debug_data = {
            'soft_thr': None,
            'std': None,
            'adjusted_thr': None
        }

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 16, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 16, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 32, 3, 3)))

    def forward(self, x):
        x_input = x
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv3_forward, padding=1)

        std = torch.std(x_forward, dim=[2, 3], keepdim=True) + 1e-6
        adjusted_thr = self.soft_thr.view(1, -1, 1, 1) * std

        self.debug_data.update({
            'soft_thr': self.soft_thr.detach(),
            'std': std.detach(),
            'adjusted_thr': adjusted_thr.detach()
        })

        x = torch.sign(x_forward) * F.relu(torch.abs(x_forward) - adjusted_thr)

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv2_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv3_backward, padding=1)

        return x_input + x_backward


class FSOINet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(FSOINet, self).__init__()
        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)

        self.LayerNo = LayerNo
        self.fcs1 = nn.ModuleList([TraBlock() for _ in range(LayerNo)])
        self.fcs2 = nn.ModuleList([GradBlock(self.base) for _ in range(LayerNo)])

    def inspect_layers(self):

        debug_info = []
        for idx, tra_block in enumerate(self.fcs1):
            layer_info = {
                'layer_idx': idx + 1,
                'soft_thr': tra_block.debug_data['soft_thr'],
                'std': tra_block.debug_data['std'],
                'adjusted_thr': tra_block.debug_data['adjusted_thr']
            }
            debug_info.append(layer_info)
        return debug_info

    def forward(self, x):
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32)
        x = F.conv2d(y, PhiT, padding=0)
        x = nn.PixelShuffle(32)(x)
        x = self.conv1(x)

        for i in range(self.LayerNo):
            x = self.fcs2[i](x, y, Phi, PhiT)
            x = self.fcs1[i](x)

        x = self.conv2(x)
        phi_cons = torch.mm(self.Phi, self.Phi.t())
        return x, phi_cons


"""
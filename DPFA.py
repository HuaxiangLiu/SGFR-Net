import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
import numpy as np



class StarReLU(nn.Module):
    """
    StarReLU 激活函数：s * (ReLU(x))^2 + b
    s 和 b 可以是可学习参数或固定常数。
    这里实现为可学习参数，并设置了论文推荐的初始值。
    """

    def __init__(self, scale_init=0.8944, bias_init=-0.4472):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.scale * (self.relu(x) ** 2) + self.bias

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # 将默认激活函数改为 StarReLU
    default_act = StarReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        c = self.conv(x)
        c = self.bn(c)
        c = self.act(c)
        return c

class FAConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, kernel_num=4, act_fn=StarReLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.kernel_num = kernel_num
        self.act_fn = act_fn

        # 确保中间通道数至少为1
        mid_channels = max(1, in_channels // reduction)

        # 注意力模块，用于生成频率注意力权重
        self.attention_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()  # 生成 [0,1] 范围的注意力权重
        )

        # 1x1 卷积核，用于通道混合
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        orig_dtype = x.dtype  # 保存原始数据类型

        # 生成频率注意力权重 (B, C, 1, 1)
        freq_attn = self.attention_module(x)

        # 确保使用 float32 进行 FFT 操作
        x = x.to(torch.float32)
        freq_attn = freq_attn.to(torch.float32)

        # 对输入特征图进行 FFT，转换到频域
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # 将注意力权重应用于频域特征的实部和虚部
        x_freq_real = x_freq.real * freq_attn
        x_freq_imag = x_freq.imag * freq_attn
        x_freq_modulated = torch.complex(x_freq_real, x_freq_imag)

        # 逆 FFT 转换回空间域
        x_modulated = torch.fft.irfft2(x_freq_modulated, s=(h, w), norm='ortho')

        # 转换回原始数据类型
        x_modulated = x_modulated.to(orig_dtype)

        # 应用 1x1 卷积进行通道混合
        output = self.conv(x_modulated)
        output = self.bn(output)
        output = self.act_fn(output)

        return output

class DPFA(nn.Module):


    def __init__(self, channels, rep_scale=4, reduction=4, kernel_num=4):
        super(DPFA, self).__init__()
        # 全局路径: AvgPool -> DynamicAttentionConv -> Sigmoid
        self.global_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            FAConv(channels, channels, reduction=reduction, kernel_num=kernel_num),
            nn.Sigmoid()
        )

        # 局部路径: MaxPool -> DynamicAttentionConv -> Sigmoid
        self.local_path = nn.Sequential(
            FAConv(1, channels, reduction=reduction, kernel_num=kernel_num),  # 输入通道数为1
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局特征提取
        A_g = self.global_path(x)  #

        # 全局加权特征
        W_g = x * A_g

        # 局部特征提取 (在通道维度取最大值)
        max_out, _ = torch.max(W_g, dim=1, keepdim=True)  #
        A_l = self.local_path(max_out)  #

        # 双路径注意力融合
        attention = A_g * A_l
        return attention * x




# --- 测试模块 ---
if __name__ == '__main__':
    print("--- 测试 DPFA ---")
    x = torch.randn(8, 64, 32, 32)
    channels_hdpa = x.shape[1]
    model = DPFA(channels=channels_hdpa, rep_scale=4)
    output_dpfa = model(x)
    print(f"DPFA 输入形状: {x.shape}")
    print(f"DPFA 输出形状: {output_dpfa.shape}")
    params_dpfa = sum(p.numel() for p in model.parameters())
    print(f"DPFA 总参数量: {params_dpfa / 1e6:.2f}M")
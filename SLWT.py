import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange


def normalization(channels: int):
    return nn.InstanceNorm2d(num_features=channels)

class Splitting_row(nn.Module):
    def __init__(self):
        super(Splitting_row, self).__init__()

    def forward(self, x):
        x_even_row = x[:, :, ::2, :]
        x_odd_row = x[:, :, 1::2, :]
        return x_even_row, x_odd_row

class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        if dim < num_heads:
            num_heads = 1

        self.num_heads = num_heads
        head_dim = dim // num_heads

        if head_dim == 0:
            head_dim = 1
            num_heads = 1
            self.num_heads = num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=4, padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=4, padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=4, padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        self.topk = topk

    def forward(self, x, y):
        if x.size(1) < self.num_heads:
            return x

        original_size = (y.size(2), y.size(3))

        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        y = y1 + y2 + y3

        y = y.flatten(-2, -1)
        y = y.transpose(1, 2)
        y = self.layer_norm(y)

        x_down = F.avg_pool2d(x, kernel_size=4, stride=4)
        x = rearrange(x_down, 'b c h w -> b (h w) c')

        B, N1, C = y.shape
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        chunk_size = 128
        attn_chunks = []
        for i in range(0, N1, chunk_size):
            k_chunk = k[:, :, i:i + chunk_size, :]
            attn_chunk = torch.einsum('bhid,bhjd->bhij', q, k_chunk)
            attn_chunks.append(attn_chunk)

        attn = torch.cat(attn_chunks, dim=-1) * self.scale

        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        hw = int(sqrt(N))
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw, w=hw)

        x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)

        return x

class LiftingScheme2D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size // 2, kernel_size - 1 - kernel_size // 2)
        self.P_row1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.U_row1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.P_row2 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.U_row2 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )

        self.P_col1 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.U_col1 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.P_col2 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )
        self.U_col2 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1,kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
            )


        self.ll_attention = MSC(dim=channels)
        self.lh_attention = MSC(dim=channels)
        self.hl_attention = MSC(dim=channels)
        self.hh_attention = MSC(dim=channels)

    def split_row(self, x):
        B, C, H, W = x.shape
        x_even_row = x[:, :, ::2, :]
        x_odd_row = x[:, :, 1::2, :]
        return x_even_row, x_odd_row

    def split_col(self, x):
        B, C, H, W = x.shape
        x_even_col = x[:, :, :, ::2]
        x_odd_col = x[:, :, :, 1::2]
        return x_even_col, x_odd_col


    def forward(self, x):
        x_even_row, x_odd_row = self.split_row(x)

        # 第一组处理
        P_row1_out = self.P_row1(x_even_row)
        d1 = x_odd_row - P_row1_out

        U_row1_out = self.U_row1(d1)
        c1 = x_even_row + U_row1_out

        c1_e, c1_o = self.split_col(c1)
        d1_e, d1_o = self.split_col(d1)

        P_col1_c_out = self.P_col1(c1_e)
        d_col1_c = c1_o - P_col1_c_out

        U_col1_c_out = self.U_col1(d_col1_c)
        c_col1_c = c1_e + U_col1_c_out
        LL1 = c_col1_c
        LH1 = d_col1_c

        P_col1_d_out = self.P_col1(d1_e)
        d_col1_d = d1_o - P_col1_d_out

        U_col1_d_out = self.U_col1(d_col1_d)
        c_col1_d = d1_e + U_col1_d_out
        HL1 = c_col1_d
        HH1 = d_col1_d

        # 第二组处理
        P_row2_out = self.P_row2(x_odd_row)
        d2 = x_even_row - P_row2_out

        U_row2_out = self.U_row2(d2)
        c2 = x_odd_row + U_row2_out

        c2_e, c2_o = self.split_col(c2)
        d2_e, d2_o = self.split_col(d2)

        P_col2_c_out = self.P_col2(c2_e)
        d_col2_c = c2_o - P_col2_c_out

        U_col2_c_out = self.U_col2(d_col2_c)
        c_col2_c = c2_e + U_col2_c_out
        LL2 = c_col2_c
        LH2 = d_col2_c

        P_col2_d_out = self.P_col2(d2_e)
        d_col2_d = d2_o - P_col2_d_out

        U_col2_d_out = self.U_col2(d_col2_d)
        c_col2_d = d2_e + U_col2_d_out
        HL2 = c_col2_d
        HH2 = d_col2_d

        LL = self.ll_attention(LL1, LL2)
        LH = self.lh_attention(LH1, LH2)
        HL = self.hl_attention(HL1, HL2)
        HH = self.hh_attention(HH1, HH2)
        return LL, LH, HL, HH

class InverseLiftingScheme2D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = (kernel_size // 2, kernel_size - 1 - kernel_size // 2)

        # 第一组列方向算子
        self.P_col1 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )
        self.U_col1 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )

        # 第一组行方向算子
        self.P_row1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )
        self.U_row1 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )

        # 第二组列方向算子
        self.P_col2 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )
        self.U_col2 = nn.Sequential(
            nn.ReflectionPad2d((pad[0], pad[1], 0, 0)),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )

        # 第二组行方向算子
        self.P_row2 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )
        self.U_row2 = nn.Sequential(
            nn.ReflectionPad2d((0, 0, pad[0], pad[1])),
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), stride=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm2d(channels)
        )

    def merge_col(self, even, odd):
        B, C, H, W = even.size()
        x = torch.zeros((B, C, H, 2 * W), device=even.device)
        x[:, :, :, ::2] = even
        x[:, :, :, 1::2] = odd
        return x

    def merge_row(self, even, odd):
        B, C, H, W = even.size()
        x = torch.zeros((B, C, 2 * H, W), device=even.device)
        x[:, :, ::2, :] = even
        x[:, :, 1::2, :] = odd
        return x

    def forward(self, LL, LH, HL, HH):
        # 第一组逆列重建
        c1_even_col = LL - self.U_col1(LH)
        c1_odd_col = LH + self.P_col1(c1_even_col)

        d1_even_col = HL - self.U_col1(HH)
        d1_odd_col = HH + self.P_col1(d1_even_col)

        c1 = self.merge_col(c1_even_col, c1_odd_col)
        d1 = self.merge_col(d1_even_col, d1_odd_col)

        # 第一组逆行重建
        x_even_row1 = c1 - self.U_row1(d1)
        x_odd_row1 = d1 + self.P_row1(x_even_row1)

        # 第二组逆列重建
        c2_even_col = LL - self.U_col2(LH)
        c2_odd_col = LH + self.P_col2(c2_even_col)

        d2_even_col = HL - self.U_col2(HH)
        d2_odd_col = HH + self.P_col2(d2_even_col)

        # 合并列
        c2 = self.merge_col(c2_even_col, c2_odd_col)
        d2 = self.merge_col(d2_even_col, d2_odd_col)

        # 第二组逆行重建
        x_even_row2 = c2 - self.U_row2(d2)
        x_odd_row2 = d2 + self.P_row2(x_even_row2)

        x_even_row = (x_even_row1 + x_even_row2) / 2
        x_odd_row = (x_odd_row1 + x_odd_row2) / 2

        # 合并行
        x = self.merge_row(x_even_row, x_odd_row)

        return x

if __name__ == '__main__':
    x = torch.randn(8, 1, 512, 512)
    print(f"Input shape: {x.shape}")

    DWT = LiftingScheme2D(1, 4)
    LL, LH, HL, HH = DWT(x)

    print(f"Output shapes: LL={LL.shape}, LH={LH.shape}, HL={HL.shape}, HH={HH.shape}")

    IWT = InverseLiftingScheme2D(1, 4)
    result = IWT(LL, LH, HL, HH)

    print(f"Reconstructed shape: {result.shape}")
    print(f"Reconstruction diff: {torch.abs(x - result).mean().item()}")
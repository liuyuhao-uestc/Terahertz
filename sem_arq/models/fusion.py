import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    输入:
      z_seq:   (B, T, 3, H, W)
      m_seq:   (B, T, 1, H, W)   1=valid, 0=lost
      snr_seq: (B, T, 1, 1, 1)   可选，若没有可全0
    输出:
      z_fuse:  (B, 3, H, W)
    """
    def __init__(self, d_model=32, heads=4):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0
        self.dk = d_model // heads

        # 每轮每个位置的输入特征：[3(latent) + 1(mask) + 1(snr)] = 5
        self.in_proj = nn.Linear(3, d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, 3)

    def forward(self, z_receive):
        B, T, C, H, W = z_receive.shape

        x = z_receive.permute(0, 3, 4, 1, 2)          # B H W T C=3

        #x = torch.cat([z, m, s], dim=-1)          # B H W T 5
        x = self.in_proj(x)                       # B H W T d_model

        # Q K V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 多头: (B,H,W,T,heads,dk) -> (B,H,W,heads,T,dk)
        def split_heads(t):
            return t.view(B, H, W, T, self.heads, self.dk).permute(0,1,2,4,3,5)

        q_h, k_h, v_h = split_heads(q), split_heads(k), split_heads(v)

        # attention score: (.., T, dk) x (.., dk, T) -> (.., T, T)
        attn = torch.matmul(q_h, k_h.transpose(-1, -2)) / (self.dk ** 0.5)

        # 利用 mask 屏蔽丢失帧：m=0 的位置不参与 key/value
        # m_seq: (B,T,1,H,W) -> (B,H,W,1,T) broadcast 到 heads 和 queryT
        #key_valid = m_seq.squeeze(2).permute(0,2,3,1).unsqueeze(3).unsqueeze(3)  # B H W 1 1 T
        #attn = attn.masked_fill(key_valid == 0, float('-inf'))

        w = F.softmax(attn, dim=-1)    # 权重在时间维归一化
        out = torch.matmul(w, v_h)      # (B,H,W,heads,T,dk)

        # 合并 heads -> (B,H,W,T,d_model)
        out = out.permute(0,1,2,4,3,5).contiguous().view(B, H, W, T, self.d_model)

        # 这里我们取 “最后一轮” 的 query 对应输出（也可以取平均）
        out_last = out[:, :, :, -1, :]       # (B,H,W,d_model)
        z_fuse = self.out_proj(out_last)     # (B,H,W,3)

        return z_fuse.permute(0,3,1,2).contiguous()  # (B,3,H,W)
class RefineNet(nn.Module):
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, z):
        return z + self.net(z)   # 残差细化
class FusionNet(nn.Module):
    def __init__(self, ch=32, d_model=32, heads=4):
        super().__init__()
        self.attention = Attention(d_model, heads)
        self.refine = RefineNet(ch)
    def forward(self, z_receive):
        z_fused = self.attention(z_receive)
        z_fused = self.refine(z_fused)
        return z_fused
if __name__ == "__main__":
    attention = Attention()
    refine = RefineNet()

    z_received_1 = torch.randn(48, 1, 3, 64, 64)
    z_received_2 = torch.randn(48, 1, 3, 64, 64)
    z_received = torch.cat((z_received_1, z_received_2), dim=1)
    z_fused = attention(z_received)
    print(f"attention后: 均值{z_fused.mean().item():.4f}，方差{z_fused.std().item():.4f}")
    print(f"attention后平均功率：{torch.mean(z_fused ** 2, dim=(0, 1, 2, 3), keepdim=True)}")
    z_fused = refine(z_fused)
    print(f"refine后: 均值{z_fused.mean().item():.4f}，方差{z_fused.std().item():.4f}")
    print(f"refine后平均功率：{torch.mean(z_fused ** 2, dim=(0, 1, 2, 3), keepdim=True)}")
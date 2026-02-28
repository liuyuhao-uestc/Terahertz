import torch
import torch.nn as nn

##input：(B, num_packet, packet_size)
class Channel(nn.Module):
    def __init__(self,snr_db,plr):
        super(Channel, self).__init__()
        self.snr_db = snr_db
        self.plr = plr

    ### 加高斯白噪声
    def awgn_channel(self,x):
        sig_pwr = torch.mean(x ** 2, dim=(1,2,3), keepdim=True) #(B,C,H,W)
        snr = 10 ** (self.snr_db / 10)
        noi_pwr = sig_pwr/ snr #snr = 10^(snr_db/10)
        noise = torch.randn_like(x) * torch.sqrt(noi_pwr)
        return x + noise

    ### 随机丢包
    def loss_packet(self,x):
        # B, C=3,W=64,H=64  = x.size()
        mask = torch.rand_like(x) < self.plr
        x[mask] = 1e-9
        return x

    def forward(self,x):
        x_noise = self.awgn_channel(x)
        x_received = self.loss_packet(x_noise)
        return x_received
if __name__ == "__main__":
    channel = Channel(snr_db=100,plr=0.1)

    z_noise = torch.randn(48,3,64,64) # [batch, 3, 64, 64]
    print(f"z_noise: 均值{z_noise.mean().item():.4f}，方差{z_noise.std().item():.4f}")
    print(f"z_noise平均功率：{torch.mean(z_noise ** 2, dim=(0,1,2,3), keepdim=True)}")

    z_awgn = channel.awgn_channel(z_noise)
    print(f"z_awgn: 均值{z_awgn.mean().item():.4f}，方差{z_awgn.std().item():.4f}")
    print(f"z_awgn平均功率：{torch.mean(z_awgn ** 2, dim=(0, 1, 2, 3), keepdim=True)}")

    z_received = channel.loss_packet(z_awgn)
    print(f"z_received: 均值{z_received.mean().item():.4f}，方差{z_received.std().item():.4f}")
    print(f"z_receibed平均功率：{torch.mean(z_received ** 2, dim=(0, 1, 2, 3), keepdim=True)}")

    count = (z_received == 1e-9).sum().item()
    print(f"占总元素比例: {count / z_received.numel():.4f}")

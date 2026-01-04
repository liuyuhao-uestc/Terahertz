import torch
import torch.nn as nn

##input：(B, num_packet, packet_size)
class Channel(nn.Module):
    def __init__(self,snr,plr):
        super(Channel, self).__init__()
        self.snr = snr
        self.plr = plr

    ### 加高斯白噪声
    def awgn_channel(self,x):
        sig_pwr = torch.mean(x ** 2, dim=(1,2), keepdim=True) #(B,num_packet, 1)
        noi_pwr=sig_pwr*(10**(self.snr/10)) #snr = 10^(snr_db/10)
        noise = torch.randn_like(x) * torch.sqrt(noi_pwr)
        return x + noise

    ### 随机丢包
    def loss_packet(self,x):
        # B, num_packet, packet_size = x.size()
        # mask = (torch.rand(B, num_packet, device=x.device) > self.plr).float().view(B, num_packet, 1)
        mask = torch.randn_like(x) < self.plr
        x[mask] = 1e-9
        return x

    def forward(self,x):
        x_noise = self.awgn_channel(x)
        x_received = self.loss_packet(x_noise)
        return x_received

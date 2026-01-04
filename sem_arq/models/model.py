from omegaconf import OmegaConf
import torch
import torch.nn as nn

from sem_arq.modules.channel import Channel
from sem_arq.utils.instantiate import load_model
from sem_arq.models.transformer import PositionalEncoding, FlashCrossAttention
from sem_arq.models.transformer import Transformer
from sem_arq.modules.interleave import Interleaver

## latent input: [B,C,W,H] = [48,3,64,64]
## linear map: [48,64*64 = 4096,3-->d_model]
class my_model(nn.Module):
    def __init__(self,
                 d_model = 512,
                 n_heads = 8,
                 config_ldm ='models/ldm/celeba/config.yaml',
                 ckpt_ldm='models/ldm/celeba/model.ckpt',
                 ):
        super(my_model,self).__init__()
        #系统参数
        self.d_model = d_model
        self.n_heads = n_heads

        self.channel_1 = Channel(snr=35 , plr=0.35)
        self.channel_2 = Channel(snr=15 , plr=0.08)
        self.channel_3 = Channel(snr=5 , plr=0.01)

        self.norm_wide = nn.LayerNorm(d_model)
        self.norm_mid = nn.LayerNorm(d_model)
        self.norm_narrow = nn.LayerNorm(d_model)

        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm2_3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(d_model*2, d_model),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(d_model, 3),
            nn.Dropout(p=0.1),
        )

        # ldm加载
        config_ldm = OmegaConf.load(config_ldm)
        self.ldm, self.ldm_gl = load_model(config_ldm, ckpt_ldm, 0, 0)
        self.ldm = self.ldm.eval()
        for p in self.ldm.parameters():
            p.requires_grad = False

        #线性层，位置编码，attention
        self.linear_map = nn.Linear(in_features=3, out_features=d_model)
        self.positional_encoding = PositionalEncoding(d_model = self.d_model,
                                                      max_len = 64 * 64)
        '''self.attention_layer = nn.MultiheadAttention(embed_dim = self.d_model,
                                                     num_heads = self.n_heads,
                                                     dropout = 0.1,
                                                     batch_first = True
                                                     )'''
        self.attention_layer = FlashCrossAttention(d_model = self.d_model,
                                                   n_heads = self.n_heads,
                                                   dropout = 0.1,
                                                   )
        #attn_output, attn_weights = attention_layer(query=x, key=x, value=a)

    #image-->vq-->z(latent)-->交织-->信道-->解码
    def recon_1(self, x_wide, x_wide_norm):
        # attention
        residual = x_wide
        x_hat = x_wide_norm #Pre-LN
        x_hat, _ = self.attention_layer(query=x_hat, key=x_hat, value=x_wide, need_weights=False) #attenion
        x_hat = x_hat + residual #残差

        #ffn
        residual = x_hat
        x_hat = self.norm2_1(x_hat) #Pre_LN
        x_hat = self.ffn(x_hat) #ffn
        x_hat = residual + x_hat #残差
        return x_hat

    def recon_2(self, x_wide, x_wide_norm, x_mid_norm):
        # attention
        residual = x_wide
        x_hat, _ = self.attention_layer(query=x_wide_norm, key=x_mid_norm, value=x_mid_norm, need_weights=False)  # attenion
        x_hat = x_hat + residual  # 残差

        # ffn
        residual = x_hat
        x_hat = self.norm2_2(x_hat)  # Pre_LN
        x_hat = self.ffn(x_hat)  # ffn
        x_hat = residual + x_hat  # 残差
        return x_hat

    def recon_3(self, x_wide, x_wide_norm, x_mid_norm, x_narrow_norm):
        # attention
        residual = x_wide
        x_kv = torch.concat((x_mid_norm,x_narrow_norm), dim=1)
        x_hat, _ = self.attention_layer(query=x_wide_norm, key=x_kv, value=x_kv, need_weights=False)  # attenion
        x_hat = x_hat + residual  # 残差

        # ffn
        residual = x_hat
        x_hat = self.norm2_3(x_hat)  # Pre_LN
        x_hat = self.ffn(x_hat)  # ffn
        x_hat = residual + x_hat  # 残差
        return x_hat

    def forward(self, batch):
        # 获得latent：
        out_vq = self.ldm.get_input(batch,self.ldm.first_stage_key)
        z = out_vq

        # 信道：噪声 + 丢包
        y_wide = self.channel_1(z)
        y_mid = self.channel_2(z)
        y_narrow = self.channel_3(z)

        x_wide = self.linear_map(y_wide)
        x_mid = self.linear_map(y_mid)
        x_narrow = self.linear_map(y_narrow)
        print("linear map：",x_wide.shape)

        # 位置编码
        x_wide = x_wide + self.positional_encoding.pe[:, :]
        x_mid = x_mid + self.positional_encoding.pe[:, :]
        x_narrow = x_narrow + self.positional_encoding.pe[:, :]
        print("位置编码：done")

        # Pre-LN
        x_wide_norm = self.norm_wide(x_wide)
        x_mid_norm = self.norm_mid(x_mid)
        x_narrow_norm = self.norm_narrow(x_narrow)

        # 恢复
        # 单发
        '''
        x1_hat_sing = self.recon_1(x1_wide, x1_wide_norm)
        x2_hat_sing = self.recon_1(x2_wide, x2_wide_norm)
        x3_hat_sing = self.recon_1(x3_wide, x3_wide_norm)
        #双发
        x1_hat_doub = self.recon_2(x1_wide, x1_wide_norm, x1_mid_norm)
        x2_hat_doub = self.recon_2(x2_wide, x2_wide_norm, x2_mid_norm)
        x3_hat_doub = self.recon_2(x3_wide, x3_wide_norm, x3_mid_norm)
        '''
        #三发
        print("正在recon中")
        x_hat_trip = self.recon_3(x_wide, x_wide_norm, x_mid_norm, x_narrow_norm).view(48,3,64,64) #

        return x_hat_trip
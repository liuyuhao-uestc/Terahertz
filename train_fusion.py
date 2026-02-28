from sem_arq.utils.instantiate import instantiate_from_config,load_model
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import os
from sem_arq.modules.channel import Channel
from sem_arq.models.fusion import FusionNet
import random

def train_one_epoch(model, train_dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0

    for batch in train_dataloader:
        with torch.autocast(device_type=device, dtype=torch.float16):
            # 编码 + 加噪
            out, _ = ldm.get_input(batch, ldm.first_stage_key) # 编码器，out是编码结果，[batch,3,64,64]
            t = torch.randint(0, ldm.num_timesteps, (out.shape[0],),device=ldm.device).long() # 加噪时间步,batch内相同，[batch]
            # print("时间步t形状：",t.shape)  torch.Size([batch])
            z_noise = ldm.q_sample(out, t) # 加噪过程
            # print("z_noise形状：",z_noise.shape)  torch.Size([batch, 3, 64, 64])

            # 随机信道
            selected_channel_1, selected_channel_2 = random.sample(channels, 2)
            # print(f"选中的信道: snr: {selected_channel_1.snr_db}db; plr: {selected_channel_1.plr}")
            # print(f"选中的信道: snr: {selected_channel_2.snr_db}db; plr: {selected_channel_2.plr}")

            # 两次信道传输，仿真出接收信号
            z_received_1, z_received_2 = selected_channel_1(z_noise), selected_channel_2(z_noise) # torch.Size([batch, 3, 64, 64])
            z_received_1 = z_received_1.unsqueeze(1)
            z_received_2 = z_received_2.unsqueeze(1)
            z_received = torch.cat((z_received_1, z_received_2), dim=1)
            # print("z_received形状：", z_received.shape)  torch.Size([batch,2,3,64,64])

            # 融合
            z_fused = model(z_received)

            optimizer.zero_grad(set_to_none=True)

            loss = criterion(z_fused, z_noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            bs = z_noise.size(0)
            total_loss += loss.item() * bs
            n += bs
            print("batch完成",n)
    return total_loss / n

def save_ckpt(path, model):
    ckpt = {
        "model": model.state_dict(),
    }
    torch.save(ckpt, path)

if __name__ == "__main__":
    num_epochs = 10
    device = "mps"
    model_fusion = FusionNet().to(device)

    # 加载数据集
    config = OmegaConf.load('models/ldm/celeba/config.yaml')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    # 加载ldm模型
    config_ldm ='models/ldm/celeba/config.yaml'
    ckpt_ldm='models/ldm/celeba/model.ckpt'
    config_ldm = OmegaConf.load(config_ldm)
    ldm, ldm_gl = load_model(config_ldm, ckpt_ldm, 0, 0)
    ldm = ldm.eval()
    for p in ldm.parameters():
        p.requires_grad = False

    # 信道
    channels = [
        Channel(snr_db=4, plr=0.03),
        Channel(snr_db=8, plr=0.07),
        Channel(snr_db=12, plr=0.14),
        Channel(snr_db=16, plr=0.24),
        Channel(snr_db=20, plr=0.36)
    ]

    #
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
    model_fusion.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

    for epoch in range(num_epochs):
        tr = train_one_epoch(model_fusion, train_dataloader, criterion, optimizer, device)

        lr_cur = optimizer.param_groups[0]["lr"]
        msg = (
            f"[Epoch {epoch + 1:03d}/{num_epochs}] "
            f"lr={lr_cur:.2e} | "
            f"train_loss={tr['loss']:.6f} | "
            #f"val_loss={va['loss']:.6f}"
        )
        print(msg)

        root = "models/sem"
        model_path = os.path.join(root, "last.ckpt")
        save_ckpt(model_path, model_fusion)
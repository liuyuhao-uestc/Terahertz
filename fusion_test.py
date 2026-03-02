import torch
from sem_arq.models.fusion import FusionNet
from omegaconf import OmegaConf
from sem_arq.utils.instantiate import instantiate_from_config,load_model
from sem_arq.modules.channel import Channel
import random
import matplotlib.pyplot as plt
if __name__ == '__main__':
    device = "cuda"
    # 加载数据集
    config = OmegaConf.load('models/ldm/celeba/config.yaml')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = data.val_dataloader()

    # 加载ldm模型
    config_ldm = 'models/ldm/celeba/config.yaml'
    ckpt_ldm = 'models/ldm/celeba/model.ckpt'
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

    model = FusionNet()
    model_path = "models/sem/last.ckpt"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model']
    print(checkpoint.keys())
    model.load_state_dict(state_dict)
    # 5. 切换到推理模式（重要！）
    model.eval().to(device)


    for batch in train_dataloader:
        with torch.autocast(device_type=device, dtype=torch.float16):
            ### VAE编码可视化 ###
            img1 = batch['image'][0]
            img1 = img1.detach().cpu().numpy()
            fig, axes = plt.subplots(1, 7, figsize=(16, 2))
            axes[0].imshow(img1)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # 编码 + 加噪
            out, _ = ldm.get_input(batch, ldm.first_stage_key) # 编码器，out是编码结果，[batch,3,64,64]
            t = torch.randint(0, ldm.num_timesteps, (out.shape[0],),device=ldm.device).long() # 加噪时间步,batch内相同，[batch]
            # print("时间步t形状：",t.shape)  torch.Size([batch])
            z_noise = ldm.q_sample(out, t) # 加噪过程
            # print("z_noise形状：",z_noise.shape)  torch.Size([batch, 3, 64, 64])
            print("编码加噪完成")

            # 随机信道
            selected_channel_1, selected_channel_2 = random.sample(channels, 2)
            # print(f"选中的信道: snr: {selected_channel_1.snr_db}db; plr: {selected_channel_1.plr}")
            # print(f"选中的信道: snr: {selected_channel_2.snr_db}db; plr: {selected_channel_2.plr}")
            print("信道仿真完成")

            # 两次信道传输，仿真出接收信号
            z_received_1, z_received_2 = selected_channel_1(z_noise), selected_channel_2(z_noise) # torch.Size([batch, 3, 64, 64])
            z_received_1 = z_received_1.unsqueeze(1)
            z_received_2 = z_received_2.unsqueeze(1)
            z_received = torch.cat((z_received_1, z_received_2), dim=1)
            # print("z_received形状：", z_received.shape)  torch.Size([batch,2,3,64,64])

            # 融合
            z_fused = model(z_received)
            print("融合完成")
            print(f"z_fused: 均值{z_fused.mean().item():.4f}，方差{z_fused.std().item():.4f}")
            print(f"z_fused平均功率：{torch.mean(z_fused ** 2, dim=(0, 1, 2, 3), keepdim=True)}")

            sample_clean = z_noise
            sample_fused = z_fused

            # 去噪
            z_final, _ = ldm.progressive_denoising(
                cond=None,
                shape=out.shape,
                x_T=z_fused,
                start_T=t[0] + 1,  # 因为会从 t_val 一直迭代到 0
                verbose=False
            )
            print("去噪完成")

            # 解码
            img_final = ldm.decode_first_stage(z_final)
            img_rec = img_final[0]
            img_rec = img_rec.detach().cpu().numpy()
            print("解码完成")
            axes[1].imshow(img_rec)
            axes[1].set_title("Reconstruction Image")
            axes[1].axis("off")

            break

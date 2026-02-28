from sem_arq.utils.instantiate import instantiate_from_config,load_model
from omegaconf import OmegaConf
from sem_arq.models.model import my_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from taming.data.base import NumpyPaths
import os
from torch.utils.data import DataLoader
'''
每个batch:
==============================
Batch Type: <class 'dict'>
Key: 'image' | Shape: torch.Size([48, 256, 256, 3]) | Dtype: torch.float32
    -> Min: -1.000, Max: 1.000
Key: 'file_path_' | Value: ['data/celeba_hq_npy/imgHQ19337.npy', 'data/celeba_hq_npy/imgHQ08421.npy', 'data/celeba_hq_npy/imgHQ12193.npy', 'data/celeba_hq_npy/imgHQ00269.npy', 'data/celeba_hq_npy/imgHQ03962.npy', 'data/celeba_hq_npy/imgHQ02783.npy', 'data/celeba_hq_npy/imgHQ08353.npy', 'data/celeba_hq_npy/imgHQ18634.npy', 'data/celeba_hq_npy/imgHQ12014.npy', 'data/celeba_hq_npy/imgHQ17806.npy', 'data/celeba_hq_npy/imgHQ04824.npy', 'data/celeba_hq_npy/imgHQ05955.npy', 'data/celeba_hq_npy/imgHQ11213.npy', 'data/celeba_hq_npy/imgHQ12047.npy', 'data/celeba_hq_npy/imgHQ09750.npy', 'data/celeba_hq_npy/imgHQ12864.npy', 'data/celeba_hq_npy/imgHQ02713.npy', 'data/celeba_hq_npy/imgHQ09126.npy', 'data/celeba_hq_npy/imgHQ28111.npy', 'data/celeba_hq_npy/imgHQ13069.npy', 'data/celeba_hq_npy/imgHQ27144.npy', 'data/celeba_hq_npy/imgHQ21230.npy', 'data/celeba_hq_npy/imgHQ23754.npy', 'data/celeba_hq_npy/imgHQ23904.npy', 'data/celeba_hq_npy/imgHQ02958.npy', 'data/celeba_hq_npy/imgHQ06955.npy', 'data/celeba_hq_npy/imgHQ19045.npy', 'data/celeba_hq_npy/imgHQ23883.npy', 'data/celeba_hq_npy/imgHQ15390.npy', 'data/celeba_hq_npy/imgHQ22933.npy', 'data/celeba_hq_npy/imgHQ27194.npy', 'data/celeba_hq_npy/imgHQ29380.npy', 'data/celeba_hq_npy/imgHQ07645.npy', 'data/celeba_hq_npy/imgHQ03181.npy', 'data/celeba_hq_npy/imgHQ04129.npy', 'data/celeba_hq_npy/imgHQ28931.npy', 'data/celeba_hq_npy/imgHQ28769.npy', 'data/celeba_hq_npy/imgHQ08460.npy', 'data/celeba_hq_npy/imgHQ03416.npy', 'data/celeba_hq_npy/imgHQ10654.npy', 'data/celeba_hq_npy/imgHQ15640.npy', 'data/celeba_hq_npy/imgHQ05718.npy', 'data/celeba_hq_npy/imgHQ18492.npy', 'data/celeba_hq_npy/imgHQ23828.npy', 'data/celeba_hq_npy/imgHQ25815.npy', 'data/celeba_hq_npy/imgHQ28970.npy', 'data/celeba_hq_npy/imgHQ27556.npy', 'data/celeba_hq_npy/imgHQ00090.npy']
==============================
'''


if __name__ == "__main__":
    # 加载数据集
    config = OmegaConf.load('models/ldm/celeba/config.yaml')
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    config_ldm = 'models/ldm/celeba/config.yaml'
    ckpt_ldm = 'models/ldm/celeba/model.ckpt'
    config_ldm = OmegaConf.load(config_ldm)
    ldm, ldm_gl = load_model(config_ldm, ckpt_ldm, 0, 0)
    ldm = ldm.eval()
    for p in ldm.parameters():
        p.requires_grad = False

    for batch in train_dataloader:
        with torch.autocast(device_type="mps", dtype=torch.float16):

            ### VAE编码可视化 ###
            img1 = batch['image'][0]
            img1 = img1.detach().cpu().numpy()
            fig, axes = plt.subplots(1, 7, figsize=(16, 4))
            axes[0].imshow(img1)
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            out,_ =ldm.get_input(batch, ldm.first_stage_key)

            t = torch.randint(0, ldm.num_timesteps, (out.shape[0],), device=ldm.device).long()  # 加噪时间步,batch内相同，[batch]
            print("时间步t：",t[0])
            # print("时间步t形状：",t.shape)  torch.Size([batch])
            z_noise = ldm.q_sample(out, t)  # 加噪过程
            # print("z_noise形状：",z_noise.shape)  torch.Size([batch, 3, 64, 64])

            print(out.shape)
            latent1 = out[0]
            latent1 = latent1.detach().cpu().numpy()
            for i in range(3):
                axes[i + 1].imshow(latent1[i], cmap="gray")
                axes[i + 1].set_title(f"Latent Channel {i}")
                axes[i + 1].axis("off")
            '''
            img_noi = z_noise[0]
            img_noi = img_noi.detach().cpu().numpy()
            for i in range(3):
                axes[i + 4].imshow(img_noi[i], cmap="gray")
                axes[i + 4].set_title(f"noisy Channel {i}")
                axes[i + 4].axis("off")
            '''
            print("去噪中")
            z_final, _ = ldm.progressive_denoising(
                cond=None,
                shape=out.shape,
                x_T=z_noise,
                start_T=t[0] + 1,  # 因为会从 t_val 一直迭代到 0
                verbose=False
            )
            print("解码中")
            img_final = ldm.decode_first_stage(z_final)
            img_rec = img_final[0]
            img_rec = img_rec.detach().cpu().numpy()
            for i in range(3):
                axes[i + 4].imshow(img_rec[i], cmap="gray")
                axes[i + 4].set_title(f"noisy Channel {i}")
                axes[i + 4].axis("off")

            plt.suptitle("Image vs Latent Channels")
            plt.tight_layout()
            plt.show()
            # 假设 z 是你的 latent batch
            print(f"Mean: {out.mean().item():.4f}")
            print(f"Std:  {out.std().item():.4f}")
            print(f"Mean: {z_noise.mean().item():.4f}")
            print(f"Std:  {z_noise.std().item():.4f}")

        break
import torch
from taming.data.base import NumpyPaths
import os
from torch.utils.data import DataLoader
from sem_arq.utils.instantiate import instantiate_from_config
from omegaconf import OmegaConf
from sem_arq.utils.instantiate import load_model
from sem_arq.modules.channel import Channel
import random
# 生成仿真数据，用于训练，对于每张图片，随机选取传输次数，生成对应次数的接收信号，用于融合训练
# receive_data: 长度25000的list，每个元素是 shape [k,3,64,64] 的tensor
# 按顺序仿真每张图像，以此放入receive_data

if __name__ == '__main__':

    ## 顺序batch的dataloader，用于批量仿真生成数据
    root = "data/celeba_hq_npy"
    with open("data/celebahqtrain.txt", "r") as f:
        relpaths = f.read().splitlines()
    paths = [os.path.join(root, relpath) for relpath in relpaths]
    dataset = NumpyPaths(paths=paths, size=256, random_crop=False)
    train_dataloader = DataLoader(
    dataset,
    batch_size=8,        # 先试 8/16/32
    shuffle=False,        # 保证顺序
    num_workers=4,
    drop_last=False       # 离线生成不要丢最后一批
    )

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
        Channel(snr_db=0, plr=0),
        Channel(snr_db=1, plr=0.1),
        Channel(snr_db=2, plr=0.2),
        Channel(snr_db=3, plr=0.3),
        Channel(snr_db=4, plr=0.4)
    ]

    receive_data = torch.tensor([],device="mps") # [batch,2,3,64,64]

    # 编码+加噪

    for batch in train_dataloader:
        with torch.autocast(device_type="mps", dtype=torch.float16):
            out, _ = ldm.get_input(batch, ldm.first_stage_key) # 编码器，out是编码结果，[batch,3,64,64]
            t = torch.randint(0, ldm.num_timesteps, (out.shape[0],),device=ldm.device).long() # 加噪时间步,batch内相同，[batch]
            # print("时间步t形状：",t.shape)  torch.Size([batch])
            z_noise = ldm.q_sample(out, t) # 加噪过程
            # print("z_noise形状：",z_noise.shape)  torch.Size([batch, 3, 64, 64])

            selected_channel_1, selected_channel_2 = random.sample(channels, 2)
            # print(f"选中的信道: snr: {selected_channel_1.snr_db}db; plr: {selected_channel_1.plr}")
            # print(f"选中的信道: snr: {selected_channel_2.snr_db}db; plr: {selected_channel_2.plr}")

            z_received_1, z_received_2 = selected_channel_1(z_noise), selected_channel_2(z_noise) # torch.Size([batch, 3, 64, 64])
            z_received_1 = z_received_1.unsqueeze(1)
            z_received_2 = z_received_2.unsqueeze(1)
            # print("z_received形状：", z_received_1.shape)  torch.Size([batch,1,3,64,64])
            z_received = torch.cat((z_received_1,z_received_2),dim=1)
            receive_data = torch.cat((receive_data, z_received), dim=0)
            # print("receive_data形状：", receive_data.shape)  #torch.Size([随图片数叠加, 2, 3, 64, 64])
            if receive_data.shape[0]==8:
                print(receive_data.shape[0])
                break

    if receive_data.shape[0]==8:
        receive_data = receive_data.detach().cpu()
        torch.save({
            "receive_data": receive_data,
            "meta": {"version": "v1"}
        }, "data/receive_data_train.pt")
        print(f"receive_data形状{receive_data.shape}，已保存至data/receive_data_train.pt")
    else:
        print(f"保存失败，receive_data形状{receive_data.shape}")
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sem_arq.models.fusion import Attention,RefineNet
from sem_arq.modules.channel import Channel
import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

pack = torch.load("data/receive_data_train.pt", map_location="cpu")
receive_data = pack["receive_data"]
print(receive_data.shape)

'''
z = torch.tensor([])
y=torch.randn((3,224,224))
x = torch.randn((3,224,224))
y = y.unsqueeze(0)
x = x.unsqueeze(0)

z = torch.cat((z,x,y),dim=0)
print(z.shape)
print(20%10)

root = "data/celeba_hq_npy"
with open("data/celebahqtrain.txt", "r") as f:
    relpaths = f.read().splitlines()
paths = [os.path.join(root, relpath) for relpath in relpaths]
data = NumpyPaths(paths=paths, size=256, random_crop=False,labels={"label":torch.arange(25000)})

print(data[2026])
'''


'''
attn = Attention()
z_receive = torch.randn(48, 3, 3, 64, 64) # B, T, C, H, W
z_fused = attn(z_receive)
refine = RefineNet()
print(z_fused.shape)
z_fused = refine(z_fused)
print(z_fused.shape)
'''

from sem_arq.utils.instantiate import instantiate_from_config,load_model
from omegaconf import OmegaConf
from sem_arq.models.model import my_model
import torch
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
    config = OmegaConf.load('models/ldm/celeba/config.yaml')

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()

    data1 = instantiate_from_config(config.data1)
    data1.prepare_data()
    data1.setup()
    train_dataloader1 = data1.train_dataloader()
    val_dataloader1 = data1.val_dataloader()

    data2 = instantiate_from_config(config.data2)
    data2.prepare_data()
    data2.setup()
    train_dataloader2 = data2.train_dataloader()
    val_dataloader2 = data2.val_dataloader()
    model = my_model().to('mps')
    for batch1,batch2,batch3 in zip(train_dataloader,train_dataloader1,train_dataloader2):
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            x1, x2, x3 = model([batch1,batch2,batch3])

        break
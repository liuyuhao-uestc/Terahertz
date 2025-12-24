import importlib
import torch

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, sd):
    #instantiate_from_config(config)：从配置文件config.yaml中构建出网络模型（空模型）
    model = instantiate_from_config(config)
    #model.load_state_dict(sd)载人参数state_dict
    model.load_state_dict(sd,strict=False)
    model.to('mps')
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")#pytorch lightning state_dict
        '''
                total_params_state = sum(p.numel() for p in pl_sd['state_dict'].values())
                print(f"state_dict参数总量: {total_params_state:,}")
                for name, param in pl_sd['state_dict'].items():
                    print(f"{name:80s} | {tuple(param.shape)}")
        '''
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step
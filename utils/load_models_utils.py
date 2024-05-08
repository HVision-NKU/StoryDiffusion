import yaml
import torch
from diffusers import StableDiffusionXLPipeline
from utils import PhotoMakerStableDiffusionXLPipeline
import os

def get_models_dict():
    # 打开并读取YAML文件
    with open('config/models.yaml', 'r') as stream:
        try:
            # 解析YAML文件内容
            data = yaml.safe_load(stream)
            
            # 此时 'data' 是一个Python字典，里面包含了YAML文件的所有数据
            print(data)
            return data
            
        except yaml.YAMLError as exc:
            # 如果在解析过程中发生了错误，打印异常信息
            print(exc)

def load_models(model_info,device,photomaker_path):
    path =  model_info["path"]
    single_files =  model_info["single_files"]
    use_safetensors = model_info["use_safetensors"]
    model_type = model_info["model_type"]

    if model_type == "original":
        if single_files:
            pipe = StableDiffusionXLPipeline.from_single_file(
                path, 
                torch_dtype=torch.float16
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
        pipe = pipe.to(device)
    elif model_type == "Photomaker":
        if single_files:
            print("loading from a single_files")
            pipe = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                path, 
                torch_dtype=torch.float16
            )
        else:
            pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
        pipe = pipe.to(device)
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img"  # define the trigger word
        )
        pipe.fuse_lora()
    else:
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {model_type}")
    return pipe
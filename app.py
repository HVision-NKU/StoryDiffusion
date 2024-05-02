from email.policy import default
import gradio as gr
import numpy as np
import spaces
import torch
import requests
import random
import os
import sys
import pickle
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from utils.gradio_utils import is_torch2_available
if is_torch2_available():
    from utils.gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
    # from utils.gradio_utils import SpatialAttnProcessor2_0
else:
    from utils.gradio_utils  import AttnProcessor

import diffusers
from diffusers import StableDiffusionXLPipeline
from utils import PhotoMakerStableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
import copy
import os
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles
image_encoder_path = "./data/models/ip_adapter/sdxl_models/image_encoder"
ip_ckpt = "./data/models/ip_adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"
global models_dict
use_va = True
models_dict = {
#    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
#    "RealVision": "SG161222/RealVisXL_V4.0" ,
#    "SDXL":"stabilityai/stable-diffusion-xl-base-1.0" ,
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}
photomaker_path =  hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
MAX_SEED = np.iinfo(np.int32).max
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def set_text_unfinished():
    return gr.update(visible=True, value="<h3>(Not Finished) Generating ···  The intermediate results will be shown.</h3>")
def set_text_finished():
    return gr.update(visible=True, value="<h3>Generation Finished</h3>")
#################################################
def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list

#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        # 生成一个0到1之间的随机数
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),hidden_states[:1],self.id_bank[cur_step][1].to(self.device),hidden_states[1:]))
        # 判断随机数是否大于0.5
        if cur_step <5:
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                # print("mask shape",mask1024.shape,mask4096.shape)
                if not write:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    # print(self.total_length,self.id_length,hidden_states.shape,(height//32) * (width//32))
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,:mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,:mask4096.shape[0] // self.total_length * self.id_length]
                   # print(attention_mask.shape)
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # print("hidden state shape",hidden_states.shape,self.id_length)
        residual = hidden_states
        # if encoder_hidden_states is not None:
        #     raise Exception("not implement")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # print(key.shape,value.shape,query.shape,attention_mask.shape)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        #print(query.shape,key.shape,value.shape,attention_mask.shape)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def set_attention_processor(unet,id_length,is_ipadapter = False):
    global total_count
    total_count = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
                total_count +=1
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print("successsfully load paired self-attention")
    print(f"number of the processor : {total_count}")
#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''


#################################################
title = r"""
<h1 align="center">StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'><b>StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation</b></a>.<br>
❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1️⃣ Enter a Textual Description for Character, if you add the Ref-Image, making sure to <b>follow the class word</b> you want to customize with the <b>trigger word</b>: `img`, such as: `man img` or `woman img` or `girl img`.<br>
2️⃣ Enter the prompt array, each line corrsponds to one generated image.<br>
3️⃣ Choose your preferred style template.<br>
4️⃣ Click the <b>Submit</b> button to start customizing.
"""

article = r"""

If StoryDiffusion is helpful, please help to ⭐ the <a href='https://github.com/HVision-NKU/StoryDiffusion' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/HVision-NKU/StoryDiffusion?style=social)](https://github.com/HVision-NKU/StoryDiffusion)
---
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:

```bibtex
@article{Zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  year={2024}
}
```
📋 **License**
<br>
The Contents you create are under Apache-2.0 LICENSE. The Code are under Attribution-NonCommercial 4.0 International. 

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>ypzhousdu@gmail.com</b>.
"""
version = r"""
<h3 align="center">StoryDiffusion Version 0.01 (test version)</h3>

<h5 >1. Support image ref image. (Cartoon Ref image is not support now)</h5>
<h5 >2. Support Typesetting Style and Captioning.(By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
<h5 >3. [NC]symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling."),Currently, support is only using Textual Description</h5>
<h5 align="center">Tips: Not Ready Now! Just Test</h5>
"""
#################################################
global attn_count, total_count, id_length, total_length,cur_step, cur_model_type
global write
global  sa32, sa64
global height,width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
device="cuda"
global attn_procs,unet
attn_procs = {}
###
write = False
###
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
###
global sd_model_path
sd_model_path = models_dict["Unstable"]#"SG161222/RealVisXL_V4.0"
use_safetensors= False
### LOAD Stable Diffusion Pipeline
pipe1 = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors= use_safetensors)
pipe1 = pipe1.to("cuda")
pipe1.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe1.scheduler.set_timesteps(50)
### 
pipe2 = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    sd_model_path, torch_dtype=torch.float16, use_safetensors=use_safetensors)
pipe2 = pipe2.to("cuda")
pipe2.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"  # define the trigger word
)
pipe2 = pipe2.to("cuda")
pipe2.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe2.fuse_lora()

######### Gradio Fuction #############

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def remove_tips():
    return gr.update(visible=False)

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive) 

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

def change_visiale_by_model_type(_model_type):
    if _model_type == "Only Using Textual Description":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif _model_type == "Using Ref Images":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    else:
        raise ValueError("Invalid model type",_model_type)


######### Image Generation ##############
@spaces.GPU
def process_generation(_sd_type,_model_type,_upload_images, _num_steps,style_name, _Ip_Adapter_Strength ,_style_strength_ratio, guidance_scale, seed_,  sa32_, sa64_, id_length_,  general_prompt, negative_prompt,prompt_array,G_height,G_width,_comic_type):
    _model_type = "Photomaker" if _model_type == "Using Ref Images" else "original"
    if _model_type == "Photomaker" and "img" not in general_prompt:
        raise gr.Error("Please add the triger word \" img \"  behind the class word you want to customize, such as: man img or woman img")
    if _upload_images is None and _model_type != "original":
        raise gr.Error(f"Cannot find any input face image!")
    global sa32, sa64,id_length,total_length,attn_procs,unet,cur_model_type,device
    global write
    global cur_step,attn_count
    global height,width
    height = G_height
    width = G_width
    global pipe1,pipe2
    global sd_model_path,models_dict
    sd_model_path = models_dict[_sd_type]
    use_safe_tensor = True
    if _model_type == "original":
        pipe = pipe1
        set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    elif _model_type == "Photomaker":
        pipe = pipe2
        set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    else:
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")
        ##### ########################
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    cur_model_type = _sd_type+"-"+_model_type+""+str(id_length_)
    if _model_type != "original":
        input_id_images = []
        for img in _upload_images:
            print(img)
            input_id_images.append(load_image(img))
    prompts = prompt_array.splitlines()
    start_merge_step = int(float(_style_strength_ratio) / 100 * _num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(f"start_merge_step:{start_merge_step}")
    generator = torch.Generator(device="cuda").manual_seed(seed_)
    sa32, sa64 =  sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]
    prompts = [general_prompt + "," + prompt if "[NC]" not in prompt else prompt.replace("[NC]","")  for prompt in clipped_prompts]
    prompts = [prompt.rpartition('#')[0] if "#" in prompt else prompt for prompt in prompts]
    print(prompts)
    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]
    torch.cuda.empty_cache()
    write = True
    cur_step = 0

    attn_count = 0
    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)
    setup_seed(seed_)
    total_results = []
    if _model_type == "original":
        id_images = pipe(id_prompts, num_inference_steps=_num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images
    elif _model_type == "Photomaker":
        id_images = pipe(id_prompts,input_id_images=input_id_images, num_inference_steps=_num_steps, guidance_scale=guidance_scale, start_merge_step = start_merge_step, height = height, width = width,negative_prompt = negative_prompt,generator = generator).images
    else: 
        raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")
    total_results = id_images + total_results
    yield total_results
    real_images = []
    write = False
    for real_prompt in real_prompts:
        setup_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        if _model_type == "original":   
            real_images.append(pipe(real_prompt,  num_inference_steps=_num_steps, guidance_scale=guidance_scale,  height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
        elif _model_type == "Photomaker":      
            real_images.append(pipe(real_prompt, input_id_images=input_id_images, num_inference_steps=_num_steps, guidance_scale=guidance_scale,  start_merge_step = start_merge_step, height = height, width = width,negative_prompt = negative_prompt,generator = generator).images[0])
        else:
            raise NotImplementedError("You should choice between original and Photomaker!",f"But you choice {_model_type}")
        total_results = [real_images[-1]] + total_results
        yield total_results
    if _comic_type != "No typesetting (default)":
        captions= prompt_array.splitlines()
        captions = [caption.replace("[NC]","") for caption in captions]
        captions = [caption.split('#')[-1] if "#" in caption else caption for caption in captions]
        from PIL import ImageFont
        total_results = get_comic(id_images + real_images, _comic_type,captions= captions,font=ImageFont.truetype("./fonts/Inkfree.ttf", int(45))) + total_results
    set_attention_processor(pipe.unet,id_length_,is_ipadapter = False)
    yield total_results



def array2string(arr):
    stringtmp = ""
    for i,part in enumerate(arr):
        if i != len(arr)-1:
            stringtmp += part +"\n"
        else:
            stringtmp += part

    return stringtmp


#################################################
#################################################
### define the interface
with gr.Blocks(css=css) as demo:
    binary_matrixes = gr.State([])
    color_layout = gr.State([])

    # gr.Markdown(logo)
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Group(elem_id="main-image"):
            # button_run = gr.Button("generate id images ! 😺", elem_id="main_button", interactive=True)

            prompts = []
            colors = []
            # with gr.Column(visible=False) as post_sketch:
            #     for n in range(MAX_COLORS):
            #         if n == 0 :
            #             with gr.Row(visible=False) as color_row[n]:
            #                 colors.append(gr.Image(shape=(100, 100), label="background", type="pil", image_mode="RGB", width=100, height=100))
            #                 prompts.append(gr.Textbox(label="Prompt for the background (white region)", value=""))
            #         else:
            #             with gr.Row(visible=False) as color_row[n]:
            #                 colors.append(gr.Image(shape=(100, 100), label="segment "+str(n), type="pil", image_mode="RGB", width=100, height=100))
            #                 prompts.append(gr.Textbox(label="Prompt for the segment "+str(n)))

            #     get_genprompt_run = gr.Button("(2) I've finished segment labeling ! 😺", elem_id="prompt_button", interactive=True)

            with gr.Column(visible=True) as gen_prompt_vis:
                sd_type = gr.Dropdown(choices=list(models_dict.keys()), value = "Unstable",label="sd_type", info="Select pretrained model")
                model_type = gr.Radio(["Only Using Textual Description", "Using Ref Images"], label="model_type", value = "Only Using Textual Description",  info="Control type of the Character")
                with gr.Group(visible=False) as control_image_input:
                    files = gr.Files(
                                label="Drag (Select) 1 or more photos of your face",
                                file_types=["image"],
                            )
                    uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)
                    with gr.Column(visible=False) as clear_button:
                        remove_and_reupload = gr.ClearButton(value="Remove and upload new ones", components=files, size="sm")
                general_prompt = gr.Textbox(value='', label="(1) Textual Description for Character", interactive=True)
                negative_prompt = gr.Textbox(value='', label="(2) Negative_prompt", interactive=True)
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                prompt_array = gr.Textbox(lines = 3,value='', label="(3) Comic Description (each line corresponds to a frame).", interactive=True)
                with gr.Accordion("(4) Tune the hyperparameters", open=True):
                    #sa16_ = gr.Slider(label=" (The degree of Paired Attention at 16 x 16 self-attention layers) ", minimum=0, maximum=1., value=0.3, step=0.1)
                    sa32_ = gr.Slider(label=" (The degree of Paired Attention at 32 x 32 self-attention layers) ", minimum=0, maximum=1., value=0.7, step=0.1)
                    sa64_ = gr.Slider(label=" (The degree of Paired Attention at 64 x 64 self-attention layers) ", minimum=0, maximum=1., value=0.7, step=0.1)
                    id_length_ = gr.Slider(label= "Number of id images in total images" , minimum=2, maximum=4, value=2, step=1)
                    # total_length_ = gr.Slider(label= "Number of total images", minimum=1, maximum=20, value=1, step=1)
                    seed_ = gr.Slider(label="Seed", minimum=-1, maximum=MAX_SEED, value=0, step=1)
                    num_steps = gr.Slider( 
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=50,
                    )
                    G_height = gr.Slider( 
                        label="height",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=768,
                    )
                    G_width = gr.Slider( 
                        label="width",
                        minimum=256,
                        maximum=1024,
                        step=32,
                        value=768,
                    )
                    comic_type = gr.Radio(["No typesetting (default)", "Four Pannel", "Classic Comic Style"], value = "Classic Comic Style", label="Typesetting Style", info="Select the typesetting style ")
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=5,
                    )
                    style_strength_ratio = gr.Slider(
                        label="Style strength of Ref Image (%)",
                        minimum=15,
                        maximum=50,
                        step=1,
                        value=20,
                        visible=False
                    )
                    Ip_Adapter_Strength = gr.Slider(
                        label="Ip_Adapter_Strength",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.5,
                        visible=False
                    )
                final_run_btn = gr.Button("Generate ! 😺")


        with gr.Column():
            out_image = gr.Gallery(label="Result", columns=2, height='auto')
            generated_information = gr.Markdown(label="Generation Details", value="",visible=False)
            gr.Markdown(version)
    model_type.change(fn = change_visiale_by_model_type , inputs = model_type, outputs=[control_image_input,style_strength_ratio,Ip_Adapter_Strength])
    files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
    remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

    final_run_btn.click(fn=set_text_unfinished, outputs = generated_information
    ).then(process_generation, inputs=[sd_type,model_type,files, num_steps,style, Ip_Adapter_Strength,style_strength_ratio, guidance_scale, seed_, sa32_, sa64_, id_length_, general_prompt, negative_prompt, prompt_array,G_height,G_width,comic_type], outputs=out_image
    ).then(fn=set_text_finished,outputs = generated_information)


    gr.Examples(
        examples=[
            [1,0.5,0.5,3,"a woman img, wearing a white T-shirt, blue loose hair",
                   "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                   array2string(["wake up in the bed",
                                "have breakfast",
                                "is on the road, go to company",
                                "work in the company",
                                "Take a walk next to the company at noon",
                                "lying in bed at night"]),
                                "Japanese Anime",  "Using Ref Images",get_image_path_list('./examples/taylor'),768,768
                ],
                [0,0.5,0.5,2,"a man, wearing black jacket",
                   "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                   array2string(["wake up in the bed",
                                "have breakfast",
                                "is on the road, go to the company,  close look",
                                "work in the company",
                                "laughing happily",
                                "lying in bed at night"
                                ]),
                                "Japanese Anime","Only Using Textual Description",get_image_path_list('./examples/taylor'),768,768
                ],
                [0,0.3,0.5,2,"a girl, wearing white shirt, black skirt, black tie, yellow hair",
                   "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
                    array2string([
                            "at home #at home, began to go to drawing",
                            "sitting alone on a park bench.",
                            "reading a book on a park bench.",
                            "[NC]A squirrel approaches, peeking over the bench. ",
                            "look around in the park. # She looks around and enjoys the beauty of nature.",
                            "[NC]leaf falls from the tree, landing on the sketchbook.",
                            "picks up the leaf, examining its details closely.",
                            "starts sketching the leaf with intricate lines.",
                            "holds up the sketch drawing of the leaf.",
                            "[NC]The brown squirrel appear.",
                            "is very happy # She is very happy to see the squirrel again",
                            "[NC]The brown squirrel takes the cracker and scampers up a tree. # She gives the squirrel cracker",
                            "laughs and tucks the leaf into her book as a keepsake.",
                            "ready to leave.",]),
                    "Japanese Anime","Only Using Textual Description",get_image_path_list('./examples/taylor'),768,768
                ]
                ],
        inputs=[seed_, sa32_, sa64_, id_length_,  general_prompt, negative_prompt, prompt_array,style,model_type,files,G_height,G_width],
        # outputs=[post_sketch, binary_matrixes, *color_row, *colors, *prompts, gen_prompt_vis, general_prompt, seed_],
        # run_on_click=True,
        label='😺 Examples 😺',
    )
    gr.Markdown(article)

    # demo.load(None, None, None, _js=load_js)

demo.launch(server_name="0.0.0.0", share = True if use_va else False)
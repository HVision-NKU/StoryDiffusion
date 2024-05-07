# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import copy
import random
import subprocess
import numpy as np
import time
import torch
import torch.nn.functional as F
from PIL import ImageFont
from cog import BasePredictor, Input, Path, BaseModel
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils import load_image

from utils import PhotoMakerStableDiffusionXLPipeline
from utils.style_template import styles
from utils.gradio_utils import (
    AttnProcessor2_0 as AttnProcessor,
)  # with torch2 installed
from utils.gradio_utils import cal_attn_mask_xl
from utils.utils import get_comic

MODEL_URL = "https://weights.replicate.delivery/default/HVision_NKU/StoryDiffusion.tar"
MODEL_CACHE = "model_weights"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"

global total_count, attn_count, cur_step, mask1024, mask4096, attn_procs, unet
global sa32, sa64
global write
global height, width


"""
# load and upload the weights to replicate.delivery for faster booting on Replicate
models_dict = {
    "RealVision": "SG161222/RealVisXL_V4.0",
    "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y",
}
# photomaker_path =  hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
photomaker_path = f"{MODEL_CACHE}/PhotoMaker/photomaker-v1.bin"

pipe_unstable = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    models_dict["Unstable"],
    torch_dtype=torch.float16,
    use_safetensors=False,
)
pipe_unstable.save_pretrained(f"{MODEL_CACHE}/Unstable/stablediffusionapi/sdxl-unstable-diffusers-y")

pipe_realvision = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    models_dict["RealVision"], torch_dtype=torch.float16, use_safetensors=True
)
pipe_realvision.save_pretrained(f"{MODEL_CACHE}/RealVision/SG161222/RealVisXL_V4.0")
"""


class ModelOutput(BaseModel):
    comic: Path
    individual_images: list[Path]


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)


def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [
        p.replace("{prompt}", positive) for positive in positives
    ], n + " " + negative


def set_attention_processor(unet, id_length, is_ipadapter=False):
    global total_count
    total_count = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
                total_count += 1
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
    print("Successfully load paired self-attention")
    print(f"Number of the processor : {total_count}")


#################################################
########Consistent Self-Attention################
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

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device="cuda",
        dtype=torch.float16,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
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
        temb=None,
    ):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64
        global write
        global height, width
        if write:
            self.id_bank[cur_step] = [
                hidden_states[: self.id_length],
                hidden_states[self.id_length :],
            ]
        else:
            encoder_hidden_states = torch.cat(
                (
                    self.id_bank[cur_step][0].to(self.device),
                    hidden_states[:1],
                    self.id_bank[cur_step][1].to(self.device),
                    hidden_states[1:],
                )
            )
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb
            )
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            mask1024.shape[0] // self.total_length * self.id_length :
                        ]
                    else:
                        attention_mask = mask4096[
                            mask4096.shape[0] // self.total_length * self.id_length :
                        ]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[
                            : mask1024.shape[0] // self.total_length * self.id_length,
                            : mask1024.shape[0] // self.total_length * self.id_length,
                        ]
                    else:
                        attention_mask = mask4096[
                            : mask4096.shape[0] // self.total_length * self.id_length,
                            : mask4096.shape[0] // self.total_length * self.id_length,
                        ]
                hidden_states = self.__call1__(
                    attn, hidden_states, encoder_hidden_states, attention_mask, temb
                )
            else:
                hidden_states = self.__call2__(
                    attn, hidden_states, None, attention_mask, temb
                )
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length,
                self.id_length,
                sa32,
                sa64,
                height,
                width,
                device=self.device,
                dtype=self.dtype,
            )

        return hidden_states

    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                total_batch_size, channel, height * width
            ).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(
            -1, img_nums * nums_token, channel
        )

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, nums_token, channel
            ).reshape(-1, (self.id_length + 1) * nums_token, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            total_batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch_size, channel, height, width
            )
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
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, channel = hidden_states.shape
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length, channel
            ).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        models_dict = {
            "RealVision": "SG161222/RealVisXL_V4.0",
            "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y",
        }

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        photomaker_path = f"{MODEL_CACHE}/PhotoMaker/photomaker-v1.bin"

        self.sdxl_pipe_unstable = StableDiffusionXLPipeline.from_pretrained(
            f"{MODEL_CACHE}/Unstable/sdxl/stablediffusionapi/sdxl-unstable-diffusers-y",
            torch_dtype=torch.float16,
        )
        self.sdxl_pipe_realvision = StableDiffusionXLPipeline.from_pretrained(
            f"{MODEL_CACHE}/RealVision/sdxl/SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
        )

        self.pipe_unstable = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            f"{MODEL_CACHE}/Unstable/stablediffusionapi/sdxl-unstable-diffusers-y",
            torch_dtype=torch.float16,
            use_safetensors=False,
        )
        self.pipe_unstable.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",  # define the trigger word
        )

        self.pipe_realvision = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            f"{MODEL_CACHE}/RealVision/SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.pipe_realvision.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word="img",  # define the trigger word
        )
        self.pipe_realvision.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        self.pipe_realvision.fuse_lora()

    @torch.inference_mode()
    def predict(
        self,
        sd_model: str = Input(
            description="Choose a model",
            choices=["Unstable", "RealVision"],
            default="Unstable",
        ),
        ref_image: Path = Input(
            description="Reference image for the character",
            default=None,
        ),
        character_description: str = Input(
            description="General description of the character. If ref_image above is provided, making sure to follow the class word you want to customize with the trigger word 'img', such as: 'man img' or 'woman img' or 'girl img'",
            default="a man, wearing black suit",
        ),
        negative_prompt: str = Input(
            description="Describe things you do not want to see in the output",
            default="bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
        ),
        comic_description: str = Input(
            description="Comic Description. Each frame is divided by a new line. Only the first 10 prompts are valid for demo speed! For comic_description NOT using ref_image: (1) Support Typesetting Style and Captioning. By default, the prompt is used as the caption for each image. If you need to change the caption, add a '#' at the end of each line. Only the part after the '#' will be added as a caption to the image. (2) The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the '[NC]' at the beginning of the line.",
            default="at home, read new paper #at home, The newspaper says there is a treasure house in the forest.\non the road, near the forest\n[NC] The car on the road, near the forest #He drives to the forest in search of treasure.\n[NC]A tiger appeared in the forest, at night \nvery frightened, open mouth, in the forest, at night\nrunning very fast, in the forest, at night\n[NC] A house in the forest, at night #Suddenly, he discovers the treasure house!\nin the house filled with  treasure, laughing, at night #He is overjoyed inside the house.",
        ),
        style_name: str = Input(
            description="Style template",
            choices=STYLE_NAMES,
            default=DEFAULT_STYLE_NAME,
        ),
        comic_style: str = Input(
            description="Select the comic style for the combined comic",
            choices=["Four Pannel", "Classic Comic Style"],
            default="Classic Comic Style",
        ),
        style_strength_ratio: int = Input(
            description="Style strength of Ref Image (%), only used if ref_image is provided",
            default=20,
            ge=15,
            le=50,
        ),
        image_width: int = Input(
            description="Width of output image",
            choices=[
                256,
                288,
                320,
                352,
                384,
                416,
                448,
                480,
                512,
                544,
                576,
                608,
                640,
                672,
                704,
                736,
                768,
                800,
                832,
                864,
                896,
                928,
                960,
                992,
                1024,
            ],
            default=768,
        ),
        image_height: int = Input(
            description="Height of output image",
            choices=[
                256,
                288,
                320,
                352,
                384,
                416,
                448,
                480,
                512,
                544,
                576,
                608,
                640,
                672,
                704,
                736,
                768,
                800,
                832,
                864,
                896,
                928,
                960,
                992,
                1024,
            ],
            default=768,
        ),
        num_steps: int = Input(
            description="Number of sample steps", ge=20, le=50, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0.1, le=10, default=5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        sa32_setting: float = Input(
            description="The degree of Paired Attention at 32 x 32 self-attention layers",
            default=0.5,
            ge=0,
            le=1.0,
        ),
        sa64_setting: float = Input(
            description="The degree of Paired Attention at 64 x 64 self-attention layers",
            default=0.5,
            ge=0,
            le=1.0,
        ),
        num_ids: int = Input(
            description="Number of id images in total images. This should not exceed total number of line-separated prompts",
            default=3,
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        global total_count, attn_count, cur_step, mask1024, mask4096, attn_procs, unet
        global sa32, sa64
        global write
        global height, width

        assert (
            len(character_description.strip()) > 0
        ), "Please provide the description of the character."

        if ref_image is not None:
            assert (
                "img" in character_description
            ), f"When using ref_image, please add the trigger word 'img' behind the class word you want to customize, such as: man img or woman img"
            assert (
                "[NC]" not in comic_description
            ), "You should not use trigger word [NC] when ref_image is provided."

        height = image_height
        width = image_width
        id_length = num_ids
        sa32 = sa32_setting
        sa64 = sa64_setting

        clipped_prompts = comic_description.splitlines()[:10]
        print(clipped_prompts)
        prompts = [
            (
                character_description + "," + prompt
                if "[NC]" not in prompt
                else prompt.replace("[NC]", "")
            )
            for prompt in clipped_prompts
        ]
        print(prompts)
        prompts = [
            prompt.rpartition("#")[0].strip() if "#" in prompt else prompt.strip()
            for prompt in prompts
        ]
        print(prompts)
        assert id_length <= len(
            prompts
        ), "id_length should not exceed total number of line-separated prompts"

        id_prompts = prompts[:id_length]
        real_prompts = prompts[id_length:]

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        device = "cuda:0"
        setup_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        torch.cuda.empty_cache()

        model_type = "original" if ref_image is None else "Photomaker"

        if model_type == "original":
            pipe = (
                self.sdxl_pipe_realvision
                if style_name == "(No style)"
                else self.sdxl_pipe_unstable
            )
            pipe = pipe.to(device)
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        else:
            if sd_model != "RealVision" and style_name != "(No style)":
                pipe = self.pipe_unstable.to(device)
            else:
                pipe = self.pipe_realvision.to(device)
            pipe.id_encoder.to(device)

        write = True
        cur_step = 0
        attn_count = 0

        set_attention_processor(pipe.unet, id_length, is_ipadapter=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        curmodel_type = sd_model + "-" + model_type + "" + str(id_length)

        id_prompts, negative_prompt = apply_style(
            style_name, id_prompts, negative_prompt
        )

        total_results = []
        if model_type == "original":
            id_images = pipe(
                id_prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images
        else:
            input_id_images = [load_image(str(ref_image))]
            start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
            id_images = pipe(
                id_prompts,
                input_id_images=input_id_images,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                start_merge_step=start_merge_step,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images

        total_results = id_images + total_results

        real_images = []
        write = False
        for real_prompt in real_prompts:
            cur_step = 0
            real_prompt = apply_style_positive(style_name, real_prompt)
            if model_type == "original":
                real_images.append(
                    pipe(
                        real_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt,
                        generator=generator,
                    ).images[0]
                )
            else:
                real_images.append(
                    pipe(
                        real_prompt,
                        input_id_images=input_id_images,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        start_merge_step=start_merge_step,
                        height=height,
                        width=width,
                        negative_prompt=negative_prompt,
                        generator=generator,
                    ).images[0]
                )

            total_results = [real_images[-1]] + total_results

        captions = clipped_prompts
        captions = [caption.replace("[NC]", "") for caption in captions]
        captions = [
            caption.split("#")[-1].strip() if "#" in caption else caption.strip()
            for caption in captions
        ]

        comic = get_comic(
            id_images + real_images,
            comic_style,
            captions=captions,
            font=ImageFont.truetype("./fonts/Inkfree.ttf", int(45)),
        )

        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension
        comic_out = f"/tmp/comic.{extension}"
        comic[0].save(comic_out)

        save_params = {"format": extension.upper()}
        if not output_format == "png":
            save_params["quality"] = output_quality
            save_params["optimize"] = True

        output_paths = []
        for index, sample in enumerate(total_results[::-1]):
            output_filename = f"/tmp/out-{index}.{extension}"
            sample.save(output_filename, **save_params)
            output_paths.append(Path(output_filename))

        del pipe

        return ModelOutput(comic=Path(comic_out), individual_images=output_paths)

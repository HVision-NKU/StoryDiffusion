from calendar import c
from operator import invert
from webbrowser import get
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

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
        global total_count,attn_count,cur_step,mask256,mask1024,mask4096
        global sa16, sa32, sa64
        global write
        if write:
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat(self.id_bank[cur_step][0],hidden_states[:1],self.id_bank[cur_step][1],hidden_states[1:])
        # 判断随机数是否大于0.5
        if cur_step <5:
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == 32* 32:
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    elif hidden_states.shape[1] ==16*16:
                        attention_mask = mask256[mask256.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == 32* 32:
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length]
                    elif hidden_states.shape[1] ==16*16:
                        attention_mask = mask256[:mask256.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask256,mask1024,mask4096 = cal_attn_mask(self.total_length,self.id_length,sa16,sa32,sa64, device=self.device, dtype= self.dtype)

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
        if encoder_hidden_states is not None:
            raise Exception("not implement")
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

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

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

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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


def cal_attn_mask(total_length,id_length,sa16,sa32,sa64,device="cuda",dtype= torch.float16):
    bool_matrix256 = torch.rand((1, total_length * 256),device = device,dtype = dtype) < sa16
    bool_matrix1024 = torch.rand((1, total_length * 1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * 4096),device = device,dtype = dtype) < sa64
    bool_matrix256 = bool_matrix256.repeat(total_length,1)
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix256[i:i+1,id_length*256:] = False
        bool_matrix1024[i:i+1,id_length*1024:] = False
        bool_matrix4096[i:i+1,id_length*4096:] = False
        bool_matrix256[i:i+1,i*256:(i+1)*256] = True
        bool_matrix1024[i:i+1,i*1024:(i+1)*1024] = True
        bool_matrix4096[i:i+1,i*4096:(i+1)*4096] = True
    mask256 = bool_matrix256.unsqueeze(1).repeat(1,256,1).reshape(-1,total_length * 256)
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,1024,1).reshape(-1,total_length * 1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,4096,1).reshape(-1,total_length * 4096)
    return mask256,mask1024,mask4096

def cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    return mask1024,mask4096


def cal_attn_indice_xl_effcient_memory(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((total_length,nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((total_length,nums_4096),device = device,dtype = dtype) < sa64
    # 用nonzero()函数获取所有为True的值的索引
    indices1024 = [torch.nonzero(bool_matrix1024[i], as_tuple=True)[0] for i in range(total_length)]
    indices4096 = [torch.nonzero(bool_matrix4096[i], as_tuple=True)[0] for i in range(total_length)]

    return indices1024,indices4096


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
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
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
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
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

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


def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")


# 将列表转换为字典的函数
def character_to_dict(general_prompt):
    character_dict = {}    
    generate_prompt_arr = general_prompt.splitlines()
    character_index_dict = {}
    invert_character_index_dict = {}
    character_list = []
    for ind,string in enumerate(generate_prompt_arr):
        # 分割字符串寻找key和value
        start = string.find('[')
        end = string.find(']')
        if start != -1 and end != -1:
            key = string[start:end+1]
            value = string[end+1:]
            if "#" in value:
                value =  value.rpartition('#')[0] 
            if key in character_dict:
                raise gr.Error("duplicate character descirption: " + key)
            character_dict[key] = value
            character_list.append(key)

        
    return character_dict,character_list 

def get_id_prompt_index(character_dict,id_prompts):
    replace_id_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    for ind,id_prompt in enumerate(id_prompts):
                for key in character_dict.keys():
                    if key in id_prompt:
                        if key not in character_index_dict:
                            character_index_dict[key] = []
                        character_index_dict[key].append(ind)
                        invert_character_index_dict[ind] = key
                        replace_id_prompts.append(id_prompt.replace(key,character_dict[key]))

    return character_index_dict,invert_character_index_dict,replace_id_prompts

def get_cur_id_list(real_prompt,character_dict,character_index_dict):
    list_arr = []
    for keys in character_index_dict.keys():
        if keys in real_prompt:
            list_arr = list_arr +  character_index_dict[keys]
            real_prompt = real_prompt.replace(keys,character_dict[keys])
    return list_arr,real_prompt

def process_original_prompt(character_dict,prompts,id_length):
    replace_prompts = []
    character_index_dict = {}
    invert_character_index_dict = {}
    for ind,prompt in enumerate(prompts):
                for key in character_dict.keys():
                    if key in prompt:
                        if key not in character_index_dict:
                            character_index_dict[key] = []
                        character_index_dict[key].append(ind)
                        if ind not in invert_character_index_dict:
                            invert_character_index_dict[ind] = []
                        invert_character_index_dict[ind].append(key)
                cur_prompt = prompt
                if ind in invert_character_index_dict:
                    for key in invert_character_index_dict[ind]:
                        cur_prompt = cur_prompt.replace(key,character_dict[key] + " ")
                replace_prompts.append(cur_prompt)
    ref_index_dict = {}
    ref_totals = []
    print(character_index_dict)
    for character_key in character_index_dict.keys():
        if character_key not in character_index_dict:
            raise gr.Error("{} not have prompt description, please remove it".format(character_key))
        index_list = character_index_dict[character_key]
        index_list = [index for index in index_list if len(invert_character_index_dict[index]) == 1]
        if len(index_list) < id_length:
            raise gr.Error(f"{character_key} not have enough prompt description, need no less than {id_length}, but you give {len(index_list)}")
        ref_index_dict[character_key] = index_list[:id_length]
        ref_totals = ref_totals + index_list[:id_length]
    return character_index_dict,invert_character_index_dict,replace_prompts,ref_index_dict,ref_totals


def get_ref_character(real_prompt,character_dict):
    list_arr = []
    for keys in character_dict.keys():
        if keys in real_prompt:
            list_arr = list_arr + [keys]
    return list_arr
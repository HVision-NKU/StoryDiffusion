<p align="center">
  <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/f79da6b7-0b3b-4dd7-8dd0-ba0b15306fe6" height=100>
</p>

<div align="center">
  
## StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation  [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-md-dark.svg)]()

[[Paper](https://arxiv.org/abs/2405.01434)] &emsp; [[Project Page](https://storydiffusion.github.io/)] &emsp;  [[Jittor Version](https://github.com/JittorCV/jittordiffusion/tree/master)]&emsp; [[🤗 Comic Generation Demo ](https://huggingface.co/spaces/YupengZhou/StoryDiffusion)] [![Replicate](https://replicate.com/cjwbw/StoryDiffusion/badge)](https://replicate.com/cjwbw/StoryDiffusion) [![Run Comics Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HVision-NKU/StoryDiffusion/blob/main/Comic_Generation.ipynb) <br>
</div>


---

Official implementation of **[StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation]()**.

### **Demo Video**

https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/d5b80f8f-09b0-48cd-8b10-daff46d422af


### Update History

***You can visit [here](update.md) to visit update history.***

### 🌠  **Key Features:**
StoryDiffusion can create a magic story by generating consistent images and videos. Our work mainly has two parts: 
1. Consistent self-attention for character-consistent image generation over long-range sequences. It is hot-pluggable and compatible with all SD1.5 and SDXL-based image diffusion models. For the current implementation, the user needs to provide at least 3 text prompts for the consistent self-attention module. We recommend at least 5 - 6 text prompts for better layout arrangement.
2. Motion predictor for long-range video generation, which predicts motion between Condition Images in a compressed image semantic space, achieving larger motion prediction. 



## 🔥 **Examples**


### Comics generation 


![1](https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/b3771cbc-b6ca-4e26-bdc5-d944daf9f266)



### Image-to-Video generation （Results are HIGHLY compressed for speed）
Leveraging the images produced through our Consistent Self-Attention mechanism, we can extend the process to create videos by seamlessly transitioning between these images. This can be considered as a two-stage long video generation approach.

Note: results are **highly compressed** for speed, you can visit [our website](https://storydiffusion.github.io/) for the high-quality version.
#### Two-stage Long Videos Generation (New Update)
Combining the two parts, we can generate very long and high-quality AIGC videos.
| Video1 | Video2  | Video3  |
| --- | --- | --- |
| <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/4e7e0f24-5f90-419b-9a1e-cdf36d361b26" width=224>  | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/f509343d-d691-4e2a-b615-7d96381ef7c1" width=224> | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/4f0f7abb-4ae4-47a6-b692-5bdd8d9c8006" width=224>  |


#### Long Video Results using Condition Images
Our Image-to-Video model can generate a video by providing a sequence of user-input condition images.
| Video1 | Video2  | Video3  |
| --- | --- | --- |
| <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/af6f5c50-c773-4ef2-a757-6d7a46393f39" width=224>  | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/d58e4037-d8df-4f90-8c81-ce4b6d2d868e" width=224> |  <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/40da15ba-f5c1-48d8-84d6-8d327207d696" width=224>  |

| Video4 | Video5  | Video6  |
| --- | --- | --- |
| <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/8f04c9fc-3031-49e3-9de8-83d582b80a1f" width=224>  | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/604107fb-8afe-4052-bda4-362c646a756e" width=224> |  <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/b05fa6a0-12e6-4111-abf8-18b8cd84f3ff" width=224>  |




#### Short Videos 

| Video1 | Video2  | Video3  |
| --- | --- | --- |
| <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/5e7f717f-daad-46f6-b3ba-c087bd843158" width=224>  | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/79aa52b2-bf37-4c9c-8555-c7050aec0cdf" width=224> | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/9fdfd091-10e6-434e-9ce7-6d6e6d8f4b22" width=224>  |



| Video4 | Video5  | Video6  |
| --- | --- | --- |
| <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/0b219b60-a998-4820-9657-6abe1747cb6b" width=224>  | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/d387aef0-ffc8-41b0-914f-4b0392d9f8c5" width=224> | <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/3c64958a-1079-4ca0-a9cf-e0486adbc57f" width=224>  |




## 🚩 **TODO/Updates**
- [x] Comic Results of StoryDiffusion.
- [x] Video Results of StoryDiffusion.
- [x] Source code of Comic Generation
- [x] Source code of gradio demo
- [ ] Source code of Video Generation Model
- [ ] Pretrained weight of Video Generation Model
---

# 🔧 Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0](https://pytorch.org/)
```bash
conda create --name storydiffusion python=3.10
conda activate storydiffusion
pip install -U pip

# Install requirements
pip install -r requirements.txt
```
# How to use

Currently, we provide two ways for you to generate comics.

## Use the jupyter notebook

You can open the `Comic_Generation.ipynb` and run the code.

## Start a local gradio demo
Run the following command:


**(Recommend)** We provide a low GPU Memory cost version, it was tested on a machine with 24GB GPU-memory(Tesla A10) and 30GB RAM, and expected to work well with >20 G GPU-memory.

```python
python gradio_app_sdxl_specific_id_low_vram.py
```


## Contact
If you have any questions, you are very welcome to email ypzhousdu@gmail.com and zhoudaquan21@gmail.com

   


# Disclaimer
This project strives to impact the domain of AI-driven image and video generation positively. Users are granted the freedom to create images and videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.

# Related Resources
Following are some third-party implementations of StoryDiffusion.


## API

- [runpod.io serverless worker](https://github.com/bes-dev/story-diffusion-runpod-serverless-worker) provided by [BeS](https://github.com/bes-dev).
- [Replicate worker](https://github.com/camenduru/StoryDiffusion-replicate) provided by [camenduru](https://github.com/camenduru).




# BibTeX
If you find StoryDiffusion useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{zhou2024storydiffusion,
  title={StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation},
  author={Zhou, Yupeng and Zhou, Daquan and Cheng, Ming-Ming and Feng, Jiashi and Hou, Qibin},
  journal={NeurIPS 2024},
  year={2024}
}

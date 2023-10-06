
import os
os.chdir('rich_text_to_image_main')
import torch
# import imageio
import numpy as np
import matplotlib.pyplot as plt
from rich_text_to_image_main.models.region_diffusion import RegionDiffusion
from rich_text_to_image_main.utils.attention_utils import get_token_maps
from rich_text_to_image_main.utils.richtext_utils import seed_everything
from diffusers import StableDiffusionPipeline

'''
def Receiver_T2I_prepare():
  os.chdir('rich-text-to-image-main')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model_RX = RegionDiffusion(device)

  return model_RX



def Receiver_T2I(model_RX, text_prompt):
  os.chdir('rich-text-to-image-main')
  # text_prompt= 'a white cat is running through the grass'
  seed = 0
  seed_everything(seed)
  run_dir= 'results/visualize_token_maps'
  save_path = run_dir
  os.makedirs(save_path, exist_ok=True)

  model_RX.register_tokenmap_hooks()

  negative_text = ''
  token_ids= np.arange(len(text_prompt.split(' ')))

  base_tokens = model_RX.tokenizer._tokenize(text_prompt)
  obj_token_ids = [torch.LongTensor([obj_token_id+1])
                    for obj_token_id in token_ids]
  img = model_RX.produce_attn_maps([text_prompt], [negative_text],
                                    height=256, width=256, num_inference_steps=41,         #512
                                    guidance_scale=8.5)

  # token_maps = get_token_maps(
  #     model_RX.selfattn_maps, model_RX.crossattn_maps, model_RX.n_maps, save_path,
  #                                 512//8, 512//8, obj_token_ids, seed,                     #512
  #                                 base_tokens, segment_threshold=0.45, num_segments=10)  #draw map


  return img[0]

'''





def Receiver_T2I_prepare():
   pipe= StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)  #  원래 CompVis/stable-diffusion-v1-4

   return pipe.to("cuda")

def Receiver_T2I(model_RX, text_prompt):
  generator = torch.Generator("cuda").manual_seed(100) #500
  image = model_RX(text_prompt, generator=generator, num_inference_steps=50).images[0]  # num_inference_steps=50  #10만 해도 왠만큼 나오긴 함

  return image



if __name__ == "__main__":
    model= Receiver_T2I_prepare()
    text='dog splashing waves beach'#

    img = Receiver_T2I(model, text)
    img.save('C:/Users/hyeli/Dropbox/나메렝/projects/SemanGenComm/img/dog.png')
    img.show()
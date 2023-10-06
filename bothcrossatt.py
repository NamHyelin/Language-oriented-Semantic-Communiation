# !pip install diffusers==0.11.1
# !pip instasll daam
from diffusers import StableDiffusionPipeline
import daam
from daam import set_seed
import torch.nn.functional as F

def Receiver_T2I_prepare():
   pipe= StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

   return pipe.to("cuda")

def Receiver_T2I(model_RX, text_prompt):
  generator = set_seed(500)
  with daam.trace(model_RX) as trc:
    image = model_RX(text_prompt, generator=generator).images[0]
  global_heat_map = trc.compute_global_heat_map(prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=True)

  return image, global_heat_map

def attention_map(global_heat_map, text):
  heat_maps=[]
  tokens=text.split(' ')
  for i in range(len(tokens)):
    map=global_heat_map.compute_word_heat_map(tokens[i]).value.detach().cpu().float()
    heat_maps.append(F.interpolate(map.unsqueeze(0).unsqueeze(0), size=24, mode='bilinear', align_corners=True).squeeze())
  return heat_maps
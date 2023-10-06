# !pip install openai
import textwrap
import openai
import os
import torch
from torchvision import transforms
import torch.nn.functional as F
import re
import numpy as np
from torch.utils.data import DataLoader
from flicker_dataset import FlickrDataset, MyCollate

openai.api_key = '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'
width=1000
model_name = 'gpt-3.5-turbo'

##### In-context Fine tuning
initial_messages = []
user_content = "When converting the same image to text with CLIP and BLIP, the resulting text prompts are as follows. Can you catch the difference in context-wise between CLIP and BLIP by observing the each prompt pairs? \n\nCLIP: a little girl standing on the steps of a house \n BLIP: a little girl in a pink dress\n\nCLIP: two dogs playing with each other on the street\n BLIP: two dogs are walking down the road with a ball in their mouth\n\n CLIP: a little girl that is sitting in the grass\nBLIP: a child playing in the grass with a rainbow painted on it\n\n CLIP: a man wearing an orange hat with stickers on it\n BLIP: a man wearing a hat\n\n CLIP: a little girl climbing on a rope net\n BLIP: a little girl climbing on a red rope\n\n CLIP: a dog running in the grass with a frisbee in its mouth\nBLIP: a dog running in the grass near a white picket fence\n\n CLIP: a dog playing with a ball on the beach\n BLIP: a dog playing with a ball on the beach\n\n CLIP: a little boy standing next to a fire hydrant\n  BLIP: a young boy is standing in front of a lamp post\n\nCLIP: a dog that is standing on a log\n BLIP: a dog sitting on top of a log\n\n  CLIP: a brown and white dog running in the snow\n BLIP: a dog running in the snow\n\nCLIP: a man riding skis down a snow covered slope\n  BLIP: a group of people skiing in the snow\n\n CLIP: a group of people climbing up the side of a rock\n BLIP: a group of people climbing on a rock\n\n  CLIP: a dog that is running in the grass\n  BLIP: a dog running through a hose in the grass\n\n   CLIP: a white dog chasing after a yellow ball\n   BLIP: a dog running with a ball in its mouth\n\n  CLIP: a dog jumping in the air to catch a frisbee\n   BLIP: a dog running across a lush green field\n\n  CLIP: a couple of kids flying a kite next to a body of water\n   BLIP: a man and a woman are standing near the water\n\n CLIP: a woman sitting next to a baby in a stroller\n    BLIP: a man and woman sitting on a bench in the park\n\n   CLIP: a man standing on top of a snow covered field\n   BLIP: a man standing on top of a snow covered field\n\n   CLIP: two dogs running on a beach near the ocean\n  BLIP: two dogs running on the beach"
initial_messages.append({"role" : "user", "content" : f"{user_content}"})

completion = openai.ChatCompletion.create(model=model_name, messages=initial_messages) #,top_p=top_p, temperature = temperature)
assistant_content = completion.choices[0].message["content"].strip()
initial_result = textwrap.wrap(assistant_content, width = width)



def clip_caption(image_caption):
  # image_caption="a dog is running through the water in ocean"
  messages=[]
  user_content =  "When converting the same image to text with CLIP and BLIP, the resulting text prompts are as follows. Can you catch the difference in context-wise between CLIP and BLIP by observing the each prompt pairs? \n\nCLIP: a little girl standing on the steps of a house \n BLIP: a little girl in a pink dress\n\nCLIP: two dogs playing with each other on the street\n BLIP: two dogs are walking down the road with a ball in their mouth\n\n CLIP: a little girl that is sitting in the grass\nBLIP: a child playing in the grass with a rainbow painted on it\n\n CLIP: a man wearing an orange hat with stickers on it\n BLIP: a man wearing a hat\n\n CLIP: a little girl climbing on a rope net\n BLIP: a little girl climbing on a red rope\n\n CLIP: a dog running in the grass with a frisbee in its mouth\nBLIP: a dog running in the grass near a white picket fence\n\n CLIP: a dog playing with a ball on the beach\n BLIP: a dog playing with a ball on the beach\n\n CLIP: a little boy standing next to a fire hydrant\n  BLIP: a young boy is standing in front of a lamp post\n\nCLIP: a dog that is standing on a log\n BLIP: a dog sitting on top of a log\n\n  CLIP: a brown and white dog running in the snow\n BLIP: a dog running in the snow\n\nCLIP: a man riding skis down a snow covered slope\n  BLIP: a group of people skiing in the snow\n\n CLIP: a group of people climbing up the side of a rock\n BLIP: a group of people climbing on a rock\n\n  CLIP: a dog that is running in the grass\n  BLIP: a dog running through a hose in the grass\n\n   CLIP: a white dog chasing after a yellow ball\n   BLIP: a dog running with a ball in its mouth\n\n  CLIP: a dog jumping in the air to catch a frisbee\n   BLIP: a dog running across a lush green field\n\n  CLIP: a couple of kids flying a kite next to a body of water\n   BLIP: a man and a woman are standing near the water\n\n CLIP: a woman sitting next to a baby in a stroller\n    BLIP: a man and woman sitting on a bench in the park\n\n   CLIP: a man standing on top of a snow covered field\n   BLIP: a man standing on top of a snow covered field\n\n   CLIP: two dogs running on a beach near the ocean\n  BLIP: two dogs running on the beach"+"CLIP's prompt is" + image_caption + ". Maintaining the same meaning, generate and list 10 prompts using CLIP. Among them, chose one prompt that is most likely to be generated by BLIP"

  messages.append({"role" : "user", "content" : f"{user_content}"})

  completion = openai.ChatCompletion.create(model=model_name, messages=messages) #,top_p=top_p, temperature = temperature)
  assistant_content = completion.choices[0].message["content"].strip()
  result = textwrap.wrap(assistant_content, width = width)

  return result[0]



###### Save image captions
'''
# from transformers import logging
# logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

dataset = FlickrDataset(root_dir="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/Images", caption_path="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/captions.txt", transform=transform)
pad_value = dataset.vocab.stoi["<PAD>"]


batchsize=500
image_captions=np.zeros((batchsize), dtype=object)


loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))


for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx%5==0 :

      print('{}th image of {} nums of images'.format(img_idx, batchsize),'--------------------------------------------------------------------------------------------------------------------------------------------------------')

      txt_lists= np.array(np.load('/content/drive/MyDrive/Colab_Notebooks/trial_sentences_img100_partBob_most.npy', allow_pickle=True))
      image_caption = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]
      result = clip_caption(image_caption)
      image_caption = re.split(r'(1. )', result)[re.split(r'(1. )', result).index('1. '):][1].split('2')[0]
      image_captions[int(img_idx/5)]=image_caption

    np.save('/content/drive/MyDrive/Colab_Notebooks/image_caption_skd.npy', image_captions)
  if batch_idx==499:
    break

'''
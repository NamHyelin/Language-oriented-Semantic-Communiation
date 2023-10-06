# !pip install clip-interrogator==0.5.4
# !pip install transformers==4.26.1

from PIL import Image
from clip_interrogator import Config, Interrogator
import torch
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
import re
import numpy as np
from torch.utils.data import DataLoader
from flicker_dataset import FlickrDataset, MyCollate
import numpy as np
import torch


def Transmitter_I2T_prepare():
  ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
  return ci, ci

def Transmitter_I2T(processor, args, img_path):
  image = Image.open(img_path).convert('RGB')
  text = processor.interrogate(image).split(',')[0]
  return text, text, text



##### Save image captions
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
image_captions_clipTX = np.zeros((100), dtype=object)


# load model and data
# model_TX, args = Transmitter_I2T_prepare()
loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))


for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx%5==0 :
      print('{}th image of {} nums of images'.format(img_idx, batchsize),'--------------------------------------------------------------------------------------------------------------------------------------------------------')
      transform_img= T.ToPILImage()
      img = transform_img(imgs[img_idx])
      img.save("/content/img.jpg")
      img_ref_path= '/content/img.jpg'

      image_caption, dense_caption, region_semantic = Transmitter_I2T(model_TX, args, img_ref_path)
      image_captions_clipTX[int(img_idx/5)]=image_caption

    np.save('/content/drive/MyDrive/Colab_Notebooks/image_captions_clipTX.npy', image_captions_clipTX)
  if batch_idx==0:

    break
'''
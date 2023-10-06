# !pip install torchvision lpips

import torch
import numpy as np
import lpips
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage import io
from torchmetrics.image.fid import FrechetInceptionDistance
import pytorch_fid_wrapper as pfw
import cv2




def LPIPS(image1, image2):
  lpips_model = lpips.LPIPS(net='alex')

  # # Load the images
  # image1 = Image.open('img/save/input_ref/content/input_ref/18.jpg').convert('RGB')
  # image2 = Image.open('/img/dog.png').convert('RGB')

  image1_np=np.array(image1)
  image2_np=np.array(image2)
  # image1_np=np.concatenate((np.expand_dims(image1_np[-1],0),image1_np),0)

  # Preprocess the images
  image1_tensor = [torch.from_numpy(image1_np).permute(2, 0, 1).float() / 255.]
  image2_tensor = [torch.from_numpy(image2_np).permute(2, 0, 1).float() / 255.]

  # Transform the images to tensors and normalize
  transform = transforms.Compose([
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
      transforms.Resize((256,256))
  ])

  # Convert the images to tensors and add a batch dimension
  image1_tensor = transform(image1_tensor[0])
  image2_tensor = transform(image2_tensor[0])
  lpips_score = lpips_model.forward(image1_tensor, image2_tensor).item()

  return lpips_score






def SSIM(image1, image2):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((256, 256)),
    transforms.ToPILImage()
  ])

  image1 = transform(image1)
  image2 = transform(image2)

  image1_np = np.array(image1)
  image2_np = np.array(image2)

  grayA = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(image2_np, cv2.COLOR_BGR2GRAY)

  (ssim_score, diff) = ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")

  # # show difference
  # cv2.imshow('diff',diff)
  # cv2.waitKey(0)


  return ssim_score




def FID(image1, image2):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
  ])
  image1_tensor = transform(image1).unsqueeze(0)
  image2_tensor = transform(image2).unsqueeze(0)

  pfw.set_config(batch_size=1, device='cpu')
  fid_score= pfw.fid(image1_tensor, image2_tensor)
  return fid_score


if __name__ == "__main__":
  image1 = Image.open('img/prof.png')
  image2 = Image.open('img/prof2.png')

  fid=LPIPS(image1, image2)
  print(fid)

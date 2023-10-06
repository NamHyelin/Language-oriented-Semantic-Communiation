
import os
os.environ['OPENAI_KEY'] = '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'
# # 여기서 하면 됨 https://platform.openai.com/account/api-keys
# os.chdir('/Image2Paragraph_main')

from Image2Paragraph_main.models.blip2_model import ImageCaptioning
from Image2Paragraph_main.models.grit_model import DenseCaptioning
from Image2Paragraph_main.models.gpt_model import ImageToText
from Image2Paragraph_main.models.controlnet_model import TextToImage
from Image2Paragraph_main.models.region_semantic import RegionSemantic
from Image2Paragraph_main.utils.util import read_image_width_height, display_images_and_text, resize_long_edge
import argparse
from PIL import Image
import base64
from io import BytesIO
import os
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
from flicker_dataset import FlickrDataset, MyCollate
import numpy as np
import torch



def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


class ImageTextTransformation:
    def __init__(self, args):
        # Load your big model here
        self.args = args
        self.init_models()
        self.ref_image = None

    def init_models(self):
        openai_key = os.environ['OPENAI_KEY']
        print(self.args)
        print('\033[1;34m' + "Welcome to the Image2Paragraph toolbox...".center(50, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;31m' + "This is time-consuming, please wait...".center(50, '-') + '\033[0m')
        self.image_caption_model = ImageCaptioning(device=self.args.image_caption_device, captioner_base_model=self.args.captioner_base_model)
        self.dense_caption_model = DenseCaptioning(device=self.args.dense_caption_device)
        self.gpt_model = ImageToText(openai_key)
        self.controlnet_model = TextToImage(device=self.args.contolnet_device)
        self.region_semantic_model = RegionSemantic(device=self.args.semantic_segment_device, image_caption_model=self.image_caption_model, region_classify_model=self.args.region_classify_model, sam_arch=self.args.sam_arch)
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')


    def image_to_text(self, img_src):
        # the information to generate paragraph based on the context
        self.ref_image = Image.open(img_src)
        # resize image to long edge 384
        self.ref_image = resize_long_edge(self.ref_image, 384)
        width, height = read_image_width_height(img_src)
        print(self.args)
        if self.args.image_caption:
            image_caption = self.image_caption_model.image_caption(img_src)
        else:
            image_caption = " "
        if self.args.dense_caption:
            dense_caption = self.dense_caption_model.image_dense_caption(img_src)
        else:
            dense_caption = " "
        if self.args.semantic_segment:
            region_semantic = self.region_semantic_model.region_semantic(img_src)
        else:
            region_semantic = " "
        # generated_text = self.gpt_model.paragraph_summary_with_gpt(image_caption, dense_caption, region_semantic, width, height)
        return image_caption, dense_caption, region_semantic

    def text_to_image(self, text):
        generated_image = self.controlnet_model.text_to_image(text, self.ref_image)
        return generated_image

    def text_to_image_retrieval(self, text):
        pass

    def image_to_text_retrieval(self, image):
        pass






def Transmitter_I2T_prepare():
    # os.chdir('Image2Paragraph_main')
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_src') #, default='/content/cat.jpg'
    parser.add_argument('--out_image_name')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=False, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=False, help='Set this flag to True if you want to use semantic segmentation, takes long time and occurs error')
    parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_b', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
    parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
    parser.add_argument('--image_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_caption_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--semantic_segment_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')
    parser.add_argument('--contolnet_device', choices=['cuda', 'cpu'], default='cpu', help='Select the device: cuda or cpu, <6G GPU is not recommended>')

    args = parser.parse_args('')

    processor = ImageTextTransformation(args)
    return processor, args



def Transmitter_I2T(processor, args, img_path):
    # os.chdir('Image2Paragraph_main')
    args.image_src= img_path
    args.out_image_name= img_path
    image_caption, dense_caption, region_semantic = processor.image_to_text(args.image_src) #generated_text
    return image_caption, dense_caption, region_semantic


if __name__ == "__main__":
    processor, args= Transmitter_I2T_prepare()
    img_path='C:/Users/hyeli/Dropbox/나메렝/projects/SemanGenComm/img/cat.jpg'
    image_caption, dense_caption, region_semantic = Transmitter_I2T(processor, args, img_path)


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
model_TX, args = Transmitter_I2T_prepare()
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

    np.save('/content/drive/MyDrive/Colab_Notebooks/image_captions.npy', image_captions_clipTX)
  if batch_idx==0:

    break
    
'''
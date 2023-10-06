from transformers import BertTokenizer, BlipTextModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


def CLIP_get_att(text):
  # Load the tokenizer and model

  tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
  model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

  # Encode the text
  # text = "a white cat is running through a field of flowers"
  input_text = tokenizer(text, return_tensors="pt")

  # Get the attention values and sequence length
  outputs = model(input_ids=input_text.input_ids, output_attentions=True) #attention_mask =input_text.attention_mask
  attentions = outputs.attentions

  # print("Number of attention heads:", model.config.num_attention_heads)

  # layer_idx=1  #12
  # print("Attention size of {}th layer: {}".format(layer_idx, attentions[layer_idx].size()))

  attentions_t = torch.stack(list(attentions), dim=0).squeeze(1)
  attentions_np = torch.mean(attentions_t,(0,1)).detach().numpy()[1:-1, 1:-1]

  del tokenizer, model

  return attentions_np


def BLIP_get_att(text):
  # Load the tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased")
  # tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
  # model = BlipTextModel.from_pretrained("bert-base-cased")

  # Encode the text
  # text = "a white cat is running through a field of flowers"
  input_text = tokenizer(text, return_tensors="pt")

  # Get the attention values and sequence length
  outputs = model(input_ids=input_text.input_ids, output_attentions=True) #attention_mask =input_text.attention_mask
  attentions = outputs.attentions

  # print("Number of attention heads:", model.config.num_attention_heads)

  # layer_idx=1  #12
  # print("Attention size of {}th layer: {}".format(layer_idx, attentions[layer_idx].size()))

  attentions_t = torch.stack(list(attentions), dim=0).squeeze(1)
  attentions_np = torch.mean(attentions_t,(0,1)).detach().numpy()[1:-1, 1:-1]

  del tokenizer, model

  return attentions_np


def att_to_word(attentions_np, word_idx):
  return np.sum(attentions_np[word_idx, :]) + np.sum(attentions_np[:, word_idx]) - 2*attentions_np[word_idx, word_idx]

def att_word_to_word(attentions_np, idx_1, idx_2):
  return (attentions_np[idx_1, idx_2] + attentions_np[idx_2, idx_1])/2


##### Save atts
'''
import torchvision.transforms as T

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

dataset = FlickrDataset(root_dir="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/Images", caption_path="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/captions.txt", transform=transform)
pad_value = dataset.vocab.stoi["<PAD>"]
batchsize=500
loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))


Aliceatts= np.zeros((100,50,50))
Bobatts= np.zeros((100,50,50))

for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx%5==0 :
      print('{}th image of {} nums of images'.format(img_idx, batchsize),'--------------------------------------------------------------------------------------------------------------------------------------------------------')
      transform_img= T.ToPILImage()
      img = transform_img(imgs[img_idx])
      img.save("/content/img.jpg")
      img_ref_path= '/content/img.jpg'

      # image_caption, dense_caption, region_semantic = Transmitter_I2T(model_TX, args, img_ref_path)
      txt_lists= np.array(np.load('/content/drive/MyDrive/Colab_Notebooks/trial_sentences_img100_partBob_most.npy', allow_pickle=True))
      image_caption = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]


      Aliceatt= BLIP_get_att(image_caption)
      Bobatt= CLIP_get_att(image_caption)

      Aliceatts[int(img_idx/5),:Aliceatt.shape[0],:Aliceatt.shape[1]] = Aliceatt
      Bobatts[int(img_idx/5),:Bobatt.shape[0],:Bobatt.shape[1]] = Bobatt

      print(img_idx/5)

      np.save('/content/drive/MyDrive/Colab_Notebooks/Aliceatts.npy', Aliceatts)
      np.save('/content/drive/MyDrive/Colab_Notebooks/Bobatts.npy', Bobatts)
'''
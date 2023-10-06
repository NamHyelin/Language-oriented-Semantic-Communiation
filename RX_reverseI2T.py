# !pip install torchmetrics[text]
# !pip install transformers
# !pip install bert_score
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from torchvision.utils import save_image
from torchvision import transforms
import torch
import numpy as np
import os
from Tx_I2T import Transmitter_I2T_prepare, Transmitter_I2T
from flicker_dataset import FlickrDataset, MyCollate
from torch.utils.data import DataLoader

# from transformers import logging
# logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# parameters
lpips_th = 0.6
trial=1
batchsize=500
penalty=0
sort=True

# saves
text_bleu=np.zeros((batchsize, 50))
text_bert= np.zeros((batchsize, 50), dtype=object)

# load model and data
model_TX, args = Transmitter_I2T_prepare()

# !unzip -qq '/content/bothatt_comb.zip'

directory_path = '/content/content/bothatt_comb'
txt_lists= np.array(np.load('/content/drive/MyDrive/Colab_Notebooks/trial_sentences_bothAtt_combination.npy', allow_pickle=True))

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

dataset = FlickrDataset(root_dir="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/Images", caption_path="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/captions.txt", transform=transform)
pad_value = dataset.vocab.stoi["<PAD>"]
loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))


for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx%5==0 :
        caption_ref = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]


        file_names = os.listdir(directory_path)

        for idx, file_name in enumerate(file_names):
            file_idxs=[int(a) for a in file_name.split('.')[0].split('_')]
            caption_gen, _, _ = Transmitter_I2T(model_TX, args, file_name)

            # bleu
            ref = [caption_ref.split(' ')]
            gen = caption_gen.split(' ')
            bleu_score = sentence_bleu(ref, gen)
            text_bleu[file_idxs[0]][file_idxs[1]] = bleu_score

            # bert
            ref = [caption_ref]
            gen = caption_gen
            P, R, F1 = score([gen], [ref], lang="en", verbose=True)
            text_bert[file_idxs[0]][file_idxs[1]] = F1.item()

            np.save(text_bleu, '/content/text_blue_bothatt_comb.npy')
            np.save(text_bert, '/content/text_bert_bothatt_comb.npy')




##### Examples
'''
# Reference and generated sentences
reference = [['a', 'brown', 'dog', 'is', 'running']]
generated = ['a', 'dog', 'is', 'running']

# Calculate BLEU score
bleu_score = sentence_bleu(reference, generated)
print("BLEU Score:", bleu_score)



# Reference and generated sentences
references = ['a brown dog is running']
candidate = 'a dog is running'

# Calculate BERTScore
P, R, F1 = score([candidate], [references], lang="en", verbose=True)
print("BERTScore:", F1.item())
'''
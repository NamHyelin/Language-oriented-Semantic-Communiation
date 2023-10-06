import os
import pandas
import spacy

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import transforms


class FlickrDataset(Dataset):
    def __init__(self, root_dir="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/Images", caption_path="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/captions.txt", freq_threshold=5, transform=None, data_length=10000):
        self.freq_threshold = freq_threshold
        self.transform = transform
        self.root_dir = root_dir

        self.df = pandas.read_csv(caption_path)[:data_length]

        self.captions = self.df['caption']
        self.images = self.df['image']

        self.vocab = Vocabulary(freq_threshold)

        print(len(self.captions.tolist()))
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]  #여길 *5 해야할거같음
        image = self.images[index]

        img = Image.open(os.path.join(self.root_dir,image)).convert("RGB")

        if (self.transform):
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]

        numericalized_caption += self.vocab.numericalize(caption)

        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)




spacy_eng = spacy.load("en_core_web_sm")



class Vocabulary:
    def __init__(self, freq_threshold):

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self,sentences):
        idx = 4
        frequency = {}

        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1

                if (frequency[word] > self.freq_threshold-1):
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self,sentence):
        tokenized_text = self.tokenizer_eng(sentence)

        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in tokenized_text ]





class MyCollate:
    def __init__(self, pad_value):
        self.pad_value = pad_value

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        img = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_value)

        return img, targets


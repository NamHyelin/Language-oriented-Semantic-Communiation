!pip install stanza
!pip install daam==0.1.0
import stanza
stanza.install_corenlp(dir='stanford-corenlp-4.5.4')

%env CORENLP_HOME=stanford-corenlp-4.5.4

!wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.4.zip
!unzip stanford-corenlp-4.5.4.zip

!wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

!mkdir -p coco
!mv annotations_* coco

%cd coco
!unzip annotations_*


from stanza.server import CoreNLPClient
from pathlib import Path
import json
import torch
from diffusers import StableDiffusionPipeline
from daam import set_seed, trace
from daam import GenerationExperiment
from matplotlib import pyplot as plt
import torchvision.transforms as T
import numpy as np
import torch
import re


stanza.install_corenlp(dir='stanford-corenlp-4.5.4')
client = CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse','coref'], timeout=30000, memory='6G', endpoint='http://localhost:9001', be_quiet = False)


annotations = json.load(Path('/content/coco/annotations/captions_val2014.json').open())

torch.cuda.amp.autocast().__enter__()
torch.set_grad_enabled(False);

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
pipe.to('cuda:0')
gen = set_seed(500)

caption='a white cat running through a field of flowers'

!mkdir '/content/experiments/'
output_folder = Path('/content/experiments/')





learning=True
batchsize=500
sentences = np.empty((int(batchsize/5), 2), dtype=object)
relations = np.empty((int(batchsize/5), 3), dtype=object)

client.start()

for img_idx in range(500):
  if img_idx%5==0 and img_idx<=100:
    print('{}th image of {} nums of images'.format(img_idx, batchsize),'--------------------------------------------------------------------------------------------------------------------------------------------------------')

    # txt_lists= np.array(np.load('/content/drive/MyDrive/Colab_Notebooks/trial_sentences_img100_partBob_most.npy', allow_pickle=True))
    # image_caption = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]

    txt_lists=np.load('/content/drive/MyDrive/Colab_Notebooks/image_caption_skd.npy', allow_pickle=True) # Saved image captions!!!!!!!
    image_caption = txt_lists[int(img_idx/5)]

    !ps -o pid,cmd | grep java
    sent = client.annotate(image_caption).sentence[0]

    words=[]; conjunctions=[]
    for tok in sent.token:
        try:
            words.append(tok.word)
            conjunctions.append(tok.pos)
        except ValueError:
            pass
    sentences[int(img_idx/5),0]=words
    sentences[int(img_idx/5),1]=conjunctions

    heads=[]; rels=[]; deps=[]
    for edge in sent.enhancedDependencies.edge: #edge 가 곧 rel
      head = sent.token[edge.source - 1].word # 주요한 단어들
      rel = edge.dep  # rel 은 relationship
      dep = sent.token[edge.target - 1].word #각 주요한 단어가 어떤 단어와 relationship 있는지
      print('head: ', head, '  , rel: ', rel, '   , dep: ', dep)
      heads.append(head)
      rels.append(rel)
      deps.append(dep)
    relations[int(img_idx/5),0]=heads
    relations[int(img_idx/5),1]=rels
    relations[int(img_idx/5),2]=deps
    del sent


  np.save('/content/drive/MyDrive/Colab_Notebooks/sentences_skd.npy', sentences)
  np.save('/content/drive/MyDrive/Colab_Notebooks/relations_skd.npy', relations)

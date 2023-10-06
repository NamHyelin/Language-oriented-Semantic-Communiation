from torchvision.utils import save_image
import torchvision
from torchvision import transforms
import warnings
import random
import torch
import torch.nn.functional as F
import re
from torchvision.utils import save_image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import numpy as np
import PIL.Image as Image
from flicker_dataset import FlickrDataset, MyCollate
from Tx_I2T import Transmitter_I2T, Transmitter_I2T_prepare
from RX_T2I import Receiver_T2I, Receiver_T2I_prepare
from llm_att import lang_model, language_att, choose_word, att_sums, attentions_np_m
from lpips_score import LPIPS
from TX_gradcam import Gradcam, load_model_and_preprocess
from bothAtt import att_to_word, att_word_to_word
from bothcrossatt import attention_map


# from transformers import logging
# logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

dataset = FlickrDataset(root_dir="/Flicker8k/Images", caption_path="/Flicker8k/captions.txt", transform=transform)
pad_value = dataset.vocab.stoi["<PAD>"]


# parameters
lpips_th = 0
trial = 1
batchsize = 500
penalty = 0 # add communication penalty to longer words when selecting words to transmit
method = 'prompttune'
importance = 'heads_skd'
only_heads = False # send only heads (SSC)
sort = False # send words in context-sequence

# saves
lpips_score_vector = np.zeros((batchsize, 50))
trial_sentences = np.zeros((batchsize, 50), dtype=object)
was_head = np.zeros((batchsize, 50))



# load model and data
# model_TX, args = Transmitter_I2T_prepare()
model_RX = Receiver_T2I_prepare()
loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))


for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx % 5 == 0:

      print('{}th image of {} nums of images'.format(img_idx, batchsize), '-------------------------------------------------------------------------------------------------------------------------------------------------------')
      transform_img = T.ToPILImage()
      img = transform_img(imgs[img_idx])
      img.save("/content/img.jpg")
      img_ref_path= '/content/img.jpg'

      # image_caption, dense_caption, region_semantic = Transmitter_I2T(model_TX, args, img_ref_path)
      if method == 'allBob_clipTX' or method == 'partBob_clipTX':
        txt_lists = np.load("/save/image_captions_clipTX.npy", allow_pickle=True) # CLIP_Tx_I2T.py
        image_caption = txt_lists[int(img_idx/5)]
      elif importance == 'heads_skd':
        txt_lists = np.load("/save/image_caption_skd.npy", allow_pickle=True) # prompt_engineer.py
        image_caption = txt_lists[int(img_idx/5)]
      else:
        txt_lists = np.array(np.load('/save/image_captions.npy', allow_pickle=True)) # Tx_I2T.py
        image_caption = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]


      if method == 'partBob'or method == 'partBob_clipTX'or importance == 'heads_partbob':
        tokens, attentions_np_m, att_sums = language_att(image_caption, lang_model)

      tokens = image_caption.split(' ')
      print('TX image caption: ', image_caption, '--------------------------------------------------------------------------------------------------------------------------------------------------------')

      comms = 0
      idxs = []
      pre_idx = []
      while len(idxs) < len(tokens):

        if method == 'partBob' or method == 'partBob_clipTX':
          trial = 1
          next_idx = choose_word(att_sums=att_sums, attentions_np_m=attentions_np_m, pre_idx=pre_idx, trial=trial, importance=importance)[0]
          print('next_idx: ', next_idx, '--------------------------------------------------------------------------------------------------------------------------------------------------------------------')

          idxs.append(next_idx)
          if sort == True:
              idxs.sort()
          # Receiver : T2I
          text_prompt = ' '.join([tokens[i] for i in idxs])
          img_gen= Receiver_T2I(model_RX, text_prompt)
          # transform_img = T.ToPILImage()
          # transform_img(img_gen).save("/content/img_gen{}.jpg".format(comms))


          # Calculate LPIPS
          img_ref = Image.open(img_ref_path).convert('RGB')
          lpips_score = LPIPS(img_ref, img_gen)
          print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')


        elif method=='allBob'or method=='allBob_clipTX':
          trial = len(tokens)-len(idxs)
          lpips_cand = np.zeros(trial).tolist()
          lpips_cand_penalty = np.zeros(trial)
          next_idx = [x for x in np.arange(len(tokens)).tolist() if x not in idxs]
          # Penalty by comm load
          tokens_cand = [tokens[t] for t in next_idx]
          penalty_list = [penalty * len(word) for word in tokens_cand]

          for t in range(trial):
            idxs.append(next_idx[t])
            if sort == True:
                idxs.sort()

            # Receiver : T2I
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen= Receiver_T2I(model_RX, text_prompt)

            # Calculate LPIPS
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_cand[t] += LPIPS(img_ref, img_gen)
            lpips_cand_penalty[t] += (LPIPS(img_ref, img_gen) + penalty_list[t])
            idxs.remove(next_idx[t])
            print('idxs: ', idxs)
          next_idx=next_idx[np.argmin(lpips_cand_penalty)]

          print('selected next word: ', tokens[next_idx], '--------------------------------------------------------------------------------------------------------------------------------------------')
          idxs.append(next_idx)
          #save img
          text_prompt = ' '.join([tokens[i] for i in idxs])
          img_gen= Receiver_T2I(model_RX, text_prompt)
          # transform_img = T.ToPILImage()
          # transform_img(img_gen).save("/content/img_gen{}.jpg".format(comms))

          lpips_score = np.min(lpips_cand)



        elif method=='Alice':
          img_ref = Image.open(img_ref_path).convert('RGB')
          if comms == 0:
            modelandprocessors = load_model_and_preprocess("blip_image_text_matching", "base", device='cuda', is_eval=True)
            gradcam_sums = Gradcam(img_ref, image_caption, modelandprocessors)
            idx_cands = np.argsort(gradcam_sums)[::-1]
            del modelandprocessors


          next_idx = idx_cands[comms]
          print('next_idx: ', next_idx, '--------------------------------------------------------------------------------------------------------------------------------------------------------------------')

          idxs.append(next_idx)
          # Receiver : T2I
          text_prompt = ' '.join([tokens[i] for i in idxs])
          img_gen= Receiver_T2I(model_RX, text_prompt)
          # transform_img = T.ToPILImage()
          # transform_img(img_gen).save("/content/img_gen{}.jpg".format(comms))

          # Calculate LPIPS
          lpips_score = LPIPS(img_ref, img_gen)
          print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')


        elif method=='random':
          cand_list = [elem for elem in np.arange(len(tokens)) if elem not in idxs]
          next_idx = np.random.choice(cand_list)
          idxs.append(next_idx)
          text_prompt = ' '.join([tokens[i] for i in idxs])
          img_gen= Receiver_T2I(model_RX, text_prompt)

          # Calculate LPIPS
          img_ref = Image.open(img_ref_path).convert('RGB')
          lpips_score = LPIPS(img_ref, img_gen)
          print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

        elif method == 'bothAtt':
          if comms == 0:
            Aliceatt = np.load('/save/Aliceatts.npy')[int(img_idx/5)] # bothAtt.py
            Bobatt = np.load('/save/Bobatts.npy')[int(img_idx/5)]  # bothAtt.py

            exp_values = np.exp(Aliceatt.flatten() - np.max(Aliceatt.flatten()))
            Aliceatt_soft = (exp_values / np.sum(exp_values)).reshape(Aliceatt.shape)
            exp_values = np.exp(Bobatt.flatten() - np.max(Bobatt.flatten()))
            Bobatt_soft = (exp_values / np.sum(exp_values)).reshape(Bobatt.shape)

            att_diff = Aliceatt_soft-Bobatt_soft

            att_sum_diff=[]
            for t in range(len(tokens)):
              att_sum_diff.append(att_to_word(att_diff, t))
            next_idx = np.argsort(att_sum_diff)[comms]
            idxs.append(next_idx)
            # if sort==True:
            #     idxs.sort()
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen= Receiver_T2I(model_RX, text_prompt)
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_score = LPIPS(img_ref, img_gen)
            print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

          else:
            if importance == 'combination':
              att_comb_diff=[]
              for t in range(len(tokens)):
                if t in idxs:
                  att_comb_diff.append(99)
                else:
                  att_comb_diff.append(att_word_to_word(att_diff, t, idxs[-1]))
              next_idx = np.argmin(att_comb_diff)
              idxs.append(next_idx)
              if sort == True:
                idxs.sort()
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen= Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

            elif importance == 'contribute':
              next_idx = np.argsort(att_sum_diff)[comms]
              idxs.append(next_idx)
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen= Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

            else:
              print('wrong importance setting')
              break


        elif method == 'bothcrossatt':
          if comms == 0:
            TX_attmaps = np.load('/save/TX_gradcams.npy', allow_pickle=True)[int(img_idx/5)] # TX_gradcam.py
            image, RX_heatmap = Receiver_T2I(model_RX, image_caption)
            RX_attmaps = attention_map(RX_heatmap, image_caption)
            diffs=[]
            for t in range(len(tokens)):
              diffs.append(F.l1_loss(TX_attmaps[t], RX_attmaps[t]))
            priors=np.argsort(diffs)

            next_idx=priors[comms]
            idxs.append(next_idx)
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen, _= Receiver_T2I(model_RX, text_prompt)
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_score = LPIPS(img_ref, img_gen)
            print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')
          else:
            next_idx = priors[comms]
            idxs.append(next_idx)
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen, _= Receiver_T2I(model_RX, text_prompt)
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_score = LPIPS(img_ref, img_gen)
            print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

        elif method == 'prompttune':
          tok = [] # to remove duplicate
          tok.append(tokens[0])
          for i in range(len(tokens)-1):
            if tokens[i+1] != tok[-1]:
              tok.append(tokens[i+1])
          tokens = [s.replace('#', '') for s in tok]
          if importance == 'nouns':
            if comms == 0:
              sentences = np.load('/save/sentences.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = [] # only nouns
              for w in range(len(sentences[0])):
                if 'NN' in sentences[1][w]:
                  cands.append(sentences[0][w])
              not_cands = [x for x in sentences[0] if x not in cands and x in tokens]
              cands = [x for x in cands if x in tokens]
              tokens = [x for x in tokens if x in cands or x in not_cands]
              next_idx = tokens.index(cands[0])
              cands.remove(cands[0])

              idxs.append(next_idx)
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')
            else:
              if len(cands) > 0:
                next_idx = tokens.index(cands[0])
                cands.remove(cands[0])
              else:
                next_idx = tokens.index(not_cands[0])
                not_cands.remove(not_cands[0])

              idxs.append(next_idx)
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')


          elif importance == 'nounsverbs':
            if comms == 0:
              sentences = np.load('/save/sentences.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = [] # only nouns
              for w in range(len(sentences[0])):
                if 'NN' in sentences[1][w] or 'V' in sentences[1][w]:
                  cands.append(sentences[0][w])
              not_cands = [x for x in sentences[0] if x not in cands and x in tokens]
              cands = [x for x in cands if x in tokens]
              tokens = [x for x in tokens if x in cands or x in not_cands]
              next_idx = tokens.index(cands[0])
              cands.remove(cands[0])

              idxs.append(next_idx)
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')
            else:
              if len(cands)>0:
                next_idx = tokens.index(cands[0])
                cands.remove(cands[0])
              else:
                next_idx = tokens.index(not_cands[0])
                not_cands.remove(not_cands[0])

              idxs.append(next_idx)
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

          elif importance == 'heads':
            if comms == 0:
              conjunctions = np.load('/save/relations.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = []
              cands.append(conjunctions[0][0])
              for i in range(len(conjunctions[0])-1):
                if conjunctions[0][i+1] != cands[-1] and conjunctions[0][i+1] in tokens:
                  cands.append(conjunctions[0][i+1])
              not_cands = [x for x in tokens if x not in cands]
              tokens = [x for x in tokens if x in cands or x in not_cands]
              next_idx = tokens.index(cands[0])
              was_head[int(img_idx/5)][comms] = 1
              cands.remove(cands[0])

              idxs.append(next_idx)
              if sort == True:
                idxs.sort()
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

            else:
              if len(cands) > 0:
                next_idx = tokens.index(cands[0])
                cands.remove(cands[0])
              else:
                if only_heads == False:
                  next_idx= tokens.index(not_cands[0])
                  not_cands.remove(not_cands[0])

              idxs.append(next_idx)
              if sort==True:
                idxs.sort()
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')



          elif importance=='heads_partbob':
            if comms==0:
              att_prior_idxs=  np.argsort(att_sums)[::-1]
              att_prior= [tokens[ii] for ii in att_prior_idxs]
              conjunctions= np.load('/save/relations.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = []
              cands.append(conjunctions[0][0])
              for i in range(len(conjunctions[0])-1):
                if conjunctions[0][i+1] != cands[-1] and conjunctions[0][i+1] in tokens:
                  cands.append(conjunctions[0][i+1])
              not_cands = [x for x in tokens if x not in cands]
              tokens = [x for x in tokens if x in cands or x in not_cands]
              sorted_cands = sorted(cands, key=lambda word: att_prior.index(word))
              idx_cands = [tokens.index(wo) for wo in sorted_cands]
              did_cands = []

            if len(cands) >= comms+1:
              next_idx = idx_cands[comms]
            else:
              next_idx = tokens.index(not_cands[comms-len(cands)])

            print('next_idx: ' ,next_idx, '--------------------------------------------------------------------------------------------------------------------------------------------------------------------')

            idxs.append(next_idx)
            if sort == True:
                idxs.sort()
            # Receiver : T2I
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen= Receiver_T2I(model_RX, text_prompt)
            img_gen.save("/content/prompttune_heads_mostatt/{}_{}.jpg".format(img_idx,comms))


            # Calculate LPIPS
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_score = LPIPS(img_ref, img_gen)
            print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')





          elif importance == 'heads_random':
            if comms==0:
              conjunctions= np.load('/save/relations.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = []
              cands.append(conjunctions[0][0])
              for i in range(len(conjunctions[0])-1):
                if conjunctions[0][i+1] != cands[-1] and conjunctions[0][i+1] in tokens:
                  cands.append(conjunctions[0][i+1])
              not_cands = [x for x in tokens if x not in cands]
              tokens = [x for x in tokens if x in cands or x in not_cands]

              did_cands = []

            if len(cands) >= comms+1:
              next_idx = tokens.index(random.choice([ca for ca in cands if ca not in did_cands]))
              did_cands.append(tokens[next_idx])
              was_head[img_idx][comms] = 1

            else:
              next_idx = tokens.index(random.choice(not_cands))
              not_cands.remove(tokens[next_idx])

            print('next_idx: ' ,next_idx, '--------------------------------------------------------------------------------------------------------------------------------------------------------------------')

            idxs.append(next_idx)
            if sort == True:
                idxs.sort()
            # Receiver : T2I
            text_prompt = ' '.join([tokens[i] for i in idxs])
            img_gen= Receiver_T2I(model_RX, text_prompt)
            img_gen.save("/content/prompttune_heads_random/{}_{}.jpg".format(img_idx,comms))


            # Calculate LPIPS
            img_ref = Image.open(img_ref_path).convert('RGB')
            lpips_score = LPIPS(img_ref, img_gen)
            print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')


          elif importance == 'heads_skd':
            if comms == 0:
              conjunctions = np.load('/save/relations_skd.npy', allow_pickle=True)[int(img_idx/5)] # prompt_engineer.py
              cands = []
              cands.append(conjunctions[0][0])
              for i in range(len(conjunctions[0])-1):
                if conjunctions[0][i+1] != cands[-1] and conjunctions[0][i+1] in tokens:
                  cands.append(conjunctions[0][i+1])
              not_cands = [x for x in tokens if x not in cands]
              tokens = [x for x in tokens if x in cands or x in not_cands]
              next_idx = tokens.index(cands[0])
              was_head[int(img_idx/5)][comms] = 1
              cands.remove(cands[0])

              idxs.append(next_idx)
              if sort == True:
                idxs.sort()
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')

            else:
              if len(cands) > 0:
                next_idx = tokens.index(cands[0])
                cands.remove(cands[0])
              else:
                if only_heads == False:
                  next_idx= tokens.index(not_cands[0])
                  not_cands.remove(not_cands[0])

              idxs.append(next_idx)
              if sort == True:
                idxs.sort()
              text_prompt = ' '.join([tokens[i] for i in idxs])
              img_gen = Receiver_T2I(model_RX, text_prompt)
              img_ref = Image.open(img_ref_path).convert('RGB')
              lpips_score = LPIPS(img_ref, img_gen)
              print('num of comm: {}, next word: {}, lpips: {}'.format(comms, tokens[next_idx], lpips_score), '------------------------------------------------------------------------------------------------')




          else:
            print('wrong importance setting')
            break





        pre_idx.append(next_idx)
        pre_idx.sort()
        comms += 1

        lpips_score_vector[int(img_idx/5), (comms-1)] = lpips_score
        trial_sentences[int(img_idx/5), (comms-1)] = ' '.join([tokens[i] for i in pre_idx])
        print('Transmitted words so far: ', trial_sentences[int(img_idx/5), (comms-1)])



    np.save('/lpips_score_vector_prompttune_heads_sft.npy', lpips_score_vector)
    np.save('/trial_sentences_prompttune_heads_sft.npy', trial_sentences)
    np.save('/was_head_sft.npy', was_head)


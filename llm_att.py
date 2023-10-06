from transformers import CLIPTokenizer, CLIPTextModel
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained BERT model and tokenizer
model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(model_name)
lang_model = CLIPTextModel.from_pretrained(model_name)



def language_att(text, lang_model):

  # tokens = tokenizer([text], padding=True, return_tensors="pt")
  tokens = tokenizer.tokenize(text)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_ids = torch.tensor([input_ids])

  outputs = lang_model(input_ids, output_attentions=True, output_hidden_states=True)

  # Get the attention weights from the outputs
  attentions = outputs.attentions



  layer_idx=0  # 12 layer 개수
  attentions_np=attentions[layer_idx].detach().numpy()  # 1,12(num heads),8,8 (text length)

  for i in range(len(attentions)):
    if i==0:
      attentions_np = attentions[i].detach().numpy()
    else:
      attentions_np += attentions[i].detach().numpy()

  attentions_np_m= np.mean(attentions_np, 1)
  attentions_np_m = attentions_np[:,2,:,:]
  print(attentions_np_m.shape)

  # print
  torch.set_printoptions(precision=1)
  # print(attentions[layer_idx][:,0,:,:])

  # # plot
  # data = attentions_np_m.squeeze(0)
  # plt.figure(figsize = (3,3))
  # plt.imshow(data, interpolation='nearest')
  # plt.show()



  # Attention values with all words
  text_str= text.split(' ')
  att_sum=[]
  for idx in range(len(text_str)):
    att_sum.append(np.sum(attentions_np_m[0][idx,:])+np.sum(attentions_np_m[0][:,idx])-2*attentions_np_m[0][idx,idx])
  att_sums=np.flip(att_sum)

  return tokens, attentions_np_m, att_sums


# print most un-related word
def atts_to_idx(idx, attentions_np_m):
  atts=[]
  for i in range(attentions_np_m.shape[1]):
    atts.append((attentions_np_m[:,idx,i]+attentions_np_m[:,i,idx])[0])
  return atts

# min_att_idx= np.argmin(atts)

# print('min: ', tokens[min_att_idx])

def choose_word(att_sums, attentions_np_m, pre_idx, trial, importance):
  if len(pre_idx) == 0:
    idx= [np.argmax(att_sums)]
  else:
    if importance=='least':
      atts_idx = atts_to_idx(pre_idx[-1], attentions_np_m)
      for ii in range(len(pre_idx)):
        atts_idx[pre_idx[ii]]=100
      idx= np.argsort(atts_idx)[:trial]

    elif importance=='most':
      atts_idx = atts_to_idx(pre_idx[-1], attentions_np_m)
      for ii in range(len(pre_idx)):
        atts_idx[pre_idx[ii]]=0
      idx= np.flip(np.argsort(atts_idx))[:trial]

    elif importance=='both':
      atts_idx = atts_to_idx(pre_idx[-1], attentions_np_m)
      for ii in range(len(pre_idx)):
        atts_idx[pre_idx[ii]]=100
      idx_least= np.argsort(atts_idx)[:int(trial/2)]

      for ii in range(len(pre_idx)):
        atts_idx[pre_idx[ii]]=0
      idx_most= np.flip(np.argsort(atts_idx))[:(trial-int(trial/2))]

      idx= np.concatenate((idx_least,idx_most))

  return idx





if __name__ == "__main__":
    text= 'a white cat running through a field of flowers'
    tokens, attentions_np_m, att_sums = language_att(text, lang_model)
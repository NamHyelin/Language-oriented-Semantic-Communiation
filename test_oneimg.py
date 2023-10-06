from torchvision.utils import save_image
import torchvision.transforms as T
import numpy as np
import PIL.Image as Image
from Tx_I2T import Transmitter_I2T, Transmitter_I2T_prepare
from RX_T2I import Receiver_T2I, Receiver_T2I_prepare
from llm_att_2 import lang_model, language_att, choose_word
from lpips_score import LPIPS
from ascii_code import string_2_ascii, ascii_2_string, introduce_errors_list




# load model and data
model_TX, args = Transmitter_I2T_prepare()
model_RX = Receiver_T2I_prepare()

# saves
lpips_score_vector=np.zeros((20))
trial_sentences= np.zeros((20), dtype=object)


# Transmitter : I2T
img_ref_path='C:/Users/hyeli/Dropbox/나메렝/projects/SemanGenComm/img/cat.jpg'
image_caption, dense_caption, region_semantic = Transmitter_I2T(model_TX, args, img_ref_path)
tokens, attentions_np_m, att_sums = language_att(image_caption, lang_model)
print('TX image caption: ',image_caption, '--------------------------------------------------------------------------------------------------------------------------------------------------------')

# Receiver : T2I
text_prompt = image_caption
img_gen= Receiver_T2I(model_RX, text_prompt)
img_ref = Image.open(img_ref_path).convert('RGB')
print(LPIPS(img_ref,img_gen))
img_gen.show()
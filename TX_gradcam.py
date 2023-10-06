import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from matplotlib import pyplot as plt
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from scipy.ndimage import filters
import numpy as np
import math
import copy







def gradcam(model, image, text, block_num=6):
    # from Tx_I2T import Transmitter_I2T_prepare
    # from Image2Paragraph_main.utils.util import resize_long_edge
    #
    # device='cuda:0'
    # processor,_=Transmitter_I2T_prepare()
    # model= processor

    tokenized_text = model.processor.tokenizer(text, return_tensors='pt')
    model.image_caption_model.model.text_decoder.bert.encoder.layer[block_num].crossattention.self.save_attention = True

    input = model.image_caption_model.processor(images=image, text=text, return_tensors='pt').to(device, torch.float32)
    output= model.image_caption_model.model(**input, return_dict=True)
    loss = output[0].sum()+output[1].sum()+output[2].sum()

    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(
            tokenized_text.attention_mask.size(0), 1, -1, 1, 1
        )  # (bsz,1,token_len, 1,1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        # grads and cams [bsz, num_head, seq_len, image_patch]
        grads = model.image_caption_model.model.text_decoder.bert.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams = model.image_caption_model.model.text_decoder.bert.encoder.layer[block_num].crossattention.self.get_attention_map()

        # assume using vit with 576 num image patch
        cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
        grads = (
                grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24)
                * mask
        )

        gradcams = cams * grads
        gradcam_list = []

        for ind in range(image.size(0)):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            # [enc token gradcam, average gradcam across token, gradcam for individual token]
            gradcam = torch.cat(
                (
                    gradcam[0:1, :],
                    gradcam[1 : token_length_ + 1, :].sum(dim=0, keepdim=True)
                    / token_length_,
                    gradcam[1:, :],
                )
            )
            gradcam_list.append(gradcam)

    return gradcam_list, output



def Gradcam(image, text, modelandprocessors, blocknum):
    dst_w = 720
    w, h = image.size
    scaling_factor = dst_w / w

    resized_img = image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255

    model, vis_processors, text_processors = modelandprocessors

    # Preprocess image and text inputs
    img = vis_processors["eval"](image).unsqueeze(0).to(device)
    txt = text_processors["eval"](text)

    # Compute GradCam
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=blocknum)  # 7

    # Average GradCam for the full image
    # avg_gradcam = getAttMap(norm_img, gradcam[0][1], blur=True)

    # # GradCam for each token
    # num_image = len(txt_tokens.input_ids[0]) - 2
    # fig, ax = plt.subplots(1, num_image, figsize=(6, num_image))

    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])

    gradcam_list = copy.deepcopy(list(gradcam[0][2:-1]))


    # # Plot
    # for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
    #     word = model.tokenizer.decode([token_id])
    #     # threshold=0.04
    #     # gradcam[gradcam < threshold] = 0
    #     gradcam_image = getAttMap(norm_img, gradcam, blur=True)
    #     ax[i].imshow(gradcam_image)
    #     ax[i].set_yticks([])
    #     ax[i].set_xticks([])
    #     ax[i].set_xlabel(word)

    sums = []
    gradcams=[]
    for idx in range(len(gradcam_list)):
        a = gradcam_list[idx].clone()

        # a = filters.gaussian_filter(a, 0.02 * 3)
        # a -= a.min()
        # a /= a.max()

        sums.append(torch.sum(a))
        gradcams.append(a)

        # plt.imshow(a)
        # plt.show()

    return sums, gradcams


def plot_gradcam(gradcam_list, prompt):
    tokens=prompt.split(' ')

    for idx in range(len(tokens)):
        a = gradcam_list[idx].clone()

        a = filters.gaussian_filter(a, 0.02 * 3)
        a -= a.min()
        a /= a.max()

        plt.subplot(1, len(tokens), idx+1)
        plt.title(tokens[idx])
        plt.imshow(a)

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelandprocessors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True) #large

    text = 'a large metal sculpture in front of a building'  # caption[0]
    image = Image.open('C:/Users/hyeli/Dropbox/나메렝/projects/SemanGenComm/img/save/ref_input/content/ref_input/26.jpg')
    blocknum = 7

    gradcam_sums, gradcams = Gradcam(image, text, modelandprocessors, blocknum)
    torch.save(torch.stack(gradcams), 'C:/Users/hyeli/Dropbox/나메렝/projects/SemanGenComm/TX_gradcams.pth')
    plot_gradcam(gradcams, text)
    priority = np.argsort(gradcam_sums)[::-1]


##### Save gradcams
'''
transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

dataset = FlickrDataset(root_dir="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/Images", caption_path="/content/drive/MyDrive/Colab_Notebooks/Flicker8k/captions.txt", transform=transform)
pad_value = dataset.vocab.stoi["<PAD>"]


# parameters
batchsize=500

loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=1, shuffle=False, pin_memory=True, collate_fn=MyCollate(pad_value))

TX_gradcams= []
for batch_idx, (imgs, captions) in enumerate(loader):
  for img_idx in range(imgs.size()[0]):
    if img_idx%5==0 :
      print('{}th image of {} nums of images'.format(img_idx, batchsize),'--------------------------------------------------------------------------------------------------------------------------------------------------------')
      transform_img= T.ToPILImage()
      img = transform_img(imgs[img_idx])

      txt_lists= np.array(np.load('/content/drive/MyDrive/Colab_Notebooks/trial_sentences_img100_partBob_most.npy', allow_pickle=True))
      image_caption = [a for a in txt_lists[int(img_idx/5)] if a!=0][-1]

      sums,gradcams =  Gradcam(img, image_caption)
      TX_gradcams.append(gradcams)
      np.save('/content/drive/MyDrive/Colab_Notebooks/TX_gradcams.npy', TX_gradcams)
  if batch_idx==1:
    break
    
    '''
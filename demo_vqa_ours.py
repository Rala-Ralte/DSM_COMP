# import gradio as gr
import torch
import cv2
import copy
import time
import requests
import io
import numpy as np
import re
import json
import urllib.request
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import diags, csr_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pymatting.util.util import row_sum

from PIL import Image
from vqa_data import VQATorchDataset  # For accessing dataset with embeddings

from meter.config import ex
from meter.modules import METERTransformerSS

from meter.transforms import vit_transform, clip_transform, clip_transform_randaug
from meter.datamodules.datamodule_base import get_pretrained_tokenizer
from scipy.stats import skew

from spectral.ExplanationGenerator import GeneratorOurs
from spectral.get_fev import get_grad_eigs


# @ex.automain


def main1(_config, item, model=None, viz=True, is_pert=False, tokenizer=None, dataset=None):
    # print(type(_config))
    if is_pert:
        img_path = item['img_id'] + '.jpg'
        question = item['sent']
    else:
        img_path, question = item

    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 0,
        "mlm": 1,
        "mpp": 0,
        "vqa": 1,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }

    if not is_pert:
        tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    url = 'spectral/vqa_dict.json'
    with open(url) as f:
        id2ans = json.load(f)

    _config.update({
        "loss_names": loss_names,
    })

    if not is_pert:
        model = METERTransformerSS(_config)
        model.setup("test")
        model.eval()

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    IMG_SIZE = 576
    method_type = _config["method_name"]

    def infer(url, text):
        try:
            if "http" in url:
                res = requests.get(url)
                image = Image.open(io.BytesIO(res.content)).convert("RGB")
            else:
                image = Image.open(url)
            orig_shape = np.array(image).shape
            img = clip_transform(size=IMG_SIZE)(image)
            img = img.unsqueeze(0).to(device)
        except Exception as e:
            print(f"EXCEPTION: {e}")
            return False

        batch = {"text": [text], "image": [img]}
        encoded = tokenizer(batch["text"])
        text_tokens = tokenizer.tokenize(batch["text"][0])
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)

        if not is_pert:
            ret = model.infer(batch)
        else:
            ret = model.infer_mega(batch)

        vqa_logits = model.vqa_classifier(ret["cls_feats"])
        answer = id2ans[str(vqa_logits.argmax().item())]
        output = vqa_logits

        # Get initial embeddings for DecompX (using dataset if provided, else infer)
        if dataset is not None:
            # Assume dataset is VQATorchDataset with get_item_with_embeddings
            item_data = {"img_id": os.path.basename(url).replace('.jpg', ''), "sent": text}
            _, feats, boxes, _, _, text_emb, image_emb = dataset.get_item_with_embeddings(0, model)  # Use first item as proxy
            image_emb = model.embed_image(feats, boxes)  # Recompute with current image if needed
        else:
            # Fallback: Use ret['all_image_feats'][0] and text embeddings from batch
            text_emb = ret['all_text_feats'][0]  # Assume first layer or adjust
            image_emb = ret['all_image_feats'][0]

        # Initialize explanation generator
        ours = GeneratorOurs(model)

        # Compute relevance based on method_type
        if method_type == "dsm":
            text_rel, image_rel = ours.generate_ours_dsm(text_emb, image_emb, device=model.device)
        elif method_type == "dsm_grad":
            text_rel, image_rel = ours.generate_ours_dsm_grad(ret['all_image_feats'], ret['all_text_feats'], device=model.device)
        elif method_type == "dsm_grad_cam":
            text_rel, image_rel = ours.generate_ours_dsm_grad_cam(ret['all_image_feats'], ret['all_text_feats'], device=model.device)
        elif method_type == "spectral_decompx":
            # New hybrid method with DecompX
            variant = _config.get("decompx_variant", "grad")  # Default to "grad"; configurable
            target_class = _config.get("target_class", vqa_logits.argmax().item())  # Use predicted class
            text_rel, image_rel = ours.generate_spectral_decompx(image_emb, text_emb, how_many=5, device=model.device, variant=variant, target_class=target_class)

        return answer, text_rel, image_rel, img, text_tokens

    result, text_relevance, image_relevance, image, text_tokens = infer(img_path, question)

    if viz:
        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(image_relevance, size=IMG_SIZE, mode='bilinear')
        image_relevance = image_relevance.reshape(IMG_SIZE, IMG_SIZE).cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        image = image[0].permute(1, 2, 0).cpu().numpy()
        image = (image - image.min()) / (image.max() - image.min())
        vis = show_cam_on_image(image, image_relevance)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

        fig, axs = plt.subplots(ncols=2, figsize=(20, 5))
        axs[0].imshow(vis)
        axs[0].axis('off')
        axs[0].set_title(f'({method_type}) Image Relevance')

        ti = axs[1].imshow(text_relevance.unsqueeze(dim=0).numpy())
        axs[1].set_title(f'({method_type}) Word Importance')
        plt.sca(axs[1])
        plt.xticks(np.arange(len(text_tokens) + 2), ['[CLS]'] + text_tokens + ['[SEP]'])
        plt.colorbar(ti, orientation="horizontal", ax=axs[1])
        plt.show()

    if is_pert:
        return text_relevance, image_relevance
    else:
        return text_relevance, image_relevance, result
    


if __name__ == '__main__':
    @ex.automain
    def main(_config):
        test_img = _config['img']
        test_question = _config['question']

        if test_img == '' or test_question == '':
            print("Provide an image and a corresponding question for VQA")
        else:
            # Initialize dataset for embedding access
            dataset = VQATorchDataset(VQADataset("train"))  # Use a small dataset; adjust split as needed
            item = (test_img, test_question)
            _config["method_name"] = _config.get("method_name", "spectral_decompx")  # Default to new method
            _config["decompx_variant"] = _config.get("decompx_variant", "grad")  # Configurable variant
            _config["target_class"] = _config.get("target_class", None)  # Optional target class
            text_relevance, image_relevance, answer = main1(_config, item, viz=True, dataset=dataset)
            print(f"QUESTION: {test_question}\nANSWER: {answer}")

    

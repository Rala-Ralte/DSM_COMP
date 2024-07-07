import numpy as np
import torch
import copy
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import eig

import torch.nn.functional as F
from torch.nn import Linear

from pymatting.util.util import row_sum
from scipy.sparse import diags
from scipy.stats import skew
# from .eigenshuffle import eigenshuffle
# from sentence_transformers import SentenceTransformer
# from torch.nn import CosineSimilarity as CosSim

from spectral.get_fev import get_eigs, get_grad_eigs, avg_heads


class GeneratorOurs:
    def __init__(self, model_usage):
        self.model = model_usage


    def generate_ours_dsm(self, image_feats, text_feats, how_many = 5, device = "cpu"):

        image_rel = get_eigs(image_feats, "image", how_many, device).to(device)
        text_rel = get_eigs(text_feats, "text", how_many, device).to(device)

        return text_rel, image_rel




    def generate_ours_dsm_grad(self, image_feat_list, text_feat_list, how_many = 5, device = "cpu"):
        fevs = []
        for i, feats in enumerate(image_feat_list):
            grad = self.model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach()
            fev = get_grad_eigs(feats, "image", grad, device, how_many)
            fevs.append( fev )
        
        image_rel = torch.stack(fevs, dim=0).sum(dim=0)


        fevs = []
        for i, feats in enumerate(text_feat_list):
            grad = self.model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach()
            fev = get_grad_eigs(feats, "text", grad, device, how_many)
            fevs.append( fev )
        
        text_rel = torch.stack(fevs, dim=0).sum(dim=0)

        return text_rel, image_rel




    def generate_ours_dsm_grad_cam(self, image_feat_list, text_feat_list, how_many = 5, device = "cpu"):
        fevs = []
        for i, feats in enumerate(image_feat_list):
            fev = get_eigs(feats, "image", how_many, device)
            grad = self.model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach()[:, :, 1:, 1:]
            cam = self.model.cross_modal_image_layers[i].attention.self.get_attention_map().detach()[:, :, 1:, 1:]
            cam = avg_heads(cam, grad)

            fev = fev.to(device)
            fev = cam @ fev.unsqueeze(1)
            fev = fev[:, 0]
            fevs.append( torch.abs(fev) )

        image_rel = torch.stack(fevs, dim=0).sum(dim=0)
        

        fevs = []
        for i, feats in enumerate(text_feat_list):
            fev = get_eigs(feats, "text", how_many, device)
            fev = fev[1:-1]

            grad = self.model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach()[:, :, 1:-1, 1:-1]
            cam = self.model.cross_modal_text_layers[i].attention.self.get_attention_map().detach()[:, :, 1:-1, 1:-1]
            cam = avg_heads(cam, grad)

            fev = fev.to(device)
            fev = cam @ fev.unsqueeze(1)
            fev = fev[:, 0]
            fev = torch.cat( ( torch.zeros(1).to(device), fev, torch.zeros(1).to(device) ) )
            fevs.append( torch.abs(fev) )
        
        
        text_rel = torch.stack(fevs, dim=0).sum(dim=0)

        return text_rel, image_rel




 
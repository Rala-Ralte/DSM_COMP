import numpy as np
import torch
import copy
import math
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

from spectral.get_fev import get_eigs, get_grad_eigs, avg_heads, get_grad_cam_eigs


class GeneratorOurs:
    def __init__(self, model_usage):
        self.model = model_usage

    def abs_dot_bias_decomp(self, bias, no_bias_terms):
        # AbsDot for bias decomposition (paper Eq. 5)
        # no_bias_terms: (seq, input_n, hidden)
        dots = torch.abs(torch.einsum('sih,h->si', no_bias_terms, bias))  # |b · z_{i<=k}^{NoBias}|
        weights = dots / (dots.sum(dim=1, keepdim=True) + 1e-12)  # omega_k
        return weights.unsqueeze(2) * bias.unsqueeze(0).unsqueeze(0)  # omega_k * b (decomposed per k)

    def decompose_attn(self, input_decomp, attn_layer, is_cross=False, key_decomp=None):
        # Decompose multi-head attention (self or cross) - paper Eqs. 3-6
        # input_decomp: (seq_q, total_input_n, hidden) - decomposed query input
        # For cross, provide key_decomp: (seq_k, total_input_n, hidden)
        device = input_decomp.device
        seq_q, total_input_n, hidden = input_decomp.shape
        input_q = input_decomp.sum(dim=1)  # Aggregate for forward
        if is_cross:
            seq_k = key_decomp.shape[0]
            input_k = key_decomp.sum(dim=1)
        else:
            seq_k = seq_q
            input_k = input_q

        # Get attention probs (alpha) from layer
        alpha = attn_layer.self.get_attention_map().detach() if hasattr(attn_layer.self, 'get_attention_map') else None
        if alpha is None:
            # Fallback: compute alpha (assume single head for simplicity; extend to multi-head)
            q = attn_layer.self.query(input_q)
            k = attn_layer.self.key(input_k)
            alpha = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)

        # Value no-bias decomposition
        v_no_bias = attn_layer.self.value(input_k) - attn_layer.self.value.bias.unsqueeze(0)  # Subtract bias for NoBias
        z_no_bias = torch.zeros(seq_q, total_input_n, hidden, device=device)
        for k in range(total_input_n):
            if is_cross:
                v_decomp_k = attn_layer.self.value(key_decomp[:, k, :])
            else:
                v_decomp_k = attn_layer.self.value(input_decomp[:, k, :])
            weighted = alpha @ v_decomp_k
            z_no_bias[:, k, :] += attn_layer.output.dense(weighted)

        # Decompose biases: b_v and b_O combined as b_Att
        b_v = attn_layer.self.value.bias
        b_O = attn_layer.output.dense.bias
        b_Att = attn_layer.output.dense(b_v.unsqueeze(0)).squeeze(0) + b_O
        omega = torch.abs(torch.einsum('sih,h->si', z_no_bias, b_Att)) / (torch.abs(torch.einsum('sih,h->si', z_no_bias, b_Att)).sum(1, keepdim=True) + 1e-12)
        z_decomp = z_no_bias + omega.unsqueeze(2) * b_Att.unsqueeze(0).unsqueeze(0)
        return z_decomp

    def decompose_ln(self, input_decomp, ln_layer, residual_decomp=None):
        # Decompose LayerNorm (paper Eq. 8)
        # input_decomp: (seq, total_input_n, hidden)
        # If residual, add residual_decomp before LN
        if residual_decomp is not None:
            input_decomp = input_decomp + residual_decomp
        input_agg = input_decomp.sum(dim=1)
        mean = input_agg.mean(-1, keepdim=True)
        std = input_agg.std(-1, keepdim=True) + 1e-6
        centered_decomp = input_decomp - mean.unsqueeze(1) / input_decomp.shape[1]  # Approximate uniform decomposition of mean
        scaled_decomp = centered_decomp / std.unsqueeze(1)
        gamma_decomp = scaled_decomp * ln_layer.weight.unsqueeze(0).unsqueeze(1)
        beta_decomp = self.abs_dot_bias_decomp(ln_layer.bias, gamma_decomp)
        return gamma_decomp + beta_decomp

    def decompose_ffn(self, input_decomp, ffn_layer):
        # Decompose FFN (paper Eqs. 9-11), assuming GELU activation
        # input_decomp: (seq, total_input_n, hidden)
        input_agg = input_decomp.sum(dim=1)
        zeta_agg = ffn_layer.dense1(input_agg)  # fc1
        theta = F.gelu(zeta_agg) / (zeta_agg + 1e-12)  # Monotonic linearization
        zeta_no_bias_decomp = ffn_layer.dense1(input_decomp) - ffn_layer.dense1.bias.unsqueeze(0).unsqueeze(0)  # NoBias
        omega = torch.abs(torch.einsum('sih,h->si', zeta_no_bias_decomp, ffn_layer.dense1.bias)) / (torch.abs(torch.einsum('sih,h->si', zeta_no_bias_decomp, ffn_layer.dense1.bias)).sum(1, keepdim=True) + 1e-12)
        zeta_decomp = zeta_no_bias_decomp + omega.unsqueeze(2) * ffn_layer.dense1.bias.unsqueeze(0).unsqueeze(0)
        act_decomp = theta.unsqueeze(1) * zeta_decomp
        out_no_bias_decomp = ffn_layer.dense2(act_decomp) - ffn_layer.dense2.bias.unsqueeze(0).unsqueeze(0)
        omega = torch.abs(torch.einsum('sih,h->si', out_no_bias_decomp, ffn_layer.dense2.bias)) / (torch.abs(torch.einsum('sih,h->si', out_no_bias_decomp, ffn_layer.dense2.bias)).sum(1, keepdim=True) + 1e-12)
        out_decomp = out_no_bias_decomp + omega.unsqueeze(2) * ffn_layer.dense2.bias.unsqueeze(0).unsqueeze(0)
        return out_decomp
    
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
            grad = self.model.cross_modal_image_layers[i].attention.self.get_attn_gradients().detach()
            cam = self.model.cross_modal_image_layers[i].attention.self.get_attention_map().detach()
            fev = get_grad_cam_eigs(feats, "image", grad, cam, device, how_many)
            fevs.append( torch.abs(fev) )

        image_rel = torch.stack(fevs, dim=0).sum(dim=0)
        

        fevs = []
        for i, feats in enumerate(text_feat_list):
            grad = self.model.cross_modal_text_layers[i].attention.self.get_attn_gradients().detach()
            cam = self.model.cross_modal_text_layers[i].attention.self.get_attention_map().detach()
            fev = get_grad_cam_eigs(feats, "text", grad, cam, device, how_many)
            fevs.append( torch.abs(fev) )
        
        
        text_rel = torch.stack(fevs, dim=0).sum(dim=0)

        return text_rel, image_rel
    
    def generate_spectral_decompx(self, image_emb, text_emb, how_many=5, device="cpu", variant="pure", target_class=None):
        # Hybrid: Propagate DecompX decompositions, then apply DSMI spectral on fusion feats
        # image_emb/text_emb: Initial embeddings (seq_img, hidden), (seq_text, hidden)
        # Returns text_rel, image_rel as in DSMI, but enhanced with DecompX contributions
        total_input_n = text_emb.shape[0] + image_emb.shape[0]
        text_decomp = torch.zeros(text_emb.shape[0], total_input_n, text_emb.shape[1], device=device)
        for i in range(text_emb.shape[0]):
            text_decomp[i, i, :] = text_emb[i]
        image_decomp = torch.zeros(image_emb.shape[0], total_input_n, image_emb.shape[1], device=device)
        for i in range(image_emb.shape[0]):
            image_decomp[i, text_emb.shape[0] + i, :] = image_emb[i]

        text_feat_list_decomp = []
        image_feat_list_decomp = []

        # Propagate through text self-attn layers (example; adjust for your model)
        for layer in self.model.text_transformer.encoder.layer:
            z_attn = self.decompose_attn(text_decomp, layer.attention)
            z_res = z_attn + text_decomp  # Residual
            z_ln1 = self.decompose_ln(z_res, layer.attention.output.LayerNorm)
            z_ffn = self.decompose_ffn(z_ln1, layer.intermediate)  # Adjust if intermediate/output separate
            z_res2 = z_ffn + z_ln1
            text_decomp = self.decompose_ln(z_res2, layer.output.LayerNorm)

        # Cross-modal layers
        for t_layer, i_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            # Image self-attn
            i_z_attn = self.decompose_attn(image_decomp, i_layer.attention)
            i_z_res = i_z_attn + image_decomp
            i_z_ln1 = self.decompose_ln(i_z_res, i_layer.attention.output.LayerNorm)
            i_z_ffn = self.decompose_ffn(i_z_ln1, i_layer.intermediate)
            i_z_res2 = i_z_ffn + i_z_ln1
            image_decomp = self.decompose_ln(i_z_res2, i_layer.output.LayerNorm)

            # Cross-attn for text (query=text, key=image)
            t_z_cross = self.decompose_attn(text_decomp, t_layer.crossattention, is_cross=True, key_decomp=image_decomp)
            t_z_res = t_z_cross + text_decomp
            t_z_ln1 = self.decompose_ln(t_z_res, t_layer.crossattention.output.LayerNorm)
            t_z_ffn = self.decompose_ffn(t_z_ln1, t_layer.intermediate)
            t_z_res2 = t_z_ffn + t_z_ln1
            text_decomp = self.decompose_ln(t_z_res2, t_layer.output.LayerNorm)

            # Append decomposed feats (sum over contributions for DSMI input)
            text_feat_list_decomp.append(text_decomp.sum(dim=1))
            image_feat_list_decomp.append(image_decomp.sum(dim=1))

        # Classification head (assume on text [CLS], model.logit_fc is the head)
        cls_decomp = text_decomp[0]  # (total_input_n, hidden)
        # Propagate through head (assume linear - tanh - linear)
        pool_agg = torch.tanh(self.model.logit_fc[0](cls_decomp.sum(0)))
        theta = pool_agg / (self.model.logit_fc[0](cls_decomp.sum(0)) + 1e-12)
        pool_no_bias = theta * self.model.logit_fc[0](cls_decomp)  # No bias
        omega = torch.abs(torch.einsum('ih,h->i', pool_no_bias, self.model.logit_fc[0].bias)) / (torch.abs(torch.einsum('ih,h->i', pool_no_bias, self.model.logit_fc[0].bias)).sum() + 1e-12)
        pool_decomp = pool_no_bias + omega.unsqueeze(1) * self.model.logit_fc[0].bias
        logit_no_bias = self.model.logit_fc[2](pool_decomp)
        omega = torch.abs(torch.einsum('ic,c->i', logit_no_bias, self.model.logit_fc[2].bias)) / (torch.abs(torch.einsum('ic,c->i', logit_no_bias, self.model.logit_fc[2].bias)).sum() + 1e-12)
        logit_decomp = logit_no_bias + omega.unsqueeze(1) * self.model.logit_fc[2].bias  # (total_input_n, num_classes)

        # Apply DSMI on decomposed feat lists
        text_rels = []
        image_rels = []
        for t_feats, i_feats in zip(text_feat_list_decomp, image_feat_list_decomp):
            if variant == "pure":
                t_rel = get_eigs(t_feats, "text", how_many, device)
                i_rel = get_eigs(i_feats, "image", how_many, device)
            elif variant == "grad":
                grad_t = self.model.cross_modal_text_layers[0].attention.self.get_attn_gradients().detach()  # Use first layer; adjust
                grad_i = self.model.cross_modal_image_layers[0].attention.self.get_attn_gradients().detach()
                t_rel = get_grad_eigs(t_feats, "text", grad_t, device, how_many)
                i_rel = get_grad_eigs(i_feats, "image", grad_i, device, how_many)
            elif variant == "grad_cam":
                grad_t = self.model.cross_modal_text_layers[0].attention.self.get_attn_gradients().detach()
                cam_t = self.model.cross_modal_text_layers[0].attention.self.get_attention_map().detach()
                grad_i = self.model.cross_modal_image_layers[0].attention.self.get_attn_gradients().detach()
                cam_i = self.model.cross_modal_image_layers[0].attention.self.get_attention_map().detach()
                t_rel = get_grad_cam_eigs(t_feats, "text", grad_t, cam_t, device, how_many)
                i_rel = get_grad_cam_eigs(i_feats, "image", grad_i, cam_i, device, how_many)
            else:
                raise ValueError("Invalid variant: choose 'pure', 'grad', or 'grad_cam'")

            text_rels.append(t_rel)
            image_rels.append(i_rel)

        text_rel = torch.stack(text_rels, dim=0).sum(dim=0)
        image_rel = torch.stack(image_rels, dim=0).sum(dim=0)

        # Combine with DecompX class contributions if target_class provided
        if target_class is not None:
            text_n = text_emb.shape[0]
            text_rel *= logit_decomp[:text_n, target_class]
            image_rel *= logit_decomp[text_n:, target_class]

        return text_rel, image_rel    




 

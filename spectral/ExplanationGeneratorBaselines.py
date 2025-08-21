import torch
import copy
import math  
import torch.nn.functional as F

def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention


def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.matmul(cam_ss, R_sq)
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition


def apply_mm_attention_rules(R_ss, R_qq, R_qs, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
    R_ss_normalized = R_ss
    R_qq_normalized = R_qq
    if apply_normalization:
        R_ss_normalized = handle_residual(R_ss)
        R_qq_normalized = handle_residual(R_qq)
    R_sq_addition = torch.matmul(R_ss_normalized.t(), torch.matmul(cam_sq, R_qq_normalized))
    if not apply_self_in_rule_10:
        R_sq_addition = cam_sq
    R_ss_addition = torch.matmul(cam_sq, R_qs)
    return R_sq_addition, R_ss_addition

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    # computing R hat
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    # print(f"self_attn (handle residual): {self_attention.shape} | diag idx: {diag_idx}")
    # assert self_attention[diag_idx, diag_idx].min() >= 0
    # normalizing R hat
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention


class GeneratorBaselines:
    def __init__(self, model, normalize_self_attention, apply_self_in_rule_10):
        self.model = model
        self.normalize_self_attention = normalize_self_attention 
        self.apply_self_in_rule_10 = apply_self_in_rule_10

    def abs_dot_bias_decomp(self, bias, no_bias_terms):
        # AbsDot for bias decomposition (DecompX Eq. 5)
        # no_bias_terms: (seq, input_n, hidden)
        dots = torch.abs(torch.einsum('sih,h->si', no_bias_terms, bias))  # |b · z_{i<=k}^{NoBias}|
        weights = dots / (dots.sum(dim=1, keepdim=True) + 1e-12)  # omega_k
        return weights.unsqueeze(2) * bias.unsqueeze(0).unsqueeze(0)  # omega_k * b

    def decompose_attn(self, input_decomp, attn_layer, is_cross=False, key_decomp=None):
        # Decompose multi-head attention (self/cross) - DecompX Eqs. 3-6
        device = input_decomp.device
        seq_q, total_input_n, hidden = input_decomp.shape
        input_q = input_decomp.sum(dim=1)
        if is_cross:
            seq_k = key_decomp.shape[0]
            input_k = key_decomp.sum(dim=1)
        else:
            seq_k = seq_q
            input_k = input_q

        # Get alpha; use existing get_attention_map if available, else fallback
        alpha = attn_layer.self.get_attention_map().detach() if hasattr(attn_layer.self, 'get_attention_map') else None
        if alpha is None:
            q = attn_layer.self.query(input_q)
            k = attn_layer.self.key(input_k)
            alpha = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1)), dim=-1)

        # Value decomposition (NoBias)
        z_no_bias = torch.zeros(seq_q, total_input_n, hidden, device=device)
        for k in range(total_input_n):
            if is_cross:
                v_decomp_k = attn_layer.self.value(key_decomp[:, k, :])
            else:
                v_decomp_k = attn_layer.self.value(input_decomp[:, k, :])
            weighted = alpha @ v_decomp_k
            z_no_bias[:, k, :] += attn_layer.output.dense(weighted)

        # Bias decomposition
        b_v = attn_layer.self.value.bias
        b_O = attn_layer.output.dense.bias
        b_Att = attn_layer.output.dense(b_v.unsqueeze(0)).squeeze(0) + b_O
        omega = torch.abs(torch.einsum('sih,h->si', z_no_bias, b_Att)) / (torch.abs(torch.einsum('sih,h->si', z_no_bias, b_Att)).sum(1, keepdim=True) + 1e-12)
        z_decomp = z_no_bias + omega.unsqueeze(2) * b_Att.unsqueeze(0).unsqueeze(0)
        return z_decomp

    def decompose_ln(self, input_decomp, ln_layer, residual_decomp=None):
        # Decompose LayerNorm - DecompX Eq. 8
        if residual_decomp is not None:
            input_decomp = input_decomp + residual_decomp
        input_agg = input_decomp.sum(dim=1)
        mean = input_agg.mean(-1, keepdim=True)
        std = input_agg.std(-1, keepdim=True) + 1e-6
        centered_decomp = input_decomp - mean.unsqueeze(1) / input_decomp.shape[1]  # Approx decomposition of mean
        scaled_decomp = centered_decomp / std.unsqueeze(1)
        gamma_decomp = scaled_decomp * ln_layer.weight.unsqueeze(0).unsqueeze(1)
        beta_decomp = self.abs_dot_bias_decomp(ln_layer.bias, gamma_decomp)
        return gamma_decomp + beta_decomp

    def decompose_ffn(self, input_decomp, intermediate_layer, output_layer):
        # Decompose FFN - DecompX Eqs. 9-11 (assume intermediate.dense = fc1, output.dense = fc2)
        input_agg = input_decomp.sum(dim=1)
        zeta_agg = intermediate_layer.dense(input_agg)  # fc1
        theta = F.gelu(zeta_agg) / (zeta_agg + 1e-12)
        zeta_no_bias_decomp = intermediate_layer.dense(input_decomp) - intermediate_layer.dense.bias.unsqueeze(0).unsqueeze(0) if hasattr(intermediate_layer.dense, 'bias') else intermediate_layer.dense(input_decomp)
        omega = torch.abs(torch.einsum('sih,h->si', zeta_no_bias_decomp, intermediate_layer.dense.bias)) / (torch.abs(torch.einsum('sih,h->si', zeta_no_bias_decomp, intermediate_layer.dense.bias)).sum(1, keepdim=True) + 1e-12)
        zeta_decomp = zeta_no_bias_decomp + omega.unsqueeze(2) * intermediate_layer.dense.bias.unsqueeze(0).unsqueeze(0)
        act_decomp = theta.unsqueeze(1) * zeta_decomp
        out_no_bias_decomp = output_layer.dense(act_decomp) - output_layer.dense.bias.unsqueeze(0).unsqueeze(0) if hasattr(output_layer.dense, 'bias') else output_layer.dense(act_decomp)
        omega = torch.abs(torch.einsum('sih,h->si', out_no_bias_decomp, output_layer.dense.bias)) / (torch.abs(torch.einsum('sih,h->si', out_no_bias_decomp, output_layer.dense.bias)).sum(1, keepdim=True) + 1e-12)
        out_decomp = out_no_bias_decomp + omega.unsqueeze(2) * output_layer.dense.bias.unsqueeze(0).unsqueeze(0)
        return out_decomp

    def handle_self_attention_image(self, blk):
        cam = blk.attention.self.get_attention_map().detach()
        grad = blk.attention.self.get_attn_gradients().detach()
        cam = avg_heads(cam, grad)
        R_i_i_add, R_i_t_add = apply_self_attention_rules(self.R_i_i, self.R_i_t, cam)
        self.R_i_i += R_i_i_add
        self.R_i_t += R_i_t_add


    def handle_co_attn_image(self, block):
        
        cam_i_t = block.crossattention.self.get_attention_map().detach()
        grad_i_t = block.crossattention.self.get_attn_gradients().detach()
        cam_i_t = avg_heads(cam_i_t, grad_i_t)
        R_i_t_addition, R_i_i_addition = apply_mm_attention_rules(self.R_i_i, self.R_t_t, self.R_t_i, cam_i_t,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        return R_i_t_addition, R_i_i_addition
    

    def handle_self_attention_lang(self, blk):
        cam = blk.attention.self.get_attention_map().detach()
        grad = blk.attention.self.get_attn_gradients().detach()
        # print(grad.shape, cam.shape)
        cam = avg_heads(cam, grad)
        # print(self.R_t_t[0])
        R_t_t_add, R_t_i_add = apply_self_attention_rules(self.R_t_t, self.R_t_i, cam)
        self.R_t_t += R_t_t_add
        self.R_t_i += R_t_i_add
        # print(f"R_t_t in lang self attn: {self.R_t_t[0]}")


    def handle_co_attn_lang(self, block):
        
        cam_t_i = block.crossattention.self.get_attention_map().detach()
        grad_t_i = block.crossattention.self.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        R_t_i_addition, R_t_t_addition = apply_mm_attention_rules(self.R_t_t, self.R_i_i, self.R_i_t, cam_t_i,
                                                                  apply_normalization=self.normalize_self_attention,
                                                                  apply_self_in_rule_10=self.apply_self_in_rule_10)
        # print(f"R_t_t_addition in lang co attn: {self.R_t_t[0]}")
        
        return R_t_i_addition, R_t_t_addition

    def generate_relevance_maps (self, text_tokens, image_tokens, device):
        # text self attention matrix
        # text_tokens -= 2
        # image_tokens -= 1 

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # text
        blocks = self.model.text_transformer.encoder.layer
        for blk in blocks:
            self.handle_self_attention_lang(blk)


        count = 0
        for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            self.handle_self_attention_image(image_layer)
            R_i_t_addition, R_i_i_addition = self.handle_co_attn_image(image_layer)
            self.R_i_t += R_i_t_addition
            self.R_i_i += R_i_i_addition

            self.handle_self_attention_lang(text_layer)
            R_t_i_addition, R_t_t_addition = self.handle_co_attn_lang(text_layer)
            self.R_t_i += R_t_i_addition
            self.R_t_t += R_t_t_addition

            # print(self.R_t_t[0])
            count += 1
            # if count == 1:
                # break
        self.R_t_t[0, 0] = 0
        self.R_i_i[0, 0] = 0 #baka
        self.R_t_i[0, 0] = 0
        self.R_i_t[0, 0] = 0

        # return self.R_i_t, self.R_t_i #baka
        # return self.R_t_t, self.R_t_i #baka

        return self.R_t_t, self.R_i_i #baka
    
    
    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam (self, text_tokens, image_tokens, device):
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
        text_layer = self.model.cross_modal_text_layers[-1]
        image_layer = self.model.cross_modal_image_layers[-1]

        cam_t_i = text_layer.crossattention.self.get_attention_map().detach()
        grad_t_i = text_layer.crossattention.self.get_attn_gradients().detach()
        cam_t_i = self.gradcam(cam_t_i, grad_t_i)
        self.R_t_i = cam_t_i

        cam = text_layer.attention.self.get_attention_map().detach()
        grad = text_layer.attention.self.get_attn_gradients().detach()
        self.R_t_t = self.gradcam(cam, grad)

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i




    def generate_transformer_attr (self, text_tokens, image_tokens, device):

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        # text self attention
        blocks = self.model.text_transformer.encoder.layer
        for blk in blocks:
            cam = blk.attention.self.get_attention_map().detach()
            grad = blk.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

        for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            cam = text_layer.attention.self.get_attention_map().detach()
            grad = text_layer.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_t_t += torch.matmul(cam, self.R_t_t)

            cam = image_layer.attention.self.get_attention_map().detach()
            grad = image_layer.attention.self.get_attn_gradients().detach()
            cam = avg_heads(cam, grad)
            self.R_i_i += torch.matmul(cam, self.R_i_i)
        
        cam_t_i = text_layer.crossattention.self.get_attention_map().detach()
        # print(f"cam_t_i shape: {cam_t_i.shape}")
        grad_t_i = text_layer.crossattention.self.get_attn_gradients().detach()
        cam_t_i = avg_heads(cam_t_i, grad_t_i)
        self.R_t_i = cam_t_i

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_i_i
    

    def generate_rollout (self, text_tokens, image_tokens, device):

        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

        cams_text = []
        cams_image = []

        for blk in self.model.text_transformer.encoder.layer:
            cam = blk.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

        for text_layer in self.model.cross_modal_text_layers:
            cam = text_layer.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_text.append(cam)

        for image_layer in self.model.cross_modal_image_layers:
            cam = image_layer.attention.self.get_attention_map().detach()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
            cams_image.append(cam)    

        # for text_layer, image_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):    
        self.R_t_t = compute_rollout_attention(copy.deepcopy(cams_text))
        self.R_i_i = compute_rollout_attention(cams_image)
        cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attention_map().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = torch.matmul(self.R_t_t.t(), torch.matmul(cam_t_i, self.R_i_i))


        self.R_t_t[0, 0] = 0
        # self.R_t_i[0, 0] = 0
        return self.R_t_t, self.R_i_i

    # def generate_lrp(self, text_tokens, image_tokens, device):
    #     self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
    #     # impact of images on text
    #     self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
    #     # impact of text on images
    #     self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)

    #     cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attn_cam().detach()
    #     cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
    #     self.R_t_i = cam_t_i

    #     cam = self.model.cross_modal_text_layers[-1].attention.self.get_attn_cam().detach()
    #     cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
    #     self.R_t_t = cam

    #     self.R_t_t = (self.R_t_t - self.R_t_t.min()) / (self.R_t_t.max() - self.R_t_t.min())
    #     self.R_t_i = (self.R_t_i - self.R_t_i.min()) / (self.R_t_i.max() - self.R_t_i.min())
    #     # disregard the [CLS] token itself
    #     self.R_t_t[0, 0] = 0
    #     return self.R_t_t, self.R_t_i

    def generate_raw_attn (self, text_tokens, image_tokens, device):
        self.R_t_t = torch.eye(text_tokens, text_tokens).to(device)
        # image self attention matrix
        self.R_i_i = torch.eye(image_tokens, image_tokens).to(device)
        # impact of images on text
        self.R_t_i = torch.zeros(text_tokens, image_tokens).to(device)
        # impact of text on images
        self.R_i_t = torch.zeros(image_tokens, text_tokens).to(device)      

        # cam_t_i = self.model.cross_modal_text_layers[-1].crossattention.self.get_attention_map().detach()
        # cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        # self.R_t_i = cam_t_i

        cam_t_i = self.model.cross_modal_image_layers[-1].attention.self.get_attention_map().detach()
        cam_t_i = cam_t_i.reshape(-1, cam_t_i.shape[-2], cam_t_i.shape[-1]).mean(dim=0)
        self.R_t_i = cam_t_i

        cam = self.model.cross_modal_text_layers[-1].attention.self.get_attention_map().detach()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(dim=0)
        self.R_t_t = cam  

        self.R_t_t[0, 0] = 0
        return self.R_t_t, self.R_t_i

    def generate_decompx(self, text_emb, image_emb, target_class=None, device=None):
        # Standalone DecompX baseline: Propagate token decompositions and return per-token attributions
        # text_emb/image_emb: Initial embeddings (seq_text, hidden), (seq_img, hidden)
        # Returns text_rel, image_rel: (seq_text), (seq_img) - attribution scores (positive/negative for sentiment-like tasks)
        if device is None:
            device = text_emb.device
        text_n = text_emb.shape[0]
        image_n = image_emb.shape[0]
        total_input_n = text_n + image_n
        text_decomp = torch.zeros(text_n, total_input_n, text_emb.shape[1], device=device)
        for i in range(text_n):
            text_decomp[i, i, :] = text_emb[i]
        image_decomp = torch.zeros(image_n, total_input_n, image_emb.shape[1], device=device)
        for i in range(image_n):
            image_decomp[i, text_n + i, :] = image_emb[i]

        # Propagate through text self-attn layers
        for layer in self.model.text_transformer.encoder.layer:
            z_attn = self.decompose_attn(text_decomp, layer.attention)
            z_res = z_attn + text_decomp
            z_ln1 = self.decompose_ln(z_res, layer.attention.output.LayerNorm)
            z_ffn = self.decompose_ffn(z_ln1, layer.intermediate, layer.output)
            z_res2 = z_ffn + z_ln1
            text_decomp = self.decompose_ln(z_res2, layer.output.LayerNorm)

        # Cross-modal layers
        for t_layer, i_layer in zip(self.model.cross_modal_text_layers, self.model.cross_modal_image_layers):
            # Image self-attn + FFN + LN
            i_z_attn = self.decompose_attn(image_decomp, i_layer.attention)
            i_z_res = i_z_attn + image_decomp
            i_z_ln1 = self.decompose_ln(i_z_res, i_layer.attention.output.LayerNorm)
            i_z_ffn = self.decompose_ffn(i_z_ln1, i_layer.intermediate, i_layer.output)
            i_z_res2 = i_z_ffn + i_z_ln1
            image_decomp = self.decompose_ln(i_z_res2, i_layer.output.LayerNorm)

            # Text cross-attn (query=text, key=image) + FFN + LN
            t_z_cross = self.decompose_attn(text_decomp, t_layer.crossattention, is_cross=True, key_decomp=image_decomp)
            t_z_res = t_z_cross + text_decomp
            t_z_ln1 = self.decompose_ln(t_z_res, t_layer.crossattention.output.LayerNorm)
            t_z_ffn = self.decompose_ffn(t_z_ln1, t_layer.intermediate, t_layer.output)
            t_z_res2 = t_z_ffn + t_z_ln1
            text_decomp = self.decompose_ln(t_z_res2, t_layer.output.LayerNorm)

        # Classification head (assume logit_fc[0] linear, logit_fc[1] tanh, logit_fc[2] linear)
        cls_decomp = text_decomp[0]  # Contributions to [CLS]
        pool_agg = torch.tanh(self.model.logit_fc[0](cls_decomp.sum(0)))
        theta = pool_agg / (self.model.logit_fc[0](cls_decomp.sum(0)) + 1e-12)
        pool_no_bias = theta * self.model.logit_fc[0](cls_decomp)
        omega = torch.abs(torch.einsum('ih,h->i', pool_no_bias, self.model.logit_fc[0].bias)) / (torch.abs(torch.einsum('ih,h->i', pool_no_bias, self.model.logit_fc[0].bias)).sum() + 1e-12)
        pool_decomp = pool_no_bias + omega.unsqueeze(1) * self.model.logit_fc[0].bias
        logit_no_bias = self.model.logit_fc[2](pool_decomp)
        omega = torch.abs(torch.einsum('ic,c->i', logit_no_bias, self.model.logit_fc[2].bias)) / (torch.abs(torch.einsum('ic,c->i', logit_no_bias, self.model.logit_fc[2].bias)).sum() + 1e-12)
        logit_decomp = logit_no_bias + omega.unsqueeze(1) * self.model.logit_fc[2].bias  # (total_input_n, num_classes)

        # Extract attributions (sum over hidden, for target_class if provided)
        if target_class is not None:
            attributions = logit_decomp[:, target_class]
        else:
            attributions = logit_decomp.sum(1)  # Or max for multi-class

        text_rel = attributions[:text_n]
        image_rel = attributions[text_n:]

        text_rel[0] = 0  # Disregard [CLS] self-attribution, as in other methods
        return text_rel, image_rel


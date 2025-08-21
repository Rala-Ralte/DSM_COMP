import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertModel, BertEncoder, BertLayer
from .bert_model import BertCrossLayer, BertAttention
from . import swin_transformer as swin
from . import heads, objectives, meter_utils
from .clip_model import build_model, adapt_position_encoding
from .swin_helpers import swin_adapt_position_encoding
from transformers import RobertaConfig
from .roberta_model import RobertaModel
from .layers import *

class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.is_clip = (not 'swin' in config['vit'])

        if 'roberta' in config['tokenizer']:
            bert_config = RobertaConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )
        else:
            bert_config = BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_layers"],
                num_attention_heads=config["num_heads"],
                intermediate_size=config["hidden_size"] * config["mlp_ratio"],
                max_position_embeddings=config["max_text_len"],
                hidden_dropout_prob=config["drop_rate"],
                attention_probs_dropout_prob=config["drop_rate"],
            )

        resolution_after = config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                if self.is_clip:
                    build_model(config['vit'], resolution_after=resolution_after)
                else:
                    getattr(swin, self.hparams.config["vit"])(
                        pretrained=True, config=self.hparams.config,
                    )

                if 'roberta' in config['tokenizer']:
                    RobertaModel.from_pretrained(config['tokenizer'])
                else:
                    BertModel.from_pretrained(config['tokenizer'])

        if self.is_clip:
            self.visual = build_model(config['vit'], resolution_after=resolution_after)
            adapt_position_encoding(self.visual)
        else:
            self.visual = getattr(swin, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config,
            )
            swin_adapt_position_encoding(self.visual)

        self.text_encoder = BertModel(bert_config)
        self.text_encoder.bert.embeddings = BertEmbeddings(bert_config)
        self.text_encoder.bert.encoder = BertEncoder(bert_config)

        self.cross_modal_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(config["num_layers"])]
        )

        self.cross_modal_text_pooler = heads.Pooler(config["hidden_size"])
        self.cross_modal_image_pooler = heads.Pooler(config["hidden_size"] * 2 if not self.is_clip else config["hidden_size"])

        if self.hparams.config["loss_names"]["vqa"] > 0:
            self.vqa_classifier = heads.VQAMultiHead(config["hidden_size"] * 2, self.hparams.config["num_labels"])
        if self.hparams.config["loss_names"]["itm"] > 0:
            self.itm_classifier = heads.ITMHead(config["hidden_size"] * 2)
        if self.hparams.config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.current_tasks = []

    def embed_text(self, text):
        """Embed text input into hidden representations.
        Args:
            text (str or list): Input question or text.
        Returns:
            torch.Tensor: Embedded text of shape (seq_len, hidden_size).
        """
        tokenizer = get_pretrained_tokenizer(self.hparams.config["tokenizer"])
        encoded = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=self.hparams.config["max_text_len"])
        text_ids = encoded["input_ids"].to(self.device)
        text_masks = encoded["attention_mask"].to(self.device)
        
        # Use the text embedding layer directly
        embeddings = self.cross_modal_text_transform(self.token_type_embeddings(torch.zeros_like(text_ids)).to(self.device))
        return embeddings

    def embed_image(self, feats, boxes):
        """Embed image features into hidden representations.
        Args:
            feats (torch.Tensor): Image features of shape (num_boxes, feature_dim).
            boxes (torch.Tensor): Bounding boxes of shape (num_boxes, 4).
        Returns:
            torch.Tensor: Embedded image features of shape (num_boxes, hidden_size).
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # Add batch dimension if missing
        embeddings = self.cross_modal_image_transform(feats.to(self.device))
        return embeddings

    def perturbation_text(self, item, cam_image, cam_text, is_positive_pert):
        """Perturb text input based on relevance scores.
        Args:
            item (dict): Dataset item with 'sent' key.
            cam_image (torch.Tensor): Image relevance scores.
            cam_text (torch.Tensor): Text relevance scores.
            is_positive_pert (bool): Whether to perturb positively (high relevance) or negatively.
        Returns:
            list: Perturbation accuracy results.
        """
        text = item['sent']
        tokenizer = get_pretrained_tokenizer(self.hparams.config["tokenizer"])
        encoded = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=self.hparams.config["max_text_len"])
        text_ids = encoded["input_ids"].to(self.device)
        text_masks = encoded["attention_mask"].to(self.device)

        cam_text = cam_text / cam_text.max() if cam_text.max() > 0 else cam_text
        threshold = torch.quantile(cam_text, 0.9 if is_positive_pert else 0.1)

        mask = (cam_text > threshold) if is_positive_pert else (cam_text < threshold)
        perturbed_ids = text_ids.clone()
        perturbed_ids[~mask] = self.tokenizer.pad_token_id

        batch = {"text_ids": perturbed_ids, "text_masks": text_masks}
        ret = self.infer(batch)
        vqa_logits = self.vqa_classifier(ret["cls_feats"])
        pred = vqa_logits.argmax().item()
        true_answer = self.vqa_answers[str(pred)]
        return [1.0 if true_answer == item.get('answer', '') else 0.0]

    def perturbation_image(self, item, cam_image, cam_text, is_positive_pert):
        """Perturb image input based on relevance scores.
        Args:
            item (dict): Dataset item with 'img_id' key.
            cam_image (torch.Tensor): Image relevance scores.
            cam_text (torch.Tensor): Text relevance scores.
            is_positive_pert (bool): Whether to perturb positively (high relevance) or negatively.
        Returns:
            list: Perturbation accuracy results.
        """
        from PIL import Image
        img = Image.open(item['img_id']).convert("RGB")
        img = clip_transform(size=self.hparams.config["image_size"])(img).unsqueeze(0).to(self.device)

        cam_image = cam_image / cam_image.max() if cam_image.max() > 0 else cam_image
        threshold = torch.quantile(cam_image, 0.9 if is_positive_pert else 0.1)

        mask = (cam_image > threshold) if is_positive_pert else (cam_image < threshold)
        perturbed_img = img.clone()
        perturbed_img[:, :, :, ~mask] = 0

        batch = {"image": [perturbed_img]}
        ret = self.infer(batch)
        vqa_logits = self.vqa_classifier(ret["cls_feats"])
        pred = vqa_logits.argmax().item()
        true_answer = self.vqa_answers[str(pred)]
        return [1.0 if true_answer == item.get('answer', '') else 0.0]

    def infer(self, batch):
        text_ids = batch["text_ids"]
        text_labels = batch["text_labels"]
        text_masks = batch["text_masks"]
        image = batch["image"]

        extend_text_masks = ~text_masks
        extend_image_masks = torch.ones(image.size()[:2], dtype=torch.bool, device=image.device)

        x = self.text_encoder(text_ids, attention_mask=text_masks)[0]
        y = self.visual(image)

        all_text_feats = []
        all_image_feats = []
        for layer in self.cross_modal_layers:
            x1 = layer(x, extend_text_masks)
            y1 = layer(y, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]
            all_text_feats.append(x.detach().clone())
            all_image_feats.append(y.detach().clone())

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(x)
        if self.is_clip:
            cls_feats_image = self.cross_modal_image_pooler(y)
        else:
            avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
            cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "all_text_feats": all_text_feats,
            "all_image_feats": all_image_feats
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        meter_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        meter_utils.set_task(self)
        output = self(batch)
        ret = dict()
        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        return ret

    def on_test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]
        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        meter_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return meter_utils.set_schedule(self)

from sacred import Experiment

ex = Experiment("METER")

def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "meter"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "vqa": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 576
    patch_size = 16
    draw_false_image = 1
    image_only = False
    resolution_before = 576

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = 'roberta-base'
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/16'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Explanation Method Settings
    method_name = "spectral_decompx"  # Options: dsm, dsm_grad, dsm_grad_cam, spectral_decompx, attn_gradcam, rollout, raw_attn, decompx
    decompx_variant = "grad"  # Options: pure, grad, grad_cam (for spectral_decompx)
    target_class = None  # Integer or None for predicted class (for spectral_decompx and decompx)

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-5
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 100000
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5
    lr_mult_cross_modal = 5

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = True

    # Environment Settings
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0
    num_gpus = 1
    num_nodes = 1
    load_path = "meter_clip16_288_roberta_vqa.ckpt"
    num_workers = 8
    precision = 32

    # Perturbation-specific settings (for compatibility with perturbation.py)
    COCO_val_path = ""  # Path to COCO validation images
    modality = "image"  # Options: text, image
    is_positive_pert = True  # Whether to perturb positively

# Named configs for "env" which are orthogonal to "task"
@ex.named_config
def env_vqa():
    batch_size = 64
    max_epoch = 5
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 2e-6
    lr_mult_head = 10
    lr_mult_cross_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 50
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 384
    COCO_val_path = "path/to/coco/val2014"  # Example path

# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def swin32_base224():
    vit = "swin_base_patch4_window7_224_in22k"
    patch_size = 32
    image_size = 224
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 224

@ex.named_config
def swin32_base384():
    vit = "swin_base_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1024
    resolution_before = 384

@ex.named_config
def swin32_large384():
    vit = "swin_large_patch4_window12_384_in22k"
    patch_size = 32
    image_size = 384
    train_transform_keys = ["imagenet"]
    val_transform_keys = ["imagenet"]
    input_image_embed_size = 1536
    resolution_before = 384

@ex.named_config
def clip32():
    vit = 'ViT-B/32'
    image_size = 224
    patch_size = 32
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def clip16():
    vit = 'ViT-B/16'
    image_size = 576
    patch_size = 16
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768

@ex.named_config
def text_roberta():
    tokenizer = "roberta-base"
    vocab_size = 50265
    input_text_embed_size = 768

@ex.named_config
def text_roberta_large():
    tokenizer = "roberta-large"
    vocab_size = 50265
    input_text_embed_size = 1024

@ex.named_config
def imagenet_randaug():
    train_transform_keys = ["imagenet_randaug"]

@ex.named_config
def clip_randaug():
    train_transform_keys = ["clip_randaug"]

# Named configs for explanation methods
@ex.named_config
def expl_spectral_decompx():
    method_name = "spectral_decompx"
    decompx_variant = "grad"
    target_class = None

@ex.named_config
def expl_decompx():
    method_name = "decompx"
    target_class = None

@ex.named_config
def expl_dsm():
    method_name = "dsm"

@ex.named_config
def expl_dsm_grad():
    method_name = "dsm_grad"

@ex.named_config
def expl_dsm_grad_cam():
    method_name = "dsm_grad_cam"

@ex.named_config
def expl_attn_gradcam():
    method_name = "attn_gradcam"

@ex.named_config
def expl_rollout():
    method_name = "rollout"

@ex.named_config
def expl_raw_attn():
    method_name = "raw_attn"

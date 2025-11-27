# %%
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Tuple, Dict
import warnings
import pandas as pd
import pickle
import torch
import scanpy as sc
import seaborn as sns
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scanpy.get import _get_obs_rep

sys.path.insert(0, "../")
import ukbfound as scg
from ukbfound.model import TransformerModel
from ukbfound.tokenizer import tokenize_and_pad_batch, random_mask_value
from ukbfound.model.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)
from ukbfound.tokenizer.trait_tokenizer import ValueVocab
from ukbfound.model.preprocess import Preprocessor
from ukbfound.utils import set_seed

import torch.distributed as dist




sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off" 
warnings.filterwarnings('ignore')
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="UKB",
    do_train=True,
    load_model = None,
    mask_ratio=0.15,
    epochs=50,
    n_bins=2,
    MVC=False, # Masked value prediction for individual embedding
    ecs_thres=0.0, # Elastic individual similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=1e-4,
    batch_size=40,
    layer_size=256,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_trait = True,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="ukbGPT",
    reinit=True,
    mode="offline",
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)

set_seed(config.seed)
# settings for input and preprocessing
pad_token = "<pad>"
mask_token = "<mask>"
cls_token = "<cls>"
special_tokens = [pad_token, cls_token, mask_token, "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_trait = config.include_zero_trait  # if True, include zero traits among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "raw"  # "normed_raw", "log1p", or "binned", "raw"
output_style = "raw"  # "normed_raw", "log1p", or "binned", "raw"

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
ECS = config.ecs_thres > 0  # Elastic individual similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "combine"  # "category" or "continuous" or "scaling" or "combine"
individual_emb_style = "avg-pool"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
early_stop = 5
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight



explicit_zero_prob = MLM and include_zero_trait  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-4  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True
# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned", "raw"]
assert output_style in ["normed_raw", "log1p", "binned", "raw"]
assert input_emb_style in ["category", "continuous", "scaling", "combine"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )
elif input_style == "raw":
    n_cat = n_bins

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding trait expr values
    n_input_bins = n_bins + 2
elif input_emb_style == "combine":
    mask_value = None
    pad_value = None
    n_input_bins = None
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
if dataset_name == "UKB":
    data_dir = Path("../data/UKB")
    #data_df = pd.read_csv(data_dir / "output_data.10000.csv", index_col=0)
    data_df = pd.read_csv(data_dir / "output_data.csv", index_col=0)
    #data_df = data_df.loc[:, data_df.columns.str.contains('d20002')]
    adata = sc.AnnData(data_df.iloc[:-100, :].values)
    adata_test = sc.AnnData(data_df.iloc[-100:, :].values)
    #adata = sc.AnnData(data_df.iloc[20000:, :].values)
    #adata_test = sc.AnnData(data_df.iloc[20000:22000, :].values)    

    
    
    #adata = sc.AnnData(df.values)
    #adata_test = sc.AnnData(df.values)
    
    adata.obs["individualtype"] = "ukb"
    adata_test.obs["individualtype"] = "ukb"
    adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
    adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
    adata.var.index = data_df.columns.values
    adata_test.var.index = data_df.columns.values
    data_is_raw = False
    filter_trait_by_counts = False
    adata_test_raw = adata_test.copy()
    adata_combine = adata.concatenate(adata_test, batch_key="str_batch")

    
                
# make the batch category column
batch_id_labels = adata_combine.obs["str_batch"].astype("category").cat.codes.values
adata_combine.obs["batch_id"] = batch_id_labels
individualtype_id_labels = adata_combine.obs["individualtype"].astype("category").cat.codes.values
individualtypes = adata_combine.obs["individualtype"].unique()
num_types = len(np.unique(individualtype_id_labels))
id2type = dict(enumerate(adata_combine.obs["individualtype"].astype("category").cat.categories))
adata_combine.obs["individualtype_id"] = individualtype_id_labels
adata_combine.var["trait_name"] = adata_combine.var.index.tolist()
traits = adata_combine.var["trait_name"].tolist()
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = ValueVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if trait in vocab else -1 for trait in adata.var["trait_name"]
    ]
    trait_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(trait_ids_in_vocab >= 0)}/{len(trait_ids_in_vocab)} traits "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
if config.load_model is None:
    if input_emb_style == "combine":
        # 读取traits_df
        traits_df = pd.read_csv(data_dir / 'ukb_traits.csv', encoding='latin1', quotechar='"')
        # 创建一个字典，其中 token 是键，token_id 是值
        token_dict = dict(zip(traits_df['token_id'], traits_df['token_id']))

        #  vacab need start with 0 index
        token_dict['<cls>'] = 0  # 和其他模式统一， 用0
        token_dict['<mask>'] = len(token_dict)
        token_dict['<pad>'] = len(token_dict)
        
        # 将字典保存成 JSON 文件
        with open(data_dir / "vocab.json", 'w') as f:
            json.dump(token_dict, f, indent=4)

        vocab_file = data_dir / "vocab.json"
        vocab = ValueVocab.from_file(vocab_file)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        vocab.set_default_index(vocab["<pad>"])
        # change adata value to combine 'col_value' style
        df_tmp = pd.DataFrame(adata_combine.X, columns=traits)

        adata_combine.X = df_tmp.fillna(vocab["<pad>"])

        trait_dict = {v:i for i, v in  enumerate(traits_df['trait'].unique(), 1)}
        
        value2trait_dict = {}
        for value_id in traits_df['token_id']:
            value2trait_dict[value_id] = trait_dict[traits_df.loc[traits_df['token_id'] == value_id, 'trait'].iat[0]]
        value2trait_dict.update({token_dict['<cls>']:0})
        value2trait_dict.update({token_dict['<mask>']:max(value2trait_dict.values())+1})
        value2trait_dict.update({token_dict['<pad>']:max(value2trait_dict.values())+1})

        # 创建一个向量化的函数，用于字典查找
        vectorized_replace = np.vectorize(value2trait_dict.get)
        


        #trait_dict = {v:i for i, v in  enumerate(traits_df['trait'].unique(), 1)}
        #trait_dict.update({0:0})
        #trait_cols = [i.split('_d')[0] for i in adata_combine.var['trait_name'].tolist()] 
        #value_ids = np.array([trait_dict[i] for i in trait_cols])
        #value_ids = np.array(range(df_tmp.shape[1]))
        #value_ids = adata_combine.var["trait_name"]
    else:
        vocab = Vocab(
            VocabPybind(traits + special_tokens, None)
        )  # bidirectional lookup [trait <-> int]
        vocab.set_default_index(vocab["<pad>"])
        trait_ids = np.array(vocab(traits), dtype=int)
if input_emb_style == 'combine':
    mask_value = vocab[mask_token]
    pad_value = vocab[pad_token]
    cls_value = vocab[cls_token]
    


# set up the preprocessor, use the args to config the workflow
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_trait_by_counts=filter_trait_by_counts,  # step 1
    filter_individual_by_counts=False,  # step 2
    normalize_total=None,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable traits
    hvg_flavor="seurat_v3" if data_is_raw else "individual_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    raw = True if input_style == 'raw' else False  # not trans
)


adata_test = adata_combine[adata_combine.obs["str_batch"] == "1"]
adata = adata_combine[adata_combine.obs["str_batch"] == "0"]
# 使用该函数对 all_counts 进行替换，并覆盖原数组
trait_ids = vectorized_replace(adata.X)

# 计算近邻图和UMAP嵌入
sc.pp.neighbors(adata_test, n_neighbors=15, use_rep='X')
sc.tl.umap(adata_test)

# 更新 adata_test_raw 的 UMAP 值
adata_test_raw = adata_test.copy()
adata_test_raw.obsm['X_umap'] = adata_test.obsm['X_umap']

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)

input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
    "raw": None
}[input_style]


all_counts = _get_obs_rep(adata, layer=input_layer_key)
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(all_counts)
    else all_counts
)

individualtypes_labels = adata.obs["individualtype_id"].tolist()  # make sure count from 0
individualtypes_labels = np.array(individualtypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_traits,
    valid_traits,
    train_individualtype_labels,
    valid_individualtype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, trait_ids, individualtypes_labels, batch_ids, test_size=0.1, shuffle=True
)

# Returns tokenized_train = {
#       "traits": torch.stack(trait_ids_list, dim=0),
#       "values": torch.stack(values_list, dim=0),
#   }
tokenized_train = tokenize_and_pad_batch(
    train_data,
    train_traits,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=False,  # append <cls> token at the beginning
    include_zero_value=include_zero_trait,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    valid_traits,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=False,
    include_zero_value=include_zero_trait,
)
logger.info(
    f"train set number of samples: {tokenized_train['traits'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['traits'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['traits'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['traits'].shape[1]}"
)

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
        
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_trait_ids_train, input_trait_ids_valid = (
        tokenized_train["traits"],
        tokenized_valid["traits"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_individualtype_labels_train = torch.from_numpy(train_individualtype_labels).long()
    tensor_individualtype_labels_valid = torch.from_numpy(valid_individualtype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_trait_ids_train = input_trait_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_individualtype_labels_train = tensor_individualtype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_trait_ids_valid = input_trait_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_individualtype_labels_valid = tensor_individualtype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "value_ids": input_trait_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "individualtype_labels": tensor_individualtype_labels_train,
    }
    valid_data_pt = {
        "value_ids": input_trait_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "individualtype_labels": tensor_individualtype_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["value_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分布式进程组
dist.init_process_group(backend='nccl')

# 获取本地rank
local_rank = int(os.environ["LOCAL_RANK"])

# 设置设备
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")


ntokens = len(set(value2trait_dict.values()))  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    individual_emb_style=individual_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    print("-"*20)
    print(f"name: {name}")
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
    # if config.freeze and "encoder" in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
wandb.log(
        {
            "info/pre_freeze_param_count": pre_freeze_param_count,
            "info/post_freeze_param_count": post_freeze_param_count,
        },
)

model.to(device)

# 使用DistributedDataParallel
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
'''
# 定义权重，偶数位置的权重设为较大值，奇数位置设为默认值（例如1）
weights = torch.ones_like(torch.LongTensor(range(1, len(vocab)+1)))
weights[torch.LongTensor(range(1, len(vocab)+1)) % 2 == 0] = 2.0  # 偶数位置的权重设为2.0，可以根据需要调整这个值
weights = weights.float().to(device)
# 定义损失函数
criterion_combine = nn.CrossEntropyLoss(weight=weights)
'''
criterion_combine = nn.CrossEntropyLoss()

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)
if DAB_separate_optim:
    optimizer_dab = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_dab = torch.optim.lr_scheduler.StepLR(
        optimizer_dab, schedule_interval, gamma=config.schedule_ratio
    )


scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
def define_wandb_metrcis():
    wandb.define_metric("valid/mlm", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mlm_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")
    
def run_epoch(model: nn.Module, loader: DataLoader, train=True, cls=False, cce=False, mvc=False, ecs=False, return_raw: bool = False) -> None:
    """
    Train or evaluate the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mlm,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()
    num_batches = len(loader)
    mlm_predictions = []
    mlm_targets = []
    cls_predictions = []
    cls_targets = []
    if train:
        model.train()
    else:
        model.eval()
    with torch.set_grad_enabled(train):  # 控制是否计算梯度
        for batch, batch_data in enumerate(loader):
            input_trait_ids = batch_data["value_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            individualtype_labels = batch_data["individualtype_labels"].to(device)
            src_key_padding_mask = input_trait_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_trait_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=cls,
                    CCE=cce,
                    MVC=mvc,
                    ECS=ecs,
                    do_sample=do_sample_in_train,
                    #generative_training=False
                )
                
                # bool array, True == 1, False == 0
                masked_positions = input_values.eq(mask_value)  # the postions to predict, it's value been masked not trait
                loss = 0.0
                metrics_to_log = {}
                if MLM:
                    if input_emb_style == 'combine':
                        # only calculate masked positions loss; mlm_output=(batch, seq, len(vocab) ); target_values=(batch, seq)
                        '''
                        print('output_dict["mlm_output"].shape', output_dict["mlm_output"].dtype)
                        print('output_dict["mlm_output"]', output_dict["mlm_output"])
                        print('target_values.shape', target_values.dtype)
                        print('target_values', target_values)
                        print('masked_positions.shape', masked_positions.dtype)
                        '''
                        
                        loss_mlm = criterion_combine(output_dict["mlm_output"][masked_positions], target_values[masked_positions].long()) 
                        # 只惩罚/关注 疾病部分20002
                        #disease_masked_positions = masked_positions & (1464 <= target_values) & (target_values <= 2353) 
                        #loss_mlm = criterion_combine(output_dict["mlm_output"][disease_masked_positions ], target_values[disease_masked_positions].long()) 
                        
                        # 只惩罚/关注 重要的回答
                        #loss_mlm = criterion_combine(output_dict["mlm_output"][masked_positions][target_values[masked_positions].long()%2==0], target_values[masked_positions].long()[target_values[masked_positions].long()%2==0])
                        loss = loss + loss_mlm
                        mlm_predictions.append(output_dict["mlm_output"].argmax(2).cpu().numpy()) # to 2dim, to cpu
                        mlm_targets.append(target_values.cpu().numpy())
                    else:
                        loss_mlm = criterion(
                            output_dict["mlm_output"], target_values, masked_positions
                        )
                        loss = loss + loss_mlm
                        metrics_to_log = {"train/mlm": loss_mlm.item()}
                        if explicit_zero_prob:
                            loss_zero_log_prob = criterion_neg_log_bernoulli(
                                output_dict["mlm_zero_probs"], target_values, masked_positions
                            )
                            loss = loss + loss_zero_log_prob
                            metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if ECS:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if DAB:
                    # try weighting and separate optimizer
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                    loss = loss + dab_weight * loss_dab
                    metrics_to_log.update({"train/dab": loss_dab.item()})
            if train:
                model.zero_grad()
                #print('loss=', loss)
                #print('type loss', type(loss))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings("always")
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                        error_if_nonfinite=False if scaler.is_enabled() else True,
                    )
                    if len(w) > 0:
                        logger.warning(
                            f"Found infinite gradient. This may be caused by the gradient "
                            f"scaler. The current scale is {scaler.get_scale()}. This warning "
                            "can be ignored if no longer occurs after autoscaling of the scaler."
                        )
                scaler.step(optimizer)
                scaler.update()

                wandb.log(metrics_to_log)
            
            # common part
            total_loss += loss.item()
            total_mlm += loss_mlm.item() if MLM else 0.0
            total_ecs += loss_ecs.item() if ECS else 0.0
            total_dab += loss_dab.item() if DAB else 0.0
            total_zero_log_prob += loss_zero_log_prob.item() if (explicit_zero_prob and (input_emb_style != 'combine')) else 0.0

            # train only for rerun model adv
            if train:
                if batch % log_interval == 0 and batch > 0:
                    metrics = {
                        "lr": scheduler.get_last_lr()[0],
                        "ms":(time.time() - start_time) * 1000 / log_interval,
                        "loss": total_loss / log_interval,
                        "mlm": total_mlm / log_interval if MLM else 0.0,
                        "ecs": total_ecs / log_interval if ECS else 0.0,
                        "dab": total_dab / log_interval if DAB else 0.0,
                        "zero_log_prob": total_zero_log_prob / log_interval if explicit_zero_prob else 0.0,
                        "error": total_error / log_interval,
                    }
                    
                    log_message = (
                        f"| epoch {epoch:3d} | {batch:3d}/{log_interval:3d} batches | "
                        f"lr {metrics['lr']:05.4f} | ms/batch {metrics['ms']:5.2f} | "
                        f"loss {metrics['loss']:5.2f} | "
                        + (f"mlm {metrics['mlm']:5.2f} | " if MLM else "")
                        + (f"ecs {metrics['ecs']:5.2f} | " if ECS else "")
                        + (f"dab {metrics['dab']:5.2f} | " if DAB else "")
                        + (f"nzlp {metrics['zero_log_prob']:5.2f} | " if explicit_zero_prob else "")
                    )
                    logger.info(log_message)
                    total_loss = 0
                    total_mlm = 0
                    total_cls = 0
                    total_cce = 0
                    total_mvc = 0
                    total_ecs = 0
                    total_dab = 0
                    total_adv_E = 0
                    total_adv_D = 0
                    total_zero_log_prob = 0
                    total_mvc_zero_log_prob = 0
                    total_error = 0
                    start_time = time.time()
        if not train: # evaluation
            metrics = {
                "loss": total_loss / num_batches,
                "mlm": total_mlm / num_batches if MLM else 0.0,
                "ecs": total_ecs / num_batches if ECS else 0.0,
                "dab": total_dab / num_batches if DAB else 0.0,
                "zero_log_prob": total_zero_log_prob / num_batches if explicit_zero_prob else 0.0,
                "error": total_error / num_batches,
            }
            
            log_message = (
                f"loss {metrics['loss']:5.2f} | "
                + (f"mlm {metrics['mlm']:5.2f} | " if MLM else "")
                + (f"ecs {metrics['ecs']:5.2f} | " if ECS else "")
                + (f"dab {metrics['dab']:5.2f} | " if DAB else "")
                + (f"nzlp {metrics['zero_log_prob']:5.2f} | " if explicit_zero_prob else "")
            )
            
            logger.info(log_message)

            if return_raw:
                raw_metrics = {'mlm_predictions': np.concatenate(mlm_predictions, axis=0).flatten(),
                               'mlm_targets': np.concatenate(mlm_targets, axis=0).flatten(),
                               }
                return  raw_metrics
            return metrics


best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    if config.do_train:
        run_epoch(
            model,
            loader=train_loader,
            train=True,
            return_raw= False
        )
        
    val_matrix = run_epoch(model, loader=train_loader,train=False,cls=False,cce=False,mvc=False,ecs=False,return_raw= False)
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mlm {val_matrix['mlm']:5.4f} |"
    )
    logger.info("-" * 89)
    print("ssssssssssssval_matrix['mlm']=", type(val_matrix['mlm']))
    
    if val_matrix['mlm'] < best_val_loss:
        best_val_loss = val_matrix['mlm']
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    torch.save(
        model.state_dict(),
        save_dir / f"model_{epoch}.pt",
    )

    scheduler.step()

dist.destroy_process_group()






# %% inference
def test(model: nn.Module, adata: DataLoader) -> float:
    adata_leyer = _get_obs_rep(adata, layer=input_layer_key)
    all_counts = (
        adata_leyer.A
        if issparse(adata_leyer)
        else adata_leyer
    )

    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        trait_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_value=include_zero_trait,
    )
    '''
    values_test = tokenized_test["values"].clone().detach()
    mask_test = (tokenized_test["traits"] == masked_trait)
    values_test[mask_test] = mask_value
    input_values_test = values_test 
    '''

    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )



    test_data_pt = {
        "value_ids": tokenized_test["traits"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "individualtype_labels": torch.from_numpy(individualtypes_labels).long(),
    }

    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )

    model.eval()
    val_raw_matrix = run_epoch(model, loader=train_loader,train=False,cls=False,cce=False,mvc=False,ecs=False,return_raw= True)
    target_np = val_raw_matrix['mlm_targets']
    pred_np = val_raw_matrix['mlm_predictions']
    print('11111111111', pred_np)
    print('22222222222221', target_np)
    # compute accuracy, precision, recall, f1
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(target_np, pred_np)
    precision = precision_score(target_np, pred_np, average="macro")
    recall = recall_score(target_np, pred_np, average="macro")
    macro_f1 = f1_score(target_np, pred_np, average="macro")

    logger.info(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
        f"Macro F1: {macro_f1:.3f}"
    )

    results = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/macro_f1": macro_f1,
    }

    return pred_np, target_np
# save the model into the save_dir
from collections import OrderedDict
torch.save(best_model.state_dict(), save_dir / "model.pt")
vocab_dict = OrderedDict(vocab.get_stoi())
with open(save_dir / 'vocab.json', 'w') as json_file:
    json.dump(vocab_dict, json_file, indent=4)

with open(save_dir / 'config.json', 'w') as json_file:
    json.dump(dict(config), json_file, indent=4)


# Retrieve the data-independent trait embeddings from ukbfound

trait2idx = vocab.get_stoi()
idx2trait = vocab.get_itos()
trait_ids = np.array([id for key, id in trait2idx.items() if not key.endswith('_0')])
trait_embeddings = model.value_encoder(torch.tensor(trait_ids, dtype=torch.long).to(device))
trait_embeddings = trait_embeddings.detach().cpu().numpy()
# Filter on the intersection between the Immune Human HVGs found in step 1.2 and ukbfound's 30+K foundation model vocab
trait_embeddings = {idx2trait[trait_id]: trait_embeddings[i] for i, trait_id in enumerate(trait_ids) }
print('Retrieved trait embeddings for {} traits.'.format(len(trait_embeddings)))
# Construct trait embedding network
from ukbfound.tasks import TraitEmbedding

embed = TraitEmbedding(trait_embeddings)

# Perform Louvain clustering with desired resolution; here we specify resolution=40
gdata = embed.get_adata(resolution=4)
# Retrieve the trait clusters
metatraits = embed.get_metatraits(gdata)
metatraits
# Obtain the set of trait programs from clusters with #traits >= 5
mgs = dict()
for mg, traits in metatraits.items():
    if len(traits) > 4:
        mgs[mg] = traits
# Here are the trait programs identified
len(mgs)

[(k, lst) for k, lst in mgs.items() if pd.Series(lst).str.contains('20116_[12]').any() and pd.Series(lst).str.contains('20002_1').any()]
#[(k, lst) for k, lst in mgs.items() if pd.Series(lst).str.contains('20116_[12]').any()]

import tqdm
# Retrieve trait program 2 which contains the SFTPC trait set
CD_traits = mgs['15']
print(CD_traits)
# Compute cosine similarities among traits in this trait program
df_CD = pd.DataFrame(columns=['Trait ', 'Similarity', 'Trait 1'])
for i in tqdm.tqdm(CD_traits):
    df = embed.compute_similarities(i, CD_traits)
    df['Trait 1'] = i
    df_CD = pd.concat([df_CD, df], ignore_index=True)
df_CD_sub = df_CD[df_CD['Similarity']<0.99].sort_values(by='Trait ') # Filter out edges from each trait to itself

df_CD_sub.sort_values(by='Similarity', ascending=False)

import networkx as nx

# Creates a graph from the cosine similarity network
input_node_weights = [(row['Trait '], row['Trait 1'], round(row['Similarity'], 2)) for i, row in df_CD_sub.iterrows()]
G = nx.Graph()
G.add_weighted_edges_from(input_node_weights)

# Plot the cosine similarity network; strong edges (> select threshold) are highlighted
thresh = 0.35
plt.figure(figsize=(20, 20))
widths = nx.get_edge_attributes(G, 'weight')

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > thresh]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= thresh]

pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)

width_large = {}
width_small = {}
for i, v in enumerate(list(widths.values())):
    if v > thresh:
        width_large[list(widths.keys())[i]] = v*10
    else:
        width_small[list(widths.keys())[i]] = max(v, 0)*10

nx.draw_networkx_edges(G, pos,
                       edgelist = width_small.keys(),
                       width=list(width_small.values()),
                       edge_color='lightblue',
                       alpha=0.8)
nx.draw_networkx_edges(G, pos, 
                       edgelist = width_large.keys(), 
                       width = list(width_large.values()), 
                       alpha = 0.5, 
                       edge_color = "blue", 
                      )
# node labels
nx.draw_networkx_labels(G, pos, font_size=25, font_family="sans-serif")
# edge weight labels
d = nx.get_edge_attributes(G, "weight")
edge_labels = {k: d[k] for k in elarge}
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")

# Save the figure before displaying it
plt.savefig(data_dir / "GRN_results.png", dpi=300)

# Display the plot
plt.show()

# Close the plot window
plt.close()

sns.set(font_scale=0.35)
embed.score_metatraits(adata, metatraits)
embed.plot_metatraits_scores(adata, mgs, "individualtype")

pred_np, target_np = test(best_model, adata_test)

#pred_np, target_np = test(best_model, adata_test)

import numpy as np
import matplotlib.pyplot as plt

# 例如：
#pred_np = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
#target_np = np.array([3, 0.5, 2, 2, 3, 2.5, 2, 1, 1.5, 1])

# 计算相关系数
target_np = target_np.astype(np.float32)
pred_np = pred_np.astype(np.float32)
correlation_matrix = np.corrcoef(target_np, pred_np)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

print(f"Correlation (R): {correlation_xy}")
print(f"R-squared: {r_squared}")

# 绘制散点图
plt.scatter(target_np, pred_np, color='blue', label='Data Points')

# 添加最佳拟合线
a, b = np.polyfit(target_np, pred_np, 1)
plt.plot(target_np, a*target_np + b, color='red', label=f'Fit Line: y={a:.2f}x+{b:.2f}')

# 添加标题和标签
plt.title('LLM Scatter Plot with Best Fit Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# 显示图形
plt.show()


predictions, labels, results = test(best_model, adata_test)
adata_test_raw.obs["predictions"] = [id2type[p] for p in predictions]

# plot
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] 
palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"] + plt.rcParams["axes.prop_cycle"].by_key()["color"]
palette_ = {c: palette_[i] for i, c in enumerate(individualtypes)}

with plt.rc_context({"figure.figsize": (6, 4), "figure.dpi": (300)}):
    sc.pl.umap(
        adata_test_raw,
        color=["individualtype", "predictions"],
        palette=palette_,
        show=False,
    )
    plt.savefig(save_dir / "results.png", dpi=300)

save_dict = {
    "predictions": predictions,
    "labels": labels,
    "results": results,
    "id_maps": id2type
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

results["test/individual_umap"] = wandb.Image(
    str(save_dir / "results.png"),
    caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
)
wandb.log(results)



import xgboost as xgb 
from sklearn.metrics import mean_squared_error

# 初始化一个XGBoost回归器
model = xgb.XGBRegressor()

# 使用训练数据进行拟合
model.fit(adata.X[:, :-1], adata.X[:, -1])

# 使用模型进行预测
predictions = model.predict(adata_test.X[:, :-1])

# 计算均方根误差(Rmlm)
rmse = np.sqrt(mean_squared_error(adata_test.X[:, -1], predictions))
print("RMSE:", rmse)


import numpy as np
import matplotlib.pyplot as plt

# 假设 pred_np 和 target_np 是你的预测值和实际值的numpy数组
# 例如：
#pred_np = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
#target_np = np.array([3, 0.5, 2, 2, 3, 2.5, 2, 1, 1.5, 1])

# 计算相关系数
target_np = adata_test.X[:, -1]
pred_np = predictions
correlation_matrix = np.corrcoef(target_np, pred_np)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

print(f"Correlation (R): {correlation_xy}")
print(f"R-squared: {r_squared}")

# 绘制散点图
plt.scatter(target_np, pred_np, color='blue', label='Data Points')

# 添加最佳拟合线
a, b = np.polyfit(target_np, pred_np, 1)
plt.plot(target_np, a*target_np + b, color='red', label=f'Fit Line: y={a:.2f}x+{b:.2f}')

# 添加标题和标签
plt.title('XGboost Scatter Plot with Best Fit Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# 显示图形
plt.show()



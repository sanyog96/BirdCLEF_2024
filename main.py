# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py\
# !python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
# !pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl

import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import copy
import time
import os
import timm
import random
import numpy as np
import gc
import torch
import torchaudio
import torchvision
from sklearn.model_selection import StratifiedKFold
# from warmup_scheduler import GradualWarmupScheduler
from torch.optim import AdamW
import albumentations as A
import matplotlib.pyplot as plt
from pylab import rcParams

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas.api.types

from typing import Union


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    ''' Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    '''
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    '''
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    '''
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """ Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f'All target values in {df_name} must be numeric')

    if df.min().min() < 0:
        raise ParticipantVisibleError(f'All target values in {df_name} must be at least zero')

    if df.max().max() > 1:
        raise ParticipantVisibleError(f'All target values in {df_name} must be no greater than one')

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f'Target values in {df_name} do not add to one within all rows')
    
import pandas as pd
import pandas.api.types
import sklearn.metrics


class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Version of macro-averaged ROC-AUC score that ignores all classes that have no true positive labels.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return safe_call_score(sklearn.metrics.roc_auc_score, solution[scored_columns].values, submission[scored_columns].values, average='macro')

def calculate_competition_metrics(gt, preds, target_columns, one_hot=True):
   if not one_hot:
      ground_truth = np.argmax(gt, axis=1)
      gt = np.zeros((ground_truth.size, len(target_columns)))
      gt[np.arange(ground_truth.size), ground_truth] = 1
   val_df = pd.DataFrame(gt, columns=target_columns)
   pred_df = pd.DataFrame(preds, columns=target_columns)
   cmAP_1 = padded_cmap(val_df, pred_df, padding_factor=1)
   cmAP_5 = padded_cmap(val_df, pred_df, padding_factor=5)
   mAP = map_score(val_df, pred_df)
   val_df['id'] = [f'id_{i}' for i in range(len(val_df))]
   pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))]
   train_score = score(val_df, pred_df, row_id_column_name='id')
   return {
      "cmAP_1": cmAP_1,
      "cmAP_5": cmAP_5,
      "mAP": mAP,
      "ROC": train_score,
           }
def metrics_to_string(scores, key_word):
  log_info = ""
  for key in scores.keys():
      log_info = log_info + f"{key_word} {key} : {scores[key]:.4f}, "
  return log_info
  
  
def calculate_competition_metrics_no_map(gt, preds, target_columns):
   
   ground_truth = np.argmax(gt, axis=1)
   gt = np.zeros((ground_truth.size, len(target_columns)))
   gt[np.arange(ground_truth.size), ground_truth] = 1
   val_df = pd.DataFrame(gt, columns=target_columns)
   pred_df = pd.DataFrame(preds, columns=target_columns)
  
   val_df['id'] = [f'id_{i}' for i in range(len(val_df))]
   pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))]
   train_score = score(val_df, pred_df, row_id_column_name='id')
   return {
      
      "ROC": train_score
           }


exp_name = 'Convnext_tiny_channel1_KAN'
backbone = 'eca_nfnet_l0'
seed = 42
batch_size = 64
num_workers = 0

n_epochs = 100
warmup_epo = 5
cosine_epo = n_epochs - warmup_epo

image_size = 256

lr_max = 1e-5
lr_min = 1e-7
weight_decay = 1e-6

mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
    "center" : True,
    "pad_mode" : "constant",
    "norm" : "slaney",
    "onesided" : True,
    "mel_scale" : "slaney"
}

top_db = 80
train_period = 5
val_period = 5

secondary_coef = 1.0

train_duration = train_period * mel_spec_params["sample_rate"]
val_duration = val_period * mel_spec_params["sample_rate"]

N_FOLD = 5
fold = 2

use_amp = True
max_grad_norm = 10
early_stopping = 7


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = xm.xla_device()

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, exp_name), exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(seed)

###### DATASET

df = pd.read_csv("birdclef-2024/train_metadata.csv")
df["path"] = "birdclef-2024/train_audio" + df["filename"]
df["rating"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

skf = StratifiedKFold(n_splits=N_FOLD, random_state=seed, shuffle=True)
df['fold'] = -1
for ifold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["primary_label"].values)):
    df.loc[val_idx, 'fold'] = ifold

sub = pd.read_csv("birdclef-2024/sample_submission.csv")
target_columns = sub.columns.tolist()[1:]
num_classes = len(target_columns)
bird2id = {b: i for i, b in enumerate(target_columns)}

print(num_classes)


def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


def read_wav(path):
    wav, org_sr = torchaudio.load(path, normalize=True)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=mel_spec_params["sample_rate"])
    return wav


def crop_start_wav(wav, duration_):
    while wav.size(-1) < duration_:
        wav = torch.cat([wav, wav], dim=1)
    wav = wav[:, :duration_]
    return wav


class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, add_secondary_labels=True):
        self.df = df
        self.bird2id = bird2id
        self.num_classes = num_classes
        self.secondary_coef = secondary_coef
        self.add_secondary_labels = add_secondary_labels
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def prepare_target(self, primary_label, secondary_labels):
        secondary_labels = eval(secondary_labels)
        target = np.zeros(self.num_classes, dtype=np.float32)
        if primary_label != 'nocall':
            primary_label = self.bird2id[primary_label]
            target[primary_label] = 1.0
            if self.add_secondary_labels:
                for s in secondary_labels:
                    if s != "" and s in self.bird2id.keys():
                        target[self.bird2id[s]] = self.secondary_coef
        target = torch.from_numpy(target).float()
        return target

    def prepare_spec(self, path):
        if("train_audio" in path):
            path_splits = path.split("train_audio")
            path = path_splits[0] + "train_audio/" + path_splits[1]
        wav = read_wav(path)
        wav = crop_start_wav(wav, train_duration)
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        mel_spectrogram = mel_spectrogram * 255
        # mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        mel_spectrogram = mel_spectrogram.permute(1, 2, 0).numpy()
        # print("=========================================================")
        # print(mel_spectrogram.shape)
        # print("=========================================================")
        return mel_spectrogram

    def __getitem__(self, idx):
        path = self.df["path"].iloc[idx]
        primary_label = self.df["primary_label"].iloc[idx]
        secondary_labels = self.df["secondary_labels"].iloc[idx]
        rating = self.df["rating"].iloc[idx]

        spec = self.prepare_spec(path)
        target = self.prepare_target(primary_label, secondary_labels)

        if self.transform is not None:
            # mean_tuple = (0)*x['image'].shape[2]
            # std_tuple = (0.8)*x['image'].shape[2]
            res = self.transform(image=spec)
            spec = res['image'].astype(np.float32)
        else:
            spec = spec.astype(np.float32)

        spec = spec.transpose(2, 0, 1)

        return {"spec": spec, "target": target, 'rating': rating}


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p)
        x = x.view(bs, ch)
        return x


class CNN(torch.nn.Module):
    def __init__(self, backbone, pretrained):
        super().__init__()

        out_indices = (3, 4)
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, num_classes)

    def forward(self, x, x_shapes):
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.head(x)
        return x, []

class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss

criterion = FocalLossBCE()

def init_logger(log_file='train.log'):
    from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

import itertools
import math
import warnings
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ACTIVATIONS = ("relu", "gelu")
_DATA_FORMATS = ("channels_last", "channels_first")

def _trunc_normal_(tensor: Tensor, mean, std, a, b) -> Tensor:
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x) -> float:
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    low = norm_cdf((a - mean) / std)
    upp = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * low - 1, 2 * upp - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def drop_path(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def get_activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Invalid argument {name=}. (expected one of {ACTIVATIONS})")


def remove_index_nd(x: Tensor, index: int, dim: int = -1) -> Tensor:
    """Remove values at specified index and dim.

    Args:
        x: Tensor of shape (..., D, ...)
        index: Index of the value to be removed.
        dim: Dimension to modified.

    Returns:
        Tensor of shape (..., D-1, ...)
    """
    size = x.shape[dim]
    mask = torch.full((size,), True)
    mask[index] = False
    indices = torch.where(mask)[0]

    slices: list[Any] = [slice(None)] * x.ndim
    slices[dim] = indices
    x = x[slices]

    return x


@torch.inference_mode()
def hash_model(model: nn.Module) -> int:
    params_or_buffers = itertools.chain(model.parameters(), model.buffers())
    hash_value = sum(hash_tensor(p) * (i + 1) for i, p in enumerate(params_or_buffers))
    return hash_value


@torch.inference_mode()
def hash_tensor(x: Tensor) -> int:
    x = x.cpu()
    if x.ndim > 0:
        x = x.flatten()
        dtype = x.dtype if x.dtype != torch.bool else torch.int
        x = x * torch.arange(1, len(x) + 1, device=x.device, dtype=dtype)
        x = x.nansum()

    hash_value = int(x.item())
    return hash_value

class CustomLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        if data_format not in _DATA_FORMATS:
            raise ValueError(
                f"Invalid argument {data_format=}. (expected one of {_DATA_FORMATS})"
            )

        super().__init__()
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise ValueError(f"Invalid argument {self.data_format=}.")


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PositionalEncoding(nn.Module):
    # BASED ON PYTORCH TUTORIAL : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        maxlen: int = 5000,
    ) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("pos_embedding", pos_embedding)
        self.pos_embedding: Tensor

    def forward(self, token_embedding: Tensor) -> Tensor:
        pos_embedding_value = self.pos_embedding[: token_embedding.size(0), :]
        output = self.dropout(token_embedding + pos_embedding_value)
        return output

from typing import Iterable

# !pip install torchlibrosa
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

# from dcase24t6.augmentations.mixup import do_mixup
# from dcase24t6.augmentations.speed_perturb import SpeedPerturbation
# from dcase24t6.nn.functional import trunc_normal_
# from dcase24t6.nn.modules import CustomLayerNorm, DropPath

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
        groups: int = 1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.groups = groups

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(groups, in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(
            torch.empty(groups, out_features, in_features)
        )
        self.spline_weight = torch.nn.Parameter(
            torch.empty(groups, out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.empty(groups, out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(
                        self.groups,
                        self.grid_size + 1,
                        self.in_features,
                        self.out_features,
                    )
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.transpose(-1, -2)[
                        :, self.spline_order : -self.spline_order
                    ],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (groups, batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (groups, batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 3 and x.size(2) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (groups, in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        grid = grid.unsqueeze(1)
        # print(x.shape, grid.shape)
        bases = ((x >= grid[:, :, :, :-1]) & (x < grid[:, :, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :, :, : -(k + 1)])
                / (grid[:, :, :, k:-1] - grid[:, :, :, : -(k + 1)])
                * bases[:, :, :, :-1]
            ) + (
                (grid[:, :, :, k + 1 :] - x)
                / (grid[:, :, :, k + 1 :] - grid[:, :, :, 1:(-k)])
                * bases[:, :, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            x.size(1),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (groups, batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (groups, batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (groups, out_features, in_features, grid_size + spline_order).
        """
        # print(x.shape)
        assert x.dim() == 3 and x.size(2) == self.in_features
        assert y.size() == (x.size(0), x.size(1), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            1, 2
        )  # (groups, in_features, batch_size, grid_size + spline_order)
        B = y.transpose(1, 2)  # (groups, in_features, batch_size, out_features)
        solution: torch.Tensor = torch.linalg.lstsq(
            A, B
        ).solution  # (groups, in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            0, 3, 1, 2
        )  # (groups,out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.groups,
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(2) == self.in_features

        base_output = torch.bmm(
            self.base_activation(x), self.base_weight.permute(0, 2, 1)
        )
        spline_output = torch.bmm(
            self.b_splines(x).view(x.size(0), x.size(1), -1),
            self.scaled_spline_weight.view(x.size(0), self.out_features, -1).permute(
                0, 2, 1
            ),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 3 and x.size(2) == self.in_features
        batch = x.size(1)

        splines = self.b_splines(x)  # (groups, batch, in, coeff)
        splines = splines.permute(0, 2, 1, 3)  # (groups,in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (groups, out, in, coeff)
        orig_coeff = orig_coeff.permute(0, 2, 3, 1)  # (groups, in, coeff, out)
        unreduced_spline_output = torch.matmul(
            splines, orig_coeff
        )  # (groups, in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            0, 2, 1, 3
        )  # (groups, batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=1)[0]
        grid_adaptive = x_sorted[
            :,
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            ),
        ]

        uniform_step = (x_sorted[:, -1] - x_sorted[:, 0] + 2 * margin) / self.grid_size

        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device)
            .unsqueeze(1)
            .unsqueeze(0)
            * uniform_step.unsqueeze(1)
            + x_sorted[:, 0].unsqueeze(1)
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:, :1]
                - uniform_step.unsqueeze(1)
                * torch.arange(self.spline_order, 0, -1, device=x.device)
                .unsqueeze(1)
                .unsqueeze(0),
                grid,
                grid[:, -1:]
                + uniform_step.unsqueeze(1)
                * torch.arange(1, self.spline_order + 1, device=x.device)
                .unsqueeze(1)
                .unsqueeze(0),
            ],
            dim=1,
        )

        self.grid.copy_(grid.transpose(-1, -2))
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class ConvKAN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int or tuple = 3,
        stride: int or tuple = 1,
        padding: int or tuple = 0,
        dilation: int or tuple = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: torch.nn.Module = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
    ):
        """
        Convolutional layer with KAN kernels. A drop-in replacement for torch.nn.Conv2d.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel. Default: 3
            stride (int or tuple): Stride of the convolution. Default: 1
            padding (int or tuple): Padding added to both sides of the input. Default: 0
            dilation (int or tuple): Spacing between kernel elements. Default: 1
            groups (int): Number of blocked connections from input channels to output channels. Default: 1
            padding_mode (str): Padding mode. Default: 'zeros'
            bias (bool): Added for compatibility with torch.nn.Conv2d and does make any effect. Default: True
            grid_size (int): Number of grid points for the spline. Default: 5
            spline_order (int): Order of the spline. Default: 3
            scale_noise (float): Scale of the noise. Default: 0.1
            scale_base (float): Scale of the base. Default: 1.0
            scale_spline (float): Scale of the spline. Default: 1.0
            enable_standalone_scale_spline (bool): Enable standalone scale for the spline. Default: True
            base_activation (torch.nn.Module): Activation function for the base. Default: torch.nn.SiLU
            grid_eps (float): Epsilon for the grid
            grid_range (tuple): Range of the grid. Default: (-1, 1).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        self._in_dim = (
            (in_channels // groups) * self.kernel_size[0] * self.kernel_size[1]
        )
        self._reversed_padding_repeated_twice = tuple(
            x for x in reversed(self.padding) for _ in range(2)
        )

        if not bias:
            # warn the user that bias is not used
            warnings.warn("Bias is not used in ConvKAN layer", UserWarning)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.kan_layer = KANLinear(
            self._in_dim,
            out_channels // groups,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            groups=groups,
        )

    def forward(self, x):
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)  # Reset padding because we already applied it
        else:
            padding = self.padding

        x_unf = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=padding,
            stride=self.stride,
            dilation=self.dilation,
        )

        batch_size, channels_and_elem, n_patches = x_unf.shape

        # Ensuring group separation is maintained in the input
        x_unf = (
            x_unf.permute(0, 2, 1)  # [B, H_out * W_out, channels * elems]
            .reshape(
                batch_size * n_patches, self.groups, channels_and_elem // self.groups
            )  # [B * H_out * W_out, groups, out_channels // groups]
            .permute(1, 0, 2)
        )  # [groups, B * H_out * W_out, out_channels // groups]

        output = self.kan_layer(
            x_unf
        )  # [groups, B * H_out * W_out, out_channels // groups]
        output = (
            output.permute(1, 0, 2).reshape(batch_size, n_patches, -1).permute(0, 2, 1)
        )

        # Compute output dimensions
        output_height = (
            x.shape[2]
            + 2 * padding[0]
            - self.dilation[0] * (self.kernel_size[0] - 1)
            - 1
        ) // self.stride[0] + 1
        output_width = (
            x.shape[3]
            + 2 * padding[1]
            - self.dilation[1] * (self.kernel_size[1] - 1)
            - 1
        ) // self.stride[1] + 1

        # Reshape output to the expected output format
        output = output.view(
            x.shape[0],  # batch size
            self.out_channels,  # total output channels
            output_height,
            output_width,
        )

        return output


def _pair(x):
    if isinstance(x, (int, float)):
        return x, x
    return x

class CNextBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        # self.dwconv = nn.Conv2d(
        #     dim, dim, kernel_size=7, padding=3, groups=dim
        # )  # depthwise conv
        self.dwconv = ConvKAN(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = CustomLayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(
        #     dim, 4 * dim
        # )  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = KANLinear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = KANLinear(4 * dim, dim)
        self.gamma = (
            Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_ = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Iterable[int] = (3, 3, 9, 3),
        dims: Iterable[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        use_speed_perturb: bool = True,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        waveform_input: bool = False,
        return_clip_outputs: bool = True,
        return_frame_outputs: bool = False,
        use_specaug: bool = True,
    ) -> None:
        depths = list(depths)
        dims = list(dims)

        # --- Data augmentations
#         window = "hann"
#         center = True
#         pad_mode = "reflect"
#         ref = 1.0
#         amin = 1e-10
#         top_db = None

#         sample_rate = 32000
#         window_size = 1024
#         hop_size = 320
#         mel_bins = 224
#         fmin = 50
#         fmax = 14000

#         # note: build these layers even if waveform_input is False
#         # Spectrogram extractor
#         spectrogram_extractor = Spectrogram(
#             n_fft=window_size,
#             hop_length=hop_size,
#             win_length=window_size,
#             window=window,
#             center=center,
#             pad_mode=pad_mode,
#             freeze_parameters=True,
#         )
#         # Logmel feature extractor
#         logmel_extractor = LogmelFilterBank(
#             sr=sample_rate,
#             n_fft=window_size,
#             n_mels=mel_bins,
#             fmin=fmin,
#             fmax=fmax,
#             ref=ref,
#             amin=amin,
#             top_db=top_db,  # type: ignore
#             freeze_parameters=True,
#         )

#         # Spec augmenter
#         freq_drop_width = 28  # 28 = 8*224//64, in order to be the same as the nb of bins dropped in Cnn14

#         if use_specaug:
#             spec_augmenter = SpecAugmentation(
#                 time_drop_width=64,
#                 time_stripes_num=2,
#                 freq_drop_width=freq_drop_width,
#                 freq_stripes_num=2,
#             )
#         else:
#             spec_augmenter = nn.Identity()

#         if use_speed_perturb:
#             speed_perturb = SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
#         else:
#             speed_perturb = nn.Identity()

        # --- Layers
        bn0 = nn.BatchNorm2d(224)

        # stem and 3 intermediate downsampling conv layers
        downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            # nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            ConvKAN(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            CustomLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        downsample_layers.append(stem)

        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                CustomLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                # nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                ConvKAN(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            blocks = [
                CNextBlock(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depths[i])
            ]
            stage = nn.Sequential(*blocks)
            stages.append(stage)
            cur += depths[i]

        norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        # head_audioset = nn.Linear(dims[-1], num_classes)
        head_audioset = KANLinear(dims[-1], num_classes)

        super().__init__()
        self.waveform_input = waveform_input
        self.return_clip_outputs = return_clip_outputs
        self.return_frame_outputs = return_frame_outputs
#         self.spectrogram_extractor = spectrogram_extractor
#         self.logmel_extractor = logmel_extractor

#         self.use_specaug = use_specaug
#         self.spec_augmenter = spec_augmenter
#         self.use_speed_perturb = use_speed_perturb
#         self.speed_perturb = speed_perturb

        self.bn0 = bn0
        self.downsample_layers = downsample_layers
        self.stages = stages
        self.norm = norm
        self.head_audioset = head_audioset

        self.apply(self._init_weights)
        # self.head_audioset.weight.data.mul_(head_init_scale)
        # self.head_audioset.bias.data.mul_(head_init_scale)

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _init_weights(self, m) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward_features(self, x: Tensor):
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = torch.mean(x, dim=3)

        # Mean+Max pooling
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.norm(x)  # global average+max pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(
        self,
        audio: Tensor,
        audio_shapes: Tensor,
        mixup_lambda: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.waveform_input:
            assert audio.ndim == 2, f"{audio.ndim=}"
            input_time_dim = -1
            x = self.spectrogram_extractor(audio)
            x = self.logmel_extractor(x)
        else:
            assert audio.ndim == 4, f"{audio.ndim=}"
            input_time_dim = -2
            x = audio

        # x: (batch_size, 1, time_steps, mel_bins)

#         if self.training and self.use_speed_perturb:
#             x = self.speed_perturb(x)

#         x = x.transpose(1, 3)
#         x = self.bn0(x)
#         x = x.transpose(1, 3)

#         if self.training:
#             x = self.spec_augmenter(x)

#         # Mixup on spectrogram
#         if self.training and mixup_lambda is not None:
#             x = do_mixup(x, mixup_lambda)

        # forward features with frame_embs
#         print(x.dtype)
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = torch.mean(x, dim=3)

        output_dict = {}
        if self.return_frame_outputs:
            frame_embs = x
            # (bsize, emb_size=768, n_frames=31)
            frame_time_dim = 2

            audio_lens = audio_shapes[:, input_time_dim]
            reduction_factor = (
                frame_embs.shape[frame_time_dim] / audio.shape[input_time_dim]
            )
            frame_embs_lens = (audio_lens * reduction_factor).round().int()

            frame_embs_shape = torch.as_tensor(
                [frame_embs_i.shape for frame_embs_i in frame_embs],
                device=frame_embs.device,
            )
            frame_embs_shape[:, frame_time_dim - 1] = frame_embs_lens

            output_dict |= {
                # (bsize, embed=768, n_frames=31)
                "frame_embs": frame_embs,
                # (bsize, 3)
                "frame_embs_shape": frame_embs_shape,
                # (bsize,)
                "frame_embs_lens": frame_embs_lens,
                # ()
                "frame_time_dim": frame_time_dim,
            }

        if self.return_clip_outputs:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2

            x = self.norm(x)  # global average+max pooling, (N, C, H, W) -> (N, C)
            # end forward features

            x = self.head_audioset(x)
            clipwise_output = torch.sigmoid(x)
            output_dict |= {"clipwise_output": clipwise_output}

        return x, output_dict

CNEXT_IMAGENET_PRETRAINED_URLS = {
    "convnext_atto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth",
    "convnext_femto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth",
    "convnext_pico_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth",
    "convnext_nano_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth",
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def convnext_small(
    pretrained: bool = False,
    strict: bool = False,
    in_22k: bool = False,
    drop_path_rate: float = 0.1,
    after_stem_dim=[56],
    **kwargs,
) -> ConvNeXt:
    model = ConvNeXt(
        in_chans=3,
        num_classes=182,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
    if pretrained:
        url = (
            CNEXT_IMAGENET_PRETRAINED_URLS["convnext_small_22k"]
            if in_22k
            else CNEXT_IMAGENET_PRETRAINED_URLS["convnext_small_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")  # type: ignore
        model.load_state_dict(checkpoint["model"], strict=strict)

        if len(after_stem_dim) < 2:
            if after_stem_dim[0] == 56:
                stem_audioset = nn.Conv2d(
                    1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
                )
            elif after_stem_dim[0] == 112:
                stem_audioset = nn.Conv2d(
                    1, 96, kernel_size=(9, 2), stride=(9, 2), padding=(4, 0)
                )
        else:
            if after_stem_dim == [252, 56]:
                stem_audioset = nn.Conv2d(
                    1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)
                )

        trunc_normal_(stem_audioset.weight, std=0.02)  # type: ignore
        nn.init.constant_(stem_audioset.bias, 0)  # type: ignore
        model.downsample_layers[0][0] = stem_audioset  # type: ignore

    return model

def convnext_tiny(
    pretrained: bool = False,
    strict: bool = False,
    in_22k: bool = False,
    drop_path_rate: float = 0.1,
    after_stem_dim: Iterable[int] = (56,),
    use_speed_perturb: bool = True,
    **kwargs,
) -> ConvNeXt:
    after_stem_dim = list(after_stem_dim)

    model = ConvNeXt(
        in_chans=1,
        num_classes=182,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        use_speed_perturb=use_speed_perturb,
        **kwargs,
    )

    if pretrained:
        url = (
            CNEXT_IMAGENET_PRETRAINED_URLS["convnext_tiny_22k"]
            if in_22k
            else CNEXT_IMAGENET_PRETRAINED_URLS["convnext_tiny_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True  # type: ignore
        )
        model.load_state_dict(checkpoint["model"], strict=strict)

    # stem_audioset = nn.Conv2d(
    #     1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
    # )
    stem_audioset = ConvKAN(
        1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
    )
    if len(after_stem_dim) < 2:
        if after_stem_dim[0] == 56:
            # stem_audioset = nn.Conv2d(
            #     1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
            # )
            stem_audioset = ConvKAN(
                1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
            )
        elif after_stem_dim[0] == 112:
            # stem_audioset = nn.Conv2d(
            #     1, 96, kernel_size=(9, 2), stride=(9, 2), padding=(4, 0)
            # )
            stem_audioset = ConvKAN(
                1, 96, kernel_size=(9, 2), stride=(9, 2), padding=(4, 0)
            )
        else:
            raise ValueError(
                "ERROR: after_stem_dim can be set to 56 or 112 or [252,56]"
            )
    else:
        if after_stem_dim == [252, 56]:
            # stem_audioset = nn.Conv2d(
            #     1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)
            # )
            stem_audioset = ConvKAN(
                1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)
            )
        elif after_stem_dim == [504, 28]:
            # stem_audioset = nn.Conv2d(
            #     1, 96, kernel_size=(4, 8), stride=(2, 8), padding=(5, 0)
            # )
            stem_audioset = ConvKAN(
                1, 96, kernel_size=(4, 8), stride=(2, 8), padding=(5, 0)
            )
        elif after_stem_dim == [504, 56]:
            # stem_audioset = nn.Conv2d(
            #     1, 96, kernel_size=(4, 4), stride=(2, 4), padding=(5, 0)
            # )
            stem_audioset = ConvKAN(
                1, 96, kernel_size=(4, 4), stride=(2, 4), padding=(5, 0)
            )
        else:
            raise ValueError(
                "ERROR: after_stem_dim can be set to 56 or 112 or [252,56]"
            )

    # trunc_normal_(stem_audioset.weight, std=0.02)
    # nn.init.constant_(stem_audioset.bias, 0)  # type: ignore
    model.downsample_layers[0][0] = stem_audioset  # type: ignore

    return model

from  torch.cuda.amp import autocast

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets

def train_one_epoch(model, loader, optimizer, scaler=None):
    model.train()
    losses = AverageMeter()
    gt = []
    preds = []
    bar = tqdm(loader, total=len(loader))
    for batch in bar:
        optimizer.zero_grad()
        spec = batch['spec']
        target = batch['target']
        
        spec_shapes = torch.Tensor([item.shape[-2] for item in spec])

        spec, target = mixup(spec, target, 0.5)
#         print(spec.shape)
#         print(target.shape)
#         exit()
        
        spec = spec.type(torch.cuda.HalfTensor)
        spec_shapes = spec_shapes.type(torch.cuda.HalfTensor)
        target = target.type(torch.cuda.HalfTensor)
        
        spec = spec.to(device)
        spec_shapes = spec_shapes.to(device)
        target = target.to(device)
#         print(spec.dtype)
#         print(spec_shapes.dtype)
#         print(target.dtype)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _ = model(spec, spec_shapes)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.cuda.amp.autocast():
                logits, _ = model(spec, spec_shapes)
                loss = criterion(logits, target)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

        losses.update(loss.item(), batch["spec"].size(0))
        bar.set_postfix(
            loss=losses.avg,
            grad=grad_norm.item(),
            lr=optimizer.param_groups[0]["lr"]
        )
        gt.append(target.cpu().detach().numpy())
        preds.append(logits.sigmoid().cpu().detach().numpy())
    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    scores = calculate_competition_metrics_no_map(gt, preds, target_columns)

    return scores, losses.avg


def valid_one_epoch(model, loader):
    model.eval()
    losses = AverageMeter()
    bar = tqdm(loader, total=len(loader))
    gt = []
    preds = []

    with torch.no_grad():
        for batch in bar:
            spec = batch['spec']
            target = batch['target']
            spec_shapes = torch.Tensor([item.shape[-2] for item in spec])
            spec = spec.type(torch.cuda.HalfTensor)
            spec_shapes = spec_shapes.type(torch.cuda.HalfTensor)
            target = target.type(torch.cuda.HalfTensor)
            
            spec = spec.to(device)
            spec_shapes = spec_shapes.to(device)
            target = target.to(device)
            
            with torch.cuda.amp.autocast():
                logits, _ = model(spec, spec_shapes)
                loss = criterion(logits, target)

                losses.update(loss.item(), batch["spec"].size(0))

            gt.append(target.cpu().detach().numpy())
            preds.append(logits.sigmoid().cpu().detach().numpy())

            bar.set_postfix(loss=losses.avg)

    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    scores = calculate_competition_metrics_no_map(gt, preds, target_columns)
    return scores, losses.avg

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

# Fix Warmup Bug
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        

transforms_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Resize(image_size, image_size),
    A.CoarseDropout(max_height=int(image_size * 0.375), max_width=int(image_size * 0.375), max_holes=1, p=0.7),
    A.Normalize(mean=(0.485), std=(0.229))
])

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485), std=(0.229))
])


def train_fold():
    logger = init_logger(log_file=os.path.join(output_folder, exp_name, f"{fold}.log"))

    logger.info("=" * 90)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 90)

    trn_df = df[df['fold'] != fold].reset_index(drop=True)
    # trn_df = trn_df.head(30*batch_size)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    print(trn_df.shape)
    logger.info(trn_df.shape)
    logger.info(trn_df['primary_label'].value_counts())
    logger.info(val_df.shape)
    logger.info(val_df['primary_label'].value_counts())


    trn_dataset = BirdDataset(df=trn_df.reset_index(drop=True), transform=transforms_train, add_secondary_labels=True)
    v_ds = BirdDataset(df=val_df.reset_index(drop=True), transform=transforms_val, add_secondary_labels=True)


    train_loader = torch.utils.data.DataLoader(trn_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(v_ds, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=num_workers, pin_memory=True)


    # model = CNN(backbone=backbone, pretrained=True).to(device)
    # model = convnext_small().to(device)
    model = convnext_tiny(pretrained=False, in_22k=True).to(device)
    # model = model.half()
    optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)


    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    patience = early_stopping
    best_score = 0.0
    n_patience = 0

    for epoch in range(1, n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)

        scheduler_warmup.step(epoch-1)

        train_scores, train_losses_avg = train_one_epoch(model, train_loader, optimizer, scaler=None)
        train_scores_str = metrics_to_string(train_scores, "Train")
        train_info = f"Epoch {epoch} - Train loss: {train_losses_avg:.4f}, {train_scores_str}"
        logger.info(train_info)

        val_scores, val_losses_avg = valid_one_epoch(model, val_loader)
        val_scores_str = metrics_to_string(val_scores, f"Valid")
        val_info = f"Epoch {epoch} - Valid loss: {val_losses_avg:.4f}, {val_scores_str}"
        logger.info(val_info)

        val_score = val_scores["ROC"]

        is_better = val_score > best_score
        best_score = max(val_score, best_score)

        if is_better:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_score,
                "optimizer": optimizer.state_dict(),
            }
            logger.info(
                f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model\n")
            torch.save(
                state,
                os.path.join(output_folder, exp_name, f"{fold}.bin")
            )
            n_patience = 0
        else:
            n_patience += 1
            logger.info(
                f"Valid loss didn't improve last {n_patience} epochs.\n")

        if n_patience >= patience:
            logger.info(
                "Early stop, Training End.\n")
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": best_score,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                state,
                os.path.join(output_folder, exp_name, f"final_{fold}.bin")
            )
            break

    del model
    torch.cuda.empty_cache()
    gc.collect()

import torch
torch.cuda.empty_cache()
train_fold()
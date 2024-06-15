import glob
import os
import random
import sys
from pathlib import Path

!pip install --no-index --find-links /kaggle/input/torchlibrosa torchlibrosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
import concurrent.futures
import shutil
import albumentations as A
import torchaudio

import time
from torch.optim import AdamW
import albumentations as A
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

import warnings
warnings.filterwarnings("ignore")
    
device = 'cpu'
# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sub = pd.read_csv("birdclef-2024/sample_submission.csv")
target_columns_ = sub.columns.tolist()
target_columns = sub.columns.tolist()[1:]

TOTAL_SECONDS_CHUNKS = 48
test_path = "birdclef-2024/test_soundscapes/"
files = glob.glob(f'{test_path}*')
if len(files) == 1:
    TOTAL_SECONDS_CHUNKS = 2

seconds = [i for i in range(5, (TOTAL_SECONDS_CHUNKS*5) + 5, 5)]


test_path = "birdclef-2024/test_soundscapes/"

files = glob.glob(f'{test_path}*')
if len(files) == 1:
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1442779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446379.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1146779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1426779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1441779.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446179.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446719.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446771.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1446789.ogg')
    shutil.copy('birdclef-2024/train_audio/redspu1/XC312771.ogg', 'temp/soundscape_1448779.ogg')
    test_path = "temp/"
    
print (test_path)

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

transforms_val = A.Compose([
    A.Resize(256, 256),
    A.Normalize()
])

class TestDataset(torchdata.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 clip: np.ndarray,
                ):
        
        self.df = df
        self.clip = clip
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)
        self.transform = transforms_val

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        sample = self.df.loc[idx, :]
        row_id = sample.row_id

        end_seconds = int(sample.seconds)
        start_seconds = int(end_seconds - 5)
        
        wave = self.clip[:, 32000 * start_seconds : 32000 * end_seconds]
        
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wave)))
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        
        res = self.transform(image=mel_spectrogram)
        spec = res['image'].astype(np.float32)
        spec = spec.transpose(2, 0, 1)
        
        return {
            "row_id": row_id,
            "wave": spec,
        }
    
import math
import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import itertools
from typing import Any, Callable
from typing import Iterable

lr_max = 1e-5
lr_min = 1e-7
weight_decay = 1e-6

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
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = CustomLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
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
            nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            CustomLayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        downsample_layers.append(stem)

        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                CustomLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
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
        head_audioset = nn.Linear(dims[-1], num_classes)

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
        self.head_audioset.weight.data.mul_(head_init_scale)
        self.head_audioset.bias.data.mul_(head_init_scale)

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
#             frame_time_dim = 2

#             audio_lens = audio_shapes[:, input_time_dim]
#             reduction_factor = (
#                 frame_embs.shape[frame_time_dim] / audio.shape[input_time_dim]
#             )
#             frame_embs_lens = (audio_lens * reduction_factor).round().int()

#             frame_embs_shape = torch.as_tensor(
#                 [frame_embs_i.shape for frame_embs_i in frame_embs],
#                 device=frame_embs.device,
#             )
#             frame_embs_shape[:, frame_time_dim - 1] = frame_embs_lens

#             output_dict |= {
#                 # (bsize, embed=768, n_frames=31)
#                 "frame_embs": frame_embs,
#                 # (bsize, 3)
#                 "frame_embs_shape": frame_embs_shape,
#                 # (bsize,)
#                 "frame_embs_lens": frame_embs_lens,
#                 # ()
#                 "frame_time_dim": frame_time_dim,
#             }

        if self.return_clip_outputs:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2

            x = self.norm(x)  # global average+max pooling, (N, C, H, W) -> (N, C)
            # end forward features

            x = self.head_audioset(x)
#             clipwise_output = torch.sigmoid(x)
#             output_dict |= {"clipwise_output": clipwise_output}

        return x

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

model = convnext_small().to(device)
# model = model.half()
optimizer = AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)

checkpoint = torch.load("/kaggle/input/convnext-bin-file/final_2.bin", map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()
print ("Loaded")

def prediction_for_clip(audio_path):
    
    prediction_dict = {}
    
    wav, org_sr = torchaudio.load(audio_path, normalize=True)
    clip = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=32000)
    
    name_ = audio_path.split(".ogg")[0].split("/")[-1]
    row_ids = [name_+f"_{second}" for second in seconds]

    test_df = pd.DataFrame({
        "row_id": row_ids,
        "seconds": seconds,
    })
    
    dataset = TestDataset(
        df=test_df, 
        clip=clip,
    )
        
    loader = torchdata.DataLoader(
        dataset,
        batch_size=4, 
        num_workers=os.cpu_count(),
        drop_last=False,
        shuffle=False,
        pin_memory=True
    )
    
    for inputs in loader:

        row_ids = inputs['row_id']
        inputs.pop('row_id')

        for row_id in row_ids:
            if row_id not in prediction_dict:
                prediction_dict[str(row_id)] = []

        probas = []

        with torch.no_grad():
            output = model(inputs["wave"], torch.Tensor([item.shape[-2] for item in inputs["wave"]]))

        for row_id_idx, row_id in enumerate(row_ids):
            prediction_dict[str(row_id)].append(output[row_id_idx, :].sigmoid().detach().numpy())
                                                        
    for row_id in list(prediction_dict.keys()):
        logits = prediction_dict[row_id]
        logits = np.array(logits)[0]#.mean(0)
        prediction_dict[row_id] = {}
        for label in range(len(target_columns)):
            prediction_dict[row_id][target_columns[label]] = logits[label]

    return prediction_dict


def main():
    
    
    all_audios = list(glob.glob(f'{test_path}*.ogg'))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        dicts = list(executor.map(prediction_for_clip, all_audios))
    
    prediction_dicts = {}
    for d in dicts:
        prediction_dicts.update(d)
        
    submission = pd.DataFrame.from_dict(prediction_dicts, "index").rename_axis("row_id").reset_index()
    submission.to_csv("submission.csv", index=False)
    print ("Done")


if __name__ == "__main__":
    main()
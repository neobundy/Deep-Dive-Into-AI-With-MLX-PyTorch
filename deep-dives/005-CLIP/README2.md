# Deep Dive into CLIP Part II - Diving into Codebase

![samurai.jpeg](images%2Fsamurai.jpeg)

**🏠 Official Repo**: https://github.com/openai/CLIP

We're going to explore the inner workings of CLIP, focusing on its model implementation and the way it handles both images and text. Our journey will stick to the main elements of the model, aiming for a broad perspective on its structure and how it operates.

Part II builds upon the knowledge you've acquired from earlier resources, and it's expected that you come equipped with an understanding of several key concepts:

- The intricacies of attention mechanisms and the Transformer architecture.
- The fundamentals and applications of Convolutional Neural Networks (CNNs).
- The principles of tokenization and the nuances of processing textual data.
- Proficiency in using PyTorch, particularly its modules related to neural networks.
- Techniques for image processing and the methods used to extract features from visual data.

**In Sidebars**:

[Attention Is All You Need For Now](..%2F..%2Fbook%2Fsidebars%2Fattention-is-all-you-need-for-now%2FAttention-Is-All-You-Need-For-Now.md)

**In 1st Book**:

[Chapter 13 - Tenny the Transformer](..%2F..%2Fbook%2F013-tenny-the-transformer%2FREADME.md)

[Chapter 14 - Tenny the Transformer Sentiment Analyst](..%2F..%2Fbook%2F014-tenny-the-transformer-sentiment-analyst-with-an-attitude%2FREADME.md)

[Chapter 15 - Tenny the Transformer Sentiment Analyst with an Attitude](..%2F..%2Fbook%2F015-tenny-the-transformer-sentiment-analyst-with-an-attitude-is-born%2FREADME.md)

[Chapter 16 - Tenny the Convoluter](..%2F..%2Fbook%2F016-tenny-the-convoluter%2FREADME.md)

[Chapter 17 - Tenny the Vision Weaver](..%2F..%2Fbook%2F017-tenny-the-vision-weaver%2FREADME.md)

**In 2nd Book**:

[Chapter 6 - Menny the Face Detector](..%2F..%2Fmlx-book%2F006-menny-the-face-detector%2FREADME.md)

These topics have been thoroughly explored and explained in the preceding two books, various informative sidebars, and in-depth discussions presented earlier in the series.

## Tokenizer - `simple_tokenizer.py`

The `simple_tokenizer.py` script implements a tokenizer used in the CLIP model. This tokenizer is responsible for converting text into a format that the model can understand and process, using Byte Pair Encoding (BPE). 

```python
import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
```

1. **Imports and Helper Functions**:
   - The code imports necessary libraries and defines helper functions. `gzip` for reading compressed files, `html` for HTML unescaping, `os` for OS-level operations, `ftfy` for fixing text encoding issues, and `regex` for regular expression operations.
   - `lru_cache` is a decorator that caches the results of functions to avoid repeated calculations.

2. **Path to BPE Vocabulary**:
   - `default_bpe()` returns the file path to the BPE vocabulary file, which is essential for the tokenizer to work.

3. **Byte-Pair Encoding Preparation**:
   - `bytes_to_unicode()` creates a mapping of bytes to Unicode characters, which is used for encoding and decoding text in the BPE process.

4. **Tokenization Helpers**:
   - `get_pairs()` identifies all the unique pairs of symbols (characters or bytes) in a word, which is a step in the BPE algorithm.
   - `basic_clean()` uses `ftfy` to fix text encoding and unescape HTML entities twice, which ensures that the text is properly formatted for processing.
   - `whitespace_clean()` removes extra spaces and ensures the text is clean and consistent.

5. **SimpleTokenizer Class**:
   - This class encapsulates the entire functionality of the tokenizer.
   - Upon initialization, it prepares the byte encoder and decoder, loads the BPE merges from the vocabulary file, and builds the encoder and decoder dictionaries.
   - The BPE ranks are also prepared, which are used to determine the order of merges during the tokenization process.
   - A regex pattern (`pat`) is compiled to match different kinds of text tokens including words, numbers, and special tokens.

6. **BPE Tokenization**:
   - The `bpe()` method applies the BPE algorithm to a given token, gradually merging pairs of symbols based on the ranks determined during initialization.
   - The `encode()` method takes a string of text, cleans it, and converts it into a sequence of BPE tokens. It does this by finding all tokens that match the compiled pattern, encoding them to bytes, applying BPE, and then converting the BPE tokens into their corresponding ids in the vocabulary.
   - The `decode()` method reverses the process, converting a sequence of token ids back into a human-readable string. It replaces the BPE end-of-word markers (`</w>`) with spaces to reconstruct the original text.

This tokenizer is a key component for the text processing in CLIP, allowing it to handle a diverse range of textual inputs efficiently and effectively, which are then used for downstream tasks such as matching with images or other text.

In the given context of the `simple_tokenizer.py` script, the file `bpe_simple_vocab_16e6.txt.gz` is likely a compressed text file that contains the vocabulary for the Byte Pair Encoding (BPE) tokenizer used within the CLIP model.

## Vocabulary Storage - `bpe_simple_vocab_16e6.txt.gz`

1. **Vocabulary Storage**: This file stores a pre-defined list of tokens, which are the most common sequences of characters found in the text data that CLIP was trained on. These tokens are the building blocks that the BPE algorithm uses to encode and decode text.

2. **BPE Merges**: The vocabulary file not only includes individual tokens (like words or subwords) but also the rules for merging pairs of characters or character sequences. These rules are derived from a large corpus of text and dictate how the tokenizer should combine characters to form tokens.

3. **Efficient Text Encoding**: By using BPE, the tokenizer can efficiently process text by breaking it down into these common sequences, which reduces the model's complexity and makes it better at handling a variety of text inputs, including those it hasn't seen before.

4. **Compression**: The `.gz` extension indicates that the vocabulary file is compressed using the gzip compression algorithm, making it smaller and faster to load into memory. The `simple_tokenizer.py` script would use the `gzip` library to read this file and load the vocabulary into memory for the tokenizer to use.

The `bpe_simple_vocab_16e6.txt.gz` file is a crucial resource for the CLIP model's tokenizer, enabling it to convert text into a sequence of tokens that the model can understand and process, which is a critical step in aligning text with images in the way that CLIP does.

## CLIP Utils - `clip.py`

The `clip.py` file provides utilities to load and use the CLIP model. 

```python
import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
```

1. **Imports and Preliminaries**:
   - The script imports required libraries and checks for the right version of PyTorch, issuing a warning if the version is less than 1.7.1.
   - It defines a dictionary `_MODELS` that maps model names to their download URLs.

2. **Download Function (_download)**:
   - The `_download` function is responsible for downloading the model weights from a given URL if they are not already present in the specified directory.
   - It performs a SHA-256 checksum to ensure the integrity of the downloaded file.

3. **Image Transformation (_transform)**:
   - Defines a transformation pipeline for images that includes resizing, center cropping, converting to RGB, transforming to a tensor, and normalizing with predefined mean and standard deviation values.

4. **Model Availability (available_models)**:
   - A utility function that lists all the available CLIP models that can be loaded.

5. **Model Loading (load)**:
   - The `load` function downloads (if necessary) and loads a specified CLIP model into memory, placing it on the desired device (CPU or GPU).
   - It also prepares the image preprocessing function that should be applied to images before passing them to the model.
   - The function handles both JIT-compiled and non-JIT models and patches the device and data type settings to match the user's configuration.

6. **Tokenization (tokenize)**:
   - The `tokenize` function takes text input and tokenizes it using the CLIP tokenizer, padding or truncating to a specified context length.
   - The tokens are returned as a tensor suitable for input into the CLIP model.

This file serves as an interface to easily load and use the CLIP model, including pre-processing images and tokenizing text, making it accessible for tasks such as zero-shot classification, where the model can predict the content of an image given a text description without having been explicitly trained on that task.

## Model Architecture - `model.py`

The `model.py` script defines the neural network architectures used within CLIP.

```python
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
```

1. **Bottleneck Module**:
   - A standard bottleneck architecture commonly used in ResNet models. It includes three convolutional layers with batch normalization and ReLU activations. If downsampling is needed, it adds an average pooling layer followed by a downsample layer.

2. **AttentionPool2d Module**:
   - A custom pooling layer that uses attention, instead of the usual average pooling. It applies a multi-head attention mechanism to the spatial dimensions of its input feature map.

3. **ModifiedResNet Class**:
   - A variant of the ResNet architecture, adapted for use in CLIP. It includes a custom stem with three convolutional layers and average pooling instead of max pooling, followed by four stages of bottleneck layers.

4. **LayerNorm Class**:
   - A subclass of PyTorch's LayerNorm that ensures compatibility with mixed precision training by handling `fp16` inputs.

5. **QuickGELU Class**:
   - A quick approximation of the GELU activation function, which is computationally more efficient.

6. **ResidualAttentionBlock Class**:
   - A block combining multi-head self-attention and a feed-forward network, both followed by skip connections and layer normalization.

7. **Transformer Class**:
   - Defines a sequence of residual attention blocks, constituting the transformer architecture used in the text processing part of CLIP.

8. **VisionTransformer Class**:
   - Implements the Vision Transformer (ViT) architecture, which processes images as a sequence of flattened 2D patches and applies self-attention mechanisms to them.

9. **CLIP Class**:
   - The main class that integrates both the vision and text components of the model. It can encode images and text separately and computes the similarity between the two modalities.
   - Includes methods for initializing parameters and building the attention mask required for the transformer's self-attention.

10. **convert_weights Function**:
    - Converts the model's weights to `fp16` for mixed precision training, which can reduce memory usage and improve performance on compatible hardware.

11. **build_model Function**:
    - Constructs the CLIP model from a state dictionary, which contains the pre-trained weights. It determines the architecture to use based on the keys present in the state dictionary.
    - Loads the state dictionary into the model and prepares it for evaluation.

This script is responsible for constructing the architecture of the CLIP model, setting up the necessary layers and blocks, and providing functionality for loading pre-trained weights. It defines how the image and text inputs are processed through their respective pathways within the model.

#### More on AttentionPool2d - A Key Component of CLIP

Let's delve deeper into the `AttentionPool2d` module, which is a pivotal component of the CLIP architecture.

The **AttentionPool2d** module is an innovative adaptation of pooling layers that leverages the attention mechanism, a key innovation in deep learning, particularly for tasks involving sequence modeling and transduction problems like language translation and image recognition.

Traditional pooling layers like average pooling work by downsampling the feature map, typically by taking the average value of a patch of neurons in the input feature map. This operation reduces the spatial resolution, thus reducing the computational complexity for subsequent layers. However, this process is purely mechanical and doesn't take into account the content of the feature map, potentially leading to the loss of important information.

The `AttentionPool2d` module, on the other hand, brings a context-aware processing step to the pooling operation. It does not simply downsample based on local patches but uses a multi-head self-attention mechanism to weigh the importance of different regions in the feature map. Here's how it works:

1. **Spatial Dimension Handling**: The module first flattens the spatial dimensions of the input feature map (height and width) while preserving the batch and channel dimensions. This flattening is necessary because attention mechanisms typically operate on sequences, so the 2D feature map is treated as a sequence of patches.

2. **Positional Embeddings**: To maintain the spatial information which is crucial for image tasks, the module adds positional embeddings to the flattened feature map. These embeddings encode the original position of each patch in the feature map, allowing the model to understand the relative or absolute position of the patches in the 2D space.

3. **Multi-Head Attention**: The core of the module is the application of the multi-head attention mechanism. It allows the model to focus on different parts of the feature map when computing the representation of a particular patch. In other words, it helps the network to learn where to pay attention in the spatial domain of the feature map, based on the content of the entire image.

4. **Output Projection**: After computing the attention, the module projects the attended feature map to either the same dimensionality or a different one specified by `output_dim`. This projection is similar to the fully connected layers in CNNs and serves to transform the attended features into a space that is more suitable for the task at hand.

The `AttentionPool2d` module essentially replaces the standard pooling operation with a learnable attention-based pooling, which is more dynamic and content-aware. This is especially beneficial in a model like CLIP, where understanding the global context and nuanced details of an image is crucial for matching it with text descriptions accurately. This attention pooling layer thus contributes to the robustness and versatility of the CLIP model in various visual tasks.

## Using the CLIP Model

The CLIP model can be used for a variety of tasks, including zero-shot classification, image-text matching, and visual question answering. Here, we'll explore how to use the CLIP model for image and text processing tasks.

### API Overview

The `clip` module provides an interface to work with the CLIP model, which can understand and link visual and textual content. The following methods are part of the API:

- **clip.available_models()**: 
  Lists the names of pre-trained CLIP models that can be loaded. These are the models that have been made publicly available and can be utilized for various tasks.

```python
print(clip.available_models())
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
```

- **clip.load(name, device=..., jit=False)**: 
  This function loads a CLIP model specified by the `name` parameter. The model's name should be one of the strings returned by `clip.available_models()`. Alternatively, `name` can be a path to a model checkpoint on your local disk. If the model is not present locally, it will be downloaded. The `device` parameter allows you to specify the computational device (`'cuda'` or `'cpu'`) on which the model will run. The default behavior is to use a CUDA device if available, otherwise CPU. Setting `jit` to `False` means the non-JIT version of the model is loaded, which allows for greater model manipulation and inspection.

```python
model, preprocess = clip.load("ViT-B/32", device=device)
```

- **clip.tokenize(texts, context_length=77)**: 
  Converts a string or a list of strings into a tensor of tokenized sequences suitable for model input. It pads or truncates the sequences to a uniform length specified by `context_length`.

```python
text = clip.tokenize(["a puppy", "a girl", "glasses"]).to(device)
```

### Model Methods

Once loaded, the CLIP model object provides several methods for processing images and text:

- **model.encode_image(image)**: 
  Accepts a tensor representing a batch of images and returns a tensor of image features extracted by the vision component of the CLIP model.

- **model.encode_text(text)**: 
  Takes a tensor of tokenized text sequences and returns a tensor of text features encoded by the language component of the CLIP model.

- **model(image, text)**: 
  This method accepts a batch of images and a batch of tokenized text as input. It outputs two tensors with logit scores, which represent the cosine similarity between each image and text feature pair, multiplied by 100. These scores quantify the alignment between the visual and textual content as understood by the model.

By utilizing these methods, developers can employ the CLIP model for a variety of applications, such as zero-shot classification tasks, where the model can make predictions about images based on textual descriptions without any additional fine-tuning or training.

For more advanced use cases, refer to the official documentation and examples provided by OpenAI.

## Final Thoughts

![time.jpeg](images%2Ftime.jpeg)

In our journey through the intricacies of the CLIP model, we've uncovered the capabilities of this pioneering vision-language model, celebrated for its adeptness at bridging the gap between visual and textual content. CLIP, with its innovative use of a vast dataset and a multi-modal training regimen, has set new benchmarks in a variety of tasks that span both vision and language domains. This includes notable achievements in zero-shot classification and the precise alignment of images with textual descriptions.

For those interested in navigating the realms of image generation, text-to-image synthesis, or even tackling visual question answering challenges, CLIP emerges as a robust and adaptable framework. Its proficiency in deciphering and connecting disparate data modalities positions it as an invaluable resource for advancing projects in computer vision and natural language processing.

Equipping yourself with an understanding of CLIP not only simplifies grasping the mechanics behind models like Stable Diffusion but also enhances your ability to produce high-quality imagery through AI. Indeed, CLIP stands as a foundational pillar for those venturing into the world of multi-modal models.

Incorporating CLIP into your array of tools is not just recommended; it's essential for anyone aiming to excel in the evolving landscape of AI research and application.
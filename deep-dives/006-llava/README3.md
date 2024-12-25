# Deep Dive into LLaVA - Large Language and Vision Assistant Part III

![puppy.png](images%2Fpuppy.png)

✍️ [Part I](README.md) | ✍️ [Part II](README2.md) | ✍️ [Part III ](README3.md)

🏠 Official Repo: https://github.com/haotian-liu/LLaVA

## LLaVA-v1.6 - The New Benchmark in Visual Reasoning, OCR, and Knowledge Processing

Building on the momentum of LLaVA-1.5's successful launch in October 2023, the LLaVA team has unveiled LLaVA-1.6, featuring groundbreaking improvements in the realms of visual reasoning, OCR capabilities, and expansive world knowledge integration.

The team asserts that v1.6 surpasses the highly regarded Gemini Pro across multiple performance benchmarks.

1. **Elevated Image Resolution Handling:**
   - By quadrupling the input image pixels, LLaVA-1.6 provides a profound boost in capturing visual nuances across three supported aspect ratios, maxing out at impressive resolutions of 672x672, 336x1344, and 1344x336.

2. **Refined Visual Reasoning and OCR:**
   - The model showcases notably improved visual instruction tuning, leveraging a richer data mixture to elevate both its visual reasoning and OCR performances.

3. **Expanded Visual Conversation Competence:**
   - LLaVA-1.6 exhibits a broadened capability for visual conversations, adeptly tailored for a variety of applications, coupled with enriched world knowledge and sophisticated logical reasoning.

4. **Streamlined Deployment with SGLang:**
   - Efficiency is paramount, and LLaVA-1.6 champions this with SGLang's enhanced deployment and inference processes.

Whilst achieving these substantial performance milestones, LLaVA-1.6 remains committed to the virtues of its predecessor—simplicity, efficiency, and a data-conscious approach. This iteration continues to utilize the pretrained connector from LLaVA-1.5 and operates with under 1M visual instruction tuning samples.

Despite its sophisticated architecture, the 34B variant of LLaVA-1.6 remarkably finished its training cycle within a mere 24 hours utilizing 32 A100 GPUs.

?? We will focus exclusively on the fundamental components of the LLaVA architecture, setting aside aspects related to training, evaluation, deployment, and auxiliary utilities. If you're interested in exploring these areas, please refer to the official documentation and resources provided by the LLaVA team.

### Multi-Modal Language Model - Swapping out LLMLs : `language_model/llava_llama.py`

The `llava_llama.py` script illustrates how to substitute the default LLaVA's LLM (e.g., LLaMA) with another LLM of choice. This flexibility is showcased in the `llava_llama.py` implementation, guiding through the replacement process.

```python
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
```

- **Multimodal Input Processing**: The primary modification involves processing multimodal inputs, particularly images alongside text. This is achieved by integrating specific functions within the `forward` and `generate` methods to accommodate visual data.
  
- **Custom Configuration and Model Classes**: The code defines custom classes (`LlavaConfig`, `LlavaLlamaModel`, and `LlavaLlamaForCausalLM`) extending from Hugging Face's Transformer library classes. These custom classes allow for the integration of LLaVA-specific configurations and functionalities.

### Key Functions:

- **`prepare_inputs_labels_for_multimodal`**: This function is crucial for processing multimodal inputs, including images. It adjusts the input data structure to ensure the model can handle both text and visual inputs effectively.

- **`generate` Method**: Enhanced to support image inputs during the generation process. This allows the model to consider visual context when producing text outputs.

- **`prepare_inputs_for_generation`**: Altered to incorporate images and their sizes into the model inputs, enabling the generation of responses that consider both textual and visual information.

### Efficiency and Flexibility:

- The adaptation retains a significant portion of the code from the original `llama.py` provided by Hugging Face (HF), focusing on the integration of multimodal data handling capabilities.

- Despite appearing complex, the modifications are centered around enabling LLaVA to process additional data types (i.e., images) alongside textual inputs, enhancing its applicability in multimodal scenarios.

### Deployment:

- **Efficient Deployment**: The final part of the code ensures that the custom LLaVA model, with its newly integrated components, can be seamlessly deployed and utilized within the larger AI ecosystem, supported by the registration of the custom config and model classes for easy reference and instantiation.

This guide not only facilitates the customization of LLaVA with different LLMs and visual encoders but also emphasizes the model's versatility in handling complex multimodal data, making it a potent tool for a wide range of applications.

### Other LLMs

The `language_model` directory includes additional files pivotal to expanding the versatility and capabilities of LLaVA through integration with other cutting-edge models:

- **`llava_mistral.py`**: This file introduces support for Mistral AI Models, incorporating their unique methodologies and strengths into the LLaVA framework.

- **`llava_mpt.py`**: This script integrates the MPT model, a notable contribution from the MosaicML team. Characterized by its deployment in various sizes and finetuned iterations, the MPT model stands out as an open-source, commercially viable series of LLMs pre-trained on an extensive dataset of 1 trillion tokens.

### CLIP Integration - `multimodal_encoder/builder.py` and `multimodal_encoder/clip_encoder.py`

The `multimodal_encoder/builder.py`script defines a function for constructing a vision tower component of a multimodal encoder, specifically targeting the use of a CLIP vision model. 

```python
import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
```

- **Function Definition**: `build_vision_tower` accepts a configuration object `vision_tower_cfg` and additional keyword arguments. It aims to instantiate a vision tower based on the provided configuration.

- **Vision Tower Selection**: It tries to identify the specific vision tower to use by looking for `mm_vision_tower` or `vision_tower` attributes within the given configuration. This flexibility allows for the integration of different vision encoders.

- **Path Validation and Model Instantiation**: The function checks if the specified vision tower path exists or if the name matches known model sources (such as "openai", "laion", or a specific "ShareGPT4V" model). If so, it proceeds to create an instance of `CLIPVisionTower` with the specified configuration and any additional arguments.

- **Error Handling**: If the vision tower is unrecognized, the function raises a `ValueError`, indicating an unsupported or unknown vision model configuration.

The `multimodal_encoder/clip_encoder.py` file defines the `CLIPVisionTower` class, extending PyTorch's `nn.Module`, to encapsulate the functionality of a CLIP vision model within the LLaVA framework:

```python
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2 
```

- **Class Initialization**: The constructor (`__init__`) initializes the model, possibly delaying its loading based on the `delay_load` flag. It also sets up configurations like the selected layer and feature type from the provided arguments.

- **Lazy Model Loading**: The `load_model` method loads the CLIP vision model and its configurations as needed, ensuring efficient resource use. It prevents reloading if the model is already in memory.

- **Feature Selection**: The `feature_select` method extracts specific features (either all patches except the CLS token or including the CLS token, based on the `select_feature` configuration) from the model's output, facilitating tailored downstream processing.

- **Forward Pass**: The `forward` method processes input images, either individually or in batches, through the CLIP vision model, extracting and returning the selected features. This method is decorated with `@torch.no_grad()` to disable gradient calculations, optimizing inference performance.

- **Utility Properties**: The class includes several properties for ease of access to the model's device, data type, configuration, and dimensions of the extracted features.

- **Integration with Transformers Library**: Import statements at the beginning show dependencies on the `transformers` library for accessing CLIP models and configurations, indicating tight integration with Hugging Face's ecosystem for transformer models.

In essence, `CLIPVisionTower` serves as a specialized wrapper around the CLIP vision model, enabling nuanced control over feature extraction and providing a bridge between CLIP's capabilities and LLaVA's multimodal encoding requirements.

### Projection Layers - `multimodal_projector/builder.py`

The `multimodal_projector/builder.py` script is designed to construct various types of projection layers that transform features from one space to another, commonly used in multimodal learning to align or adapt features from different modalities (e.g., vision to language or vice versa). This code defines several classes and a function to build these projection layers based on configuration details. 

```python
import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}') 
```

#### Classes Defined:

- **`IdentityMap`**: A simple PyTorch module that implements an identity mapping. It takes input and returns it unchanged. This could be useful when no transformation is required between feature spaces. The `.config` property specifies the projector type as 'identity', indicating its function.

- **`SimpleResBlock`**: This class implements a simple residual block with a projection layer. It starts with layer normalization (`nn.LayerNorm`), followed by a projection sequence that includes a linear transformation (`nn.Linear`), a GELU activation function (`nn.GELU()`), and another linear transformation. The forward pass applies layer normalization to the input, then adds the original input to its projected version, implementing a form of residual connection. This can enhance the model’s ability to learn and retain information through layers.

### Function Defined:

- **`build_vision_projector`**: This function dynamically constructs a vision projector based on the provided configuration. It supports different types of projectors:

    - **Linear**: A straightforward linear transformation from the multimodal hidden size to the desired hidden size, typically used for simple feature adaptation.
  
    - **MLP with GELU**: Recognizes projector types formatted as `'mlpXd_gelu'`, where `X` denotes the depth of the MLP (multilayer perceptron). It constructs an MLP of specified depth with GELU activation functions between linear layers. This format allows for specifying more complex, non-linear transformations.
  
    - **Identity**: Returns an `IdentityMap` instance when no transformation is desired.

- The function raises a `ValueError` if an unrecognized projector type is specified, ensuring that only supported configurations are used.

### Usage:

- This setup allows for flexible configuration of the projection layer in a multimodal system, enabling the tuning of feature transformation to the specific needs of the model and the task at hand.

- By providing options like simple linear transformations, more complex MLP structures, or even bypassing transformation altogether with an identity map, this code caters to a variety of modeling scenarios, enhancing the adaptability and efficiency of multimodal learning architectures.

## The Model Architecture - `/model`

### Main Package Initialization - `__init__.py`

The `__init__.py` file is used to initialize a Python package and makes it easier to manage imports across the package by defining which modules and names will be accessible when the package is imported elsewhere in a project.

```python
try:
    from .language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
    from .language_model.llava_mpt import LlavaMptForCausalLM, LlavaMptConfig
    from .language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralConfig
except:
    pass
```

- **`try` Block**: The code within the `try` block attempts to import specific classes from different modules within the `language_model` subpackage. These classes are implementations of various models and their configurations tailored for causal language modeling tasks. The classes include:
  - `LlavaLlamaForCausalLM` and `LlavaConfig`: These are the main class and configuration for a LLaVA model based on the LLaMA architecture.
  - `LlavaMptForCausalLM` and `LlavaMptConfig`: Represents the model and configuration classes for integrating the MosaicML team's MPT model within the LLaVA framework.
  - `LlavaMistralForCausalLM` and `LlavaMistralConfig`: Similar to the above, these are for integrating Mistral AI's models into LLaVA.

- **`except` Block**: This is a catch-all `except` statement that is executed if any error occurs in the `try` block. In this case, it simply passes, meaning it does nothing if an error occurs. This could be due to various reasons, such as the modules not being found because of an incomplete installation or development environment setup. Using such a broad `except` statement without specifying an exception type or logging the error is generally not recommended in production code due to the potential for silencing important errors. However, it might be used here for simplicity or during a development phase where the package structure is still being finalized.

#### Purpose and Implications:

- **Simplifying Imports**: By importing these classes in the `__init__.py` file, the package allows users to import these classes directly from the package without needing to know the internal module structure. For example, after this setup, one could import `LlavaLlamaForCausalLM` directly from `language_model` instead of having to navigate through the submodules.

- **Package Initialization**: This file turns the directory it's in into a Python package, enabling it to include multiple modules and making it easier to distribute and reuse code.

- **Error Handling Strategy**: The use of a bare `except: pass` statement is a placeholder. In a more developed or production-ready version of the code, it might be replaced with more specific error handling or removed altogether to ensure errors are appropriately logged or addressed.

This `__init__.py` setup facilitates modularity and ease of use for the package, allowing for flexible development and integration of different language models under the LLaVA framework.


### The LLaVA Architecture - `llava_arch.py`

The `llava_arch.py` script is a comprehensive Python module designed for the LLaVA framework, focusing on integrating multimodal capabilities into models, particularly for causal language modeling tasks. This module outlines the structure for incorporating visual data processing alongside textual data, emphasizing the adaptability and extensibility of models within the LLaVA ecosystem. 

```python
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None]
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
```

### Key Components:

- **Abstract Base Classes and Mixins**: The module utilizes Python's Abstract Base Class (ABC) and abstractmethod decorator to define base functionalities that multimodal and causal language models should implement, ensuring that any subclass provides specific implementations for those functionalities.

- **Model Initialization**: The `LlavaMetaModel` class, which does not directly inherit from `nn.Module` but is expected to be part of a model's inheritance chain, initializes multimodal components such as the vision tower and projector based on the configuration provided. This includes conditional initialization for image newline parameters when merging patches from visual data.

- **Vision Tower and Projector Builders**: Utilizes helper functions `build_vision_tower` and `build_vision_projector` from respective modules to dynamically create components for processing visual data according to the model configuration. This allows the model to adapt its architecture based on the needs of the task and the nature of the input data.

- **Image Feature Processing**: Defines methods to load and prepare the vision tower for processing images, including selecting specific layers and features for extraction and determining how image patches are merged or flattened.

- **Multimodal Input Preparation**: Implements methods for preparing inputs that combine textual and visual data, adjusting token embeddings for special image-related tokens, and merging visual features with textual embeddings. This is crucial for tasks where the model needs to consider both text and images to generate responses or make predictions.

### Detailed Functionalities:

- **Dynamic Vision Module Loading**: The script can dynamically load vision modules based on runtime configuration, including support for delayed loading to optimize resource usage.

- **Flexible Projector Configuration**: Supports configuring the vision-to-text projection layer to match the dimensions and architecture required for the specific model, allowing for linear, MLP, or custom implementations.

- **Image Processing and Feature Selection**: Provides mechanisms to process images, select relevant features, and integrate these with textual data for multimodal reasoning.

- **Conditional Token Embedding Adjustments**: The script makes adjustments to token embeddings based on the configuration, especially for models utilizing special tokens to denote image data or for models enhanced with LoRA weights.

- **Multimodal Input Handling**: Outlines a comprehensive approach to handling multimodal inputs, including combining text and image data, adjusting for special tokens, and ensuring the model can process inputs of varying lengths and formats.

The `llava_arch.py` module is instrumental in the LLaVA framework, offering a foundation for developing and deploying multimodal models capable of understanding and generating content based on both textual and visual inputs. It exemplifies the integration of advanced neural network techniques with practical application scenarios, highlighting the flexibility and power of the LLaVA framework for building sophisticated AI models.

## Local Demonstration of LLaVA

For detailed instructions on how to run LLaVA demos both online and locally, please refer to the repository. While Hugging Face Spaces offers an online demo, it comes with limited traffic capacity, making local execution a preferable alternative for uninterrupted exploration.

The demo can be executed across various platforms and environments; for comprehensive guidance, consult the repository documentation.

I'll assume you followed the instructions in the repository to clone the LLaVA repository and install the required dependencies. The following steps are based on the repository's documentation and my personal experience running the demo on my local machines.

My experiences running the demo on both macOS and Windows were successful. To initiate the demo locally, you need to set up the controller, the Gradio web server, and the model worker across separate terminal sessions:

```bash
# Launch the controller
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

Note that the Gradio web server requires access to the models to function:

```bash
# Start the Gradio web server
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

Upon the initial execution, the model worker will download the specified model:

```bash
# Deploy the model worker
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.6-vicuna-13b --device mps
```

In my case, I utilized the `liuhaotian/llava-v1.6-vicuna-13b` model on an Apple M2 Ultra with 192GB RAM, setting the device to `mps`.

The demonstration proceeded without a hitch on both macOS and Windows platforms. Nevertheless, the launch of LLaVA v1.6 has been met with some obstacles, notably memory leaks and various bugs that compromise its stability. For those in search of a steadier and more dependable version, turning to the previous iteration, v1.5, is recommended.

Attempting to operate v1.6 models on my M2 Ultra with 192GB led to the machine becoming unresponsive, ultimately requiring a hard reboot. This situation serves as a clear signal of memory leaks and performance challenges.

To guarantee a seamless demo experience, I opted for the v1.5 7B model:

```python
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b --device mps
```

And indeed, the performance was flawless.

This version demonstrated effective functionality across both platforms. The same issues observed with v1.6 similarly impacted its performance on Hugging Face Spaces, resulting in less than ideal operation. Switching back to v1.5 ensures not only a smoother execution but also an enhanced overall demonstration experience.

I remain optimistic that the LLaVA team will rectify these setbacks in forthcoming updates, thereby offering a more stable and robust platform for multimodal AI exploration.

## Final Thoughts

LLaVA stands out as an exceptional open-source multimodal AI framework, skillfully bridging the realms of language and vision to foster the creation of advanced multimodal AI applications. Its modular design, alongside thorough documentation and versatile support for diverse model architectures and configurations, renders LLaVA an invaluable tool for both researchers and developers venturing into the domain of multimodal AI. Moreover, LLaVA serves as an excellent educational resource, offering profound insights into the construction, integration, and theoretical underpinnings of multimodal models.

Ultimately, the journey through LLaVA is not just about leveraging a powerful tool; it's about enriching our learning experience in the ever-evolving field of artificial intelligence, particularly in the nuanced area of multimodal interactions.

Embarking on this journey has been yet another thrilling experience for me.
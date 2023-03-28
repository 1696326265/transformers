# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SparseGPT configuration"""
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging


logger = logging.get_logger(__name__)

SPARSEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sparsegpt-small": "https://huggingface.co/sparsegpt-small/resolve/main/config.json",
}



class SparseGPTConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`SparseGPTModel`] or a [`TFSparseGPTModel`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [sparsegpt](https://huggingface.co/sparsegpt) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`SparseGPTModel`] or [`TFSparseGPTModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*, defaults to None):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (`string`, *optional*, defaults to `"cls_index"`):
            Argument used when doing sequence summary, used in the models [`SparseGPTDoubleHeadsModel`] and
            [`TFSparseGPTDoubleHeadsModel`].

            Has to be one of the following options:

                - `"last"`: Take the last token hidden state (like XLNet).
                - `"first"`: Take the first token hidden state (like BERT).
                - `"mean"`: Take the mean of all tokens hidden states.
                - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - `"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`SparseGPTDoubleHeadsModel`] and
            [`TFSparseGPTDoubleHeadsModel`].

            Whether or not to add a projection after the vector extraction.
        summary_activation (`str`, *optional*):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            [`SparseGPTDoubleHeadsModel`].

            Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
            Argument used when doing sequence summary, used in the models [`SparseGPTDoubleHeadsModel`] and
            [`TFSparseGPTDoubleHeadsModel`].

            Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
        summary_first_dropout (`float`, *optional*, defaults to 0.1):
            Argument used when doing sequence summary, used in the models [`SparseGPTDoubleHeadsModel`] and
            [`TFSparseGPTDoubleHeadsModel`].

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import SparseGPTConfig, SparseGPTModel

    >>> # Initializing a SparseGPT configuration
    >>> configuration = SparseGPTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = SparseGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sparsegpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        model_type='st-deep-k2',
        n_layer = None,
        n_head = None,
        n_embd =  None,
        att_hidden = None,
        ffd_hidden = None,
        universal = False,
        # these options must be filled in externally
        vocab_size = 50257,
        block_size = 512,
        # dropout hyperparameters
        embd_pdrop = 0,
        resid_pdrop = 0,
        attn_pdrop = 0,
        moe_pdrop = 0,
        sample_topk = 0,
        gating_size = 256,
        n_att_experts = 32,
        k_att = 2,
        n_mlp_experts = 32,
        k_mlp = 2,
        moe_type = 'moe',
        world_size = None,
        local_size = 1,
        att_func = 'stickbreaking',
        history_length = 512,
        aux_loss_type = 'mi',
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.model_type = model_type
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd =  n_embd
        self.att_hidden = att_hidden
        self.ffd_hidden = ffd_hidden
        self.universal = universal
        # these options must be filled in externally
        self.vocab_size = vocab_size
        self.block_size = block_size
        # dropout hyperparameters
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.moe_pdrop = moe_pdrop
        self.sample_topk = sample_topk
        self.gating_size = gating_size
        self.n_att_experts = n_att_experts
        self.k_att = k_att
        self.n_mlp_experts = n_mlp_experts
        self.k_mlp = k_mlp
        self.moe_type = moe_type
        self.world_size = world_size
        self.local_size = local_size
        self.att_func = att_func
        self.history_length = history_length
        self.aux_loss_type = aux_loss_type
        
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        additional_dict = {
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
                # new model
                'vt':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ),  
                'vt-large':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=2048, ffd_hidden=8092, moe_type='moe'
                    ), 
                'vt-deep':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ), 
                'vt-350m':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=1, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ), 
                'vt-350m-moe':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=1, n_mlp_experts=6, k_att=1, k_mlp=1, 
                    att_hidden=1024, ffd_hidden=4096, moe_type='moe'
                    ), 
                'st-4e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=4, n_mlp_experts=4, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'st':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=12, n_mlp_experts=12, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'st-16e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=16, n_mlp_experts=16, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-deep-16e':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=16, n_mlp_experts=16, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-deep-k2':         dict(
                    n_layer=24, n_head=8, n_embd=1024, universal=False, 
                    n_att_experts=32, n_mlp_experts=32, k_att=2, k_mlp=2, 
                    att_hidden=512, ffd_hidden=2048
                    ),
                'st-deep-wide':         dict(
                    n_layer=24, n_head=16, n_embd=2048, universal=False, 
                    n_att_experts=8, n_mlp_experts=32, 
                    att_hidden=1024, ffd_hidden=1024
                    ),
                'st-deep':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=6, n_mlp_experts=6, 
                    att_hidden=1024, ffd_hidden=2048
                    ), 
                'st-wide':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=False, 
                    n_att_experts=12, n_mlp_experts=12, 
                    att_hidden=1024, ffd_hidden=4096
                    ),
                'st-18e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=18, n_mlp_experts=18,
                    att_hidden=1024, ffd_hidden=4096
                    ),   
                'st-24e':         dict(
                    n_layer=12, n_head=16, n_embd=1024, universal=False, 
                    n_att_experts=24, n_mlp_experts=24, 
                    att_hidden=1024, ffd_hidden=4096
                    ), 
                'sut':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=30, n_mlp_experts=30, 
                    att_hidden=1024, ffd_hidden=4096),  
                'sut-12-36':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=12, n_mlp_experts=36, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-32':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=32, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-48':         dict(
                    n_layer=24, n_head=16, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=48, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-24-48-1head':         dict(
                    n_layer=24, n_head=1, n_embd=1024, universal=True, 
                    n_att_experts=24, n_mlp_experts=48, 
                    att_hidden=1024, ffd_hidden=4096), 
                'sut-wide':         dict(
                    n_layer=12, n_head=16, n_embd=2048, universal=True, 
                    n_att_experts=24, n_mlp_experts=24, 
                    att_hidden=1024, ffd_hidden=2048),
            }[self.model_type]
        
        self.__dict__.update(additional_dict)
        
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class SparseGPTOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13

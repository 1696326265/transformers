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
"""PyTorch SparseGPT model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_sparsegpt import SparseGPTConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "sparsegpt-small"
_CONFIG_FOR_DOC = "SparseGPTConfig"

SPARSEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sparsegpt-small",
    # See all SparseGPT models at https://huggingface.co/models?filter=sparsegpt
]



# Copied from transformers.models.gpt2.modeling_gpt2.load_tf_weights_in_gpt2 with gpt2->sparsegpt
def load_tf_weights_in_sparsegpt(model, config, sparsegpt_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    try:
        import re

        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(sparsegpt_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model




from .submodules.parallel_linear.src import moe

@torch.jit.script
def NewGELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@torch.jit.script
def stickbreaking(logits: torch.Tensor, mask: torch.Tensor, cum_weight: torch.Tensor) -> torch.Tensor:
    """
    Stick-breaking attention weights.
    """
    mask = (mask[None, :, :, None, None] == 0).expand_as(logits)
    log_z = F.logsigmoid(logits).masked_fill(mask, float('-inf'))
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0)
    re_cum_log_beta = torch.einsum('bijnh,jk->biknh', log_beta, cum_weight)
    log_p = log_z + re_cum_log_beta
    return log_p.exp()

@torch.jit.script
def stickbreaking_att(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, cum_weight: torch.Tensor) -> torch.Tensor:
    """
    Stick-breaking attention weights.
    """
    logits = torch.einsum('bikhd,bjhd->bkhij', q, k) / math.sqrt(k.size(-1))
    mask = (mask[None, None, None, :, :] == 0).expand_as(logits)
    z = F.sigmoid(logits).masked_fill(mask, 0)
    log_beta = F.logsigmoid(-logits).masked_fill(mask, 0)
    re_cum_log_beta = torch.einsum('bnhij,jk->bnhik', log_beta, cum_weight)
    att = z * re_cum_log_beta.exp()
    y = torch.einsum('bkhij,bjhd->bikhd', att, v) 
    return y

class SparseCausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.att_hidden % config.n_head == 0
        if True:
            self.q_proj = moe.MoE(
                input_size=config.n_embd, 
                head_size=config.att_hidden, 
                num_experts=config.n_att_experts, 
                top_k=config.k_att,
                acc_aux_loss=config.universal, 
                bias=False,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
            )
        if config.att_hidden == config.n_embd and config.n_head == 1:
            self.k_proj = nn.Identity()
            self.v_proj = nn.Identity()
        else:
            self.k_proj = nn.Linear(config.n_embd, config.att_hidden)
            self.v_proj = nn.Linear(config.n_embd, config.att_hidden)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.history_length = config.history_length

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size + config.history_length, config.block_size  + config.history_length, dtype=torch.int8))
        )
        self.register_buffer(
            "cum_weight", 
            torch.tril(torch.ones(config.block_size + config.history_length, config.block_size  + config.history_length), -1)
        )
        self.n_head = config.n_head
        self.top_k = config.k_att
        self.n_embd = config.n_embd
        self.att_hidden = config.att_hidden
        self.head_size = config.att_hidden // config.n_head

        self.att_func = config.att_func

    def add_history(self, k, v, hidden):
        if hidden is None:
            new_k = k
            new_v = v
        else:
            k_history, v_history = hidden
            new_k = torch.cat([k_history, k], dim=1)
            new_v = torch.cat([v_history, v], dim=1)
        k_history = new_k[:, -self.history_length:].detach()
        v_history = new_v[:, -self.history_length:].detach()

        return new_k, new_v, (k_history, v_history)

    def forward(self, x, hidden):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, aux_loss = self.q_proj.map(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        k, v, hidden = self.add_history(k, v, hidden)
        context_length = k.size(1)
        
        q = q.view(B, T, self.top_k, self.n_head, self.head_size) # (B, T, k, nh, hs)
        k = k.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)
        v = v.view(B, context_length, self.n_head, self.head_size) # (B, T, nh, hs)

        mask = self.mask[context_length - T:context_length, :context_length]

        y = stickbreaking_att(q, k, v, mask=mask, cum_weight=self.cum_weight[:context_length, :context_length])

        # output projection
        y = self.q_proj.reduce(y.reshape(B, T, self.top_k, self.att_hidden).type_as(x))

        y = y.view(B, T, C) # re-assemble all head outputs side by side
        return y, aux_loss, hidden

class SparseGPTBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SparseCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        if True:
            self.mlpf = moe.MoE(
                input_size=config.n_embd, 
                head_size=config.ffd_hidden, 
                num_experts=config.n_mlp_experts, 
                top_k=config.k_mlp, 
                bias=False, 
                activation=NewGELU,
                acc_aux_loss=config.universal,
                gating_dropout=config.moe_pdrop,
                sample_topk=config.sample_topk,
                gating_size=config.gating_size,
                aux_loss=config.aux_loss_type,
            )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def get_aux_loss_and_clear(self):
        return self.attn.q_proj.get_aux_loss_and_clear() + self.mlpf.get_aux_loss_and_clear()

    def mixed_residual(self, x, y):
        x0, x1 = x.chunk(2, dim=-1)
        y0, y1 = y.chunk(2, dim=-1)

        return torch.cat([x0 + y0, y1], dim=-1)

    def forward(self, x, hidden=None):
        x_att, att_loss, hidden = self.attn(self.ln_1(x), hidden)
        x = x + self.resid_dropout(x_att)
        # x = self.mixed_residual(x, x_att)
        x_mlp, mlp_loss = self.mlpf(self.ln_2(x))
        x = x + self.resid_dropout(x_mlp)
        # x = self.mixed_residual(x, x_mlp)
        return x, att_loss + mlp_loss, hidden



# Copied from transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel with GPT2->SparseGPT,gpt2->sparsegpt,OpenAI GPT-2->SparseGPT
class SparseGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SparseGPTConfig
    load_tf_weights = None
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["SparseGPTBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, SparseGPTModel):
            module.gradient_checkpointing = value


SPARSEGPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SparseGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
SPARSEGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the sparsegpt models have the
            following number of attention modules:

                - sparsegpt: 12
                - sparsegpt-medium: 24
                - sparsegpt-large: 36
                - sparsegpt-xl: 48

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using sparsegpt-xl, which has a total of 48 attention modules:
    model = SparseGPTLMHeadModel.from_pretrained("sparsegpt-xl")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        1: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
        3: [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with sparsegpt-large:
    model = SparseGPTLMHeadModel.from_pretrained("sparsegpt-large")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7],
        1: [8, 9, 10, 11, 12, 13, 14, 15],
        2: [16, 17, 18, 19, 20, 21, 22, 23],
        3: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


@add_start_docstrings(
    "The bare SPARSEGPT Model transformer outputting raw hidden-states without any specific head on top.",
    SPARSEGPT_START_DOCSTRING,
)
class SparseGPTModel(SparseGPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = []

    def __init__(self, config):
        super().__init__(config)
        self.block_size = config.block_size
        self.universal = config.universal
        
        if self.universal:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd) if config.att_func == 'softmax' else None,
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([SparseGPTBlock(config)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd) if config.att_func == 'softmax' else None,
                drop = nn.Dropout(config.embd_pdrop),
                h = nn.ModuleList([SparseGPTBlock(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.n_layer = config.n_layer

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layer):
            hidden.append(None)
        return hidden
    
    @add_start_docstrings_to_model_forward(SPARSEGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self, idx, targets=None, hidden=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
            tok_emb = tok_emb + pos_emb
        x = self.transformer.drop(tok_emb)
        
        if hidden is None:
            hidden = self.init_hidden()
        new_hidden = []
        if self.universal:
            for i in range(self.n_layer):
                x, _, hidden_i = self.transformer.h[0](x, hidden[i])
                new_hidden.append(hidden_i)
            aux_loss = self.transformer.h[0].get_aux_loss_and_clear()
        else:
            aux_loss = 0
            for block, hidden in zip(self.transformer.h, hidden):
                x, aux_loss_i, hidden_i = block(x, hidden)
                aux_loss = aux_loss + aux_loss_i
                new_hidden.append(hidden_i)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = 0
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss, aux_loss, new_hidden

    
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        
        assert 0
        
        # warnings.warn(
        #     "`SparseGPTModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
        #     " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
        #     " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
        #     " ...}",
        #     FutureWarning,
        # )
        # self.device_map = (
        #     get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        # )
        # assert_device_map(self.device_map, len(self.h))
        # self.model_parallel = True
        # self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        # self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # self.wte = self.wte.to(self.first_device)
        # self.wpe = self.wpe.to(self.first_device)
        # # Load onto devices
        # for k, v in self.device_map.items():
        #     for block in v:
        #         cuda_device = "cuda:" + str(k)
        #         self.h[block] = self.h[block].to(cuda_device)
        # # ln_f to last
        # self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        
        assert 0
        
        # warnings.warn(
        #     "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
        #     FutureWarning,
        # )
        # self.model_parallel = False
        # self.device_map = None
        # self.first_device = "cpu"
        # self.last_device = "cpu"
        # self.wte = self.wte.to("cpu")
        # self.wpe = self.wpe.to("cpu")
        # for index in range(len(self.h)):
        #     self.h[index] = self.h[index].to("cpu")
        # self.ln_f = self.ln_f.to("cpu")
        # torch.cuda.empty_cache()

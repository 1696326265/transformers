# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parallel_experts import ParallelExperts
from .gate import top_k_gating, compute_gating


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self, 
        input_size, 
        head_size, 
        num_experts, 
        top_k,
        miloss=0,
        bias=False, 
        activation=None, 
        acc_aux_loss=False,
        hidden_size=None,
        gating_dropout=0.0,
        ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = nn.Linear(input_size, head_size, bias)
        self.output_experts = nn.Linear(head_size, input_size, bias)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

    def extra_repr(self):
        return 'k={}'.format(
            self.top_k)

    def get_aux_loss_and_clear(self):
        return torch.zeros(1).to(self.experts.weight.device)

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        h = self.experts(x)
        h = self.activation(h)
        y = self.output_experts(h)
        return y, torch.zeros(1).to(x.device)

    def map(self, x, skip_mask=None, sample_topk=0, return_indices=False):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        y = self.experts(x)
        return y, torch.zeros(1).to(x.device)

    def reduce(self, x, multiply_by_gates=True):
        y = self.output_experts(x)
        return y

    def init_experts(self):
        pass

    def sync_experts(self):
        pass


# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from fmoe import FMoE, FMoELinear
from fmoe.layers import _fmoe_general_global_forward
from fmoe.layers import *
from fmoe.transformer import _Expert

from .gate import top_k_gating


class MoE(FMoE):

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
        expert_dp_comm="none",
        world_size=None,
        **kwargs
        ):
        if world_size is None:
            world_size = torch.distributed.get_world_size()
        assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
        local_num_experts = num_experts // world_size

        super(MoE, self).__init__(
            num_expert=local_num_experts, 
            d_model=input_size, 
            world_size=world_size,
            top_k=top_k, 
            **kwargs
        )

        self.input_size = input_size
        self.head_size = head_size
        self.activation = activation

        self.input_experts = FMoELinear(local_num_experts, input_size, head_size, bias=bias)
        self.output_experts = FMoELinear(local_num_experts, head_size, input_size, bias=bias)

        self.gate = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            miloss=miloss, 
            acc_aux_loss=acc_aux_loss, 
            )

        self.mark_parallel_comm(expert_dp_comm)

    def experts(self, inp, fwd_expert_count):
        x = self.input_experts(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.output_experts(x, fwd_expert_count)
        return x

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.input_experts is not None:
            comm = expert_dp_comm
            if isinstance(self.input_experts, list):
                for e in self.input_experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.input_experts, comm)
        if self.output_experts is not None:
            comm = expert_dp_comm
            if isinstance(self.output_experts, list):
                for e in self.output_experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.output_experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def get_aux_loss_and_clear(self):
        return self.gate.get_aux_loss_and_clear()

    def pre_process(self, moe_inp):
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)
        return moe_inp

    def compute_gate(self, moe_inp, skip_mask):
        gate_top_k_idx, gate_score, _ = self.gate(moe_inp, skip_mask)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)
        return gate_top_k_idx, gate_score

    def run_experts(self, experts, moe_inp, gate_top_k_idx, skip_mask=None):
        # delete masked tensors
        if skip_mask is not None:
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[skip_mask > 0, :]
                return tensor

            skip_mask = skip_mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[skip_mask > 0, :]

        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, experts,
            self.num_expert, self.world_size,
        )

        # recover deleted tensors
        if skip_mask is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    skip_mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[skip_mask > 0] = tensor
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:

            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)
        return moe_outp

    def multi_gates(self, moe_outp, gate_score):
        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)
        return moe_outp

    def post_process(self, moe_outp):
        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        return moe_outp

    def forward(self, moe_inp, skip_mask=None):
        original_shape = moe_inp.shape
        moe_inp = moe_inp.reshape(-1, self.d_model)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        moe_inp = self.pre_process(moe_inp)
        
        gate_top_k_idx, gate_score = self.compute_gate(moe_inp, skip_mask)

        moe_outp = self.run_experts(self.experts, moe_inp, gate_top_k_idx, skip_mask)

        moe_outp = self.multi_gates(moe_outp, gate_score)

        moe_outp = self.post_process(moe_outp)
        return moe_outp.reshape(original_shape), self.gate.loss

    def map(self, moe_inp, skip_mask=None):
        if skip_mask is not None:
            assert moe_inp.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
        bsz, length, emb_size = moe_inp.size()
        moe_inp = moe_inp.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        moe_inp = self.pre_process(moe_inp)
        
        gate_top_k_idx, gate_score = self.compute_gate(moe_inp, skip_mask)
        self.cached_gates = (gate_top_k_idx, gate_score)

        moe_outp = self.run_experts(self.input_experts, moe_inp, gate_top_k_idx, skip_mask)

        moe_outp = self.post_process(moe_outp)
        return moe_outp.reshape(bsz, length, self.top_k, -1), self.gate.loss

    def reduce(self, moe_inp, skip_mask=None):
        bsz, length, k, emb_size = moe_inp.size()
        moe_inp = moe_inp.view(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        moe_inp = self.pre_process(moe_inp)
        
        gate_top_k_idx, gate_score = self.cached_gates
        gate_top_k_idx = gate_top_k_idx.view(-1, 1)

        moe_outp = self.run_experts(self.output_experts, moe_inp, gate_top_k_idx, skip_mask)
        moe_outp = moe_outp.view(-1, k, self.input_size)

        moe_outp = self.multi_gates(moe_outp, gate_score)

        moe_outp = self.post_process(moe_outp)
        return moe_outp.reshape(bsz, length, self.input_size)

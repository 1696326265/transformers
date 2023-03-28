# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import os
from contextlib import contextmanager
from math import prod

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .gate import top_k_gating, compute_gating
from .utils import all_to_all
from .distributed_parallel_experts import DistParallelLinear


def moe_comm_hook(state, bucket):
    """Synchronize gradients across data parallel workers."""
    # allreduce grads
    world_size = dist.get_world_size()
    fut_list = []
    buffer = bucket.buffer()
    size_list = [tensor.numel() for tensor in bucket.parameters()]
    buffer_list = buffer.split(size_list)
    for grad, tensor in zip(buffer_list, bucket.parameters()):
        if hasattr(tensor, 'param_group'):
            grad.div_(tensor.param_group.size())
            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=tensor.param_group, async_op=True).get_future()
        else:
            grad.div_(world_size)
            handle = dist.all_reduce(grad, op=dist.ReduceOp.SUM, async_op=True).get_future()
        fut_list.append(handle)
    fut = torch.futures.collect_all(fut_list)

    def concat(fut):
        fut_list = fut.wait()
        return bucket.buffer()
    return fut.then(concat)


def init_groups(world_size, local_size, rank):
    """Initialize shared local and param groups for MoE."""
    for i in range(world_size // local_size):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            local_group = group
    for i in range(local_size):
        ranks = list(range(i, world_size, local_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            param_group = group

    return local_group, param_group


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
        activation=nn.ReLU(),
        acc_aux_loss=False, 
        local_size=None,
        gating_dropout=0.0,
        sample_topk=0,
        gating_size=256,
        aux_loss='mi',
        local_group=None,
        param_group=None
        ):
        super(MoE, self).__init__()

        self.top_k = top_k
        self.input_size = input_size
        self.head_size = head_size
        self.activation = activation

        self.world_size = dist.get_world_size()
        if local_size is None:
            self.local_size = torch.cuda.device_count()
        else:
            self.local_size = local_size
        # self.device_id = int(os.environ["LOCAL_RANK"])
        self.rank = dist.get_rank()
        self.local_rank = self.rank % self.local_size
        
        assert local_group is not None and param_group is not None, 'local_group and param_group need to be provided.'
        self.local_group = local_group
        self.param_group = param_group

        self.total_num_experts = num_experts
        assert self.total_num_experts % self.local_size == 0, \
                (self.total_num_experts,  self.local_size)
        self.local_num_experts = self.total_num_experts // self.local_size

        self.input_weight = nn.Parameter(torch.Tensor(self.local_num_experts, input_size, head_size))
        self.output_weight = nn.Parameter(torch.Tensor(self.local_num_experts, head_size, input_size))

        self.gate = top_k_gating(
            input_size=input_size, 
            num_experts=num_experts, 
            top_k=top_k, 
            acc_aux_loss=acc_aux_loss, 
            dropout=gating_dropout,
            sample_topk=sample_topk,
            hidden_size=gating_size,
            aux_loss=aux_loss
            )

        def backhook(grad):
            grad.div_(self.local_size)
            return grad

        self.input_weight.register_hook(backhook)
        self.output_weight.register_hook(backhook)

        self.input_weight.param_group = self.param_group
        self.output_weight.param_group = self.param_group

        self.init_experts()

    def init_experts(self):
        nn.init.uniform_(self.input_weight, -1. / self.input_size, 1. / self.input_size)
        nn.init.uniform_(self.output_weight, -1. / self.input_size, 1. / self.input_size)

    def sync_experts(self):
        # print('syncing experts, source:', self.local_rank)
        weight = self.input_weight.data.contiguous()
        dist.broadcast(weight, src=self.local_rank, group=self.param_group)
        self.input_weight.data = weight

        weight = self.output_weight.data.contiguous()
        dist.broadcast(weight, src=self.local_rank, group=self.param_group)
        self.output_weight.data = weight

    def get_aux_loss_and_clear(self):
        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, moe_inp, skip_mask=None):
        top_k_indices, top_k_gates, probs = self.gate(moe_inp, skip_mask=skip_mask)
        self.batch_gates, self.batch_index, self.expert_size, self.gates, self.index_sorted_experts =\
            compute_gating(self.top_k, probs, top_k_gates, top_k_indices)

    def run_experts_merged(self, expert_inputs, expert_mode="all"):
        all_expert_size = [torch.zeros_like(self.expert_size) for _ in range(self.local_size)]
        dist.all_gather(all_expert_size, self.expert_size, group=self.local_group)
        all_expert_size = torch.stack(all_expert_size, dim=0) # source gpus, global experts
        all_expert_size = all_expert_size.view(self.local_size, self.local_size, self.local_num_experts) 
        # source gpus, target gpus, local experts

        partitioned_size = all_expert_size.sum(dim=2)
        source_size = partitioned_size[self.local_rank]
        target_size = partitioned_size[:, self.local_rank]
        gathered_inputs = all_to_all.apply(expert_inputs, source_size, target_size, self.local_group)

        input_sizes = all_expert_size[:, self.local_rank].flatten().tolist()
        input_list = gathered_inputs.split(input_sizes, dim=0)

        output_list = [None] * len(input_list)
        for i in range(self.local_num_experts):
            input_i_list = input_list[i::self.local_num_experts]
            input_i = torch.cat(input_i_list, dim=0)
            if expert_mode == "all":
                h = torch.mm(input_i, self.input_weight[i])
                h = self.activation(h)
                h = torch.mm(h, self.output_weight[i])
            elif expert_mode == "map":
                h = torch.mm(input_i, self.input_weight[i])
            elif expert_mode == "reduce":
                h = torch.mm(input_i, self.output_weight[i])
            h_list = h.split(input_sizes[i::self.local_num_experts], dim=0)
            output_list[i::self.local_num_experts] = h_list

        expert_outputs = torch.cat(output_list, dim=0)
        gathered_outputs = all_to_all.apply(expert_outputs, target_size, source_size, self.local_group)
        return gathered_outputs

    def run_experts_per_expert(self, expert_inputs, expert_mode="all"):
        input_size = self.expert_size.view(self.local_num_experts, self.local_size)
        size_list = input_size.sum(dim=1).tolist()
        input_list = expert_inputs.split(size_list, dim=0)

        all_target_size = [torch.zeros_like(self.expert_size) for _ in range(self.local_size)]
        dist.all_gather(all_target_size, self.expert_size, group=self.local_group)
        all_target_size = torch.stack(all_target_size, dim=1)
        target_size = all_target_size[self.local_rank::self.local_size]

        output_list = [None] * self.local_num_experts
        for i in range(self.local_num_experts):
            # gather inputs
            gathered_inputs = all_to_all.apply(input_list[i], input_size[i], target_size[i], self.local_group)

            # compute expert i
            if expert_mode == "all":
                h = torch.mm(gathered_inputs, self.input_weight[i])
                h = self.activation(h)
                h = torch.mm(h, self.output_weight[i])
            elif expert_mode == "map":
                h = torch.mm(gathered_inputs, self.input_weight[i])
            elif expert_mode == "reduce":
                h = torch.mm(gathered_inputs, self.output_weight[i])

            # return outputs
            output_i = all_to_all.apply(h, target_size[i], input_size[i], self.local_group)
            output_list[i] = output_i

        # torch.cuda.synchronize()
        expert_outputs = torch.cat(output_list, dim=0)
        return expert_outputs

    def forward(self, moe_inp, skip_mask=None):
        original_shape = moe_inp.shape
        moe_inp = moe_inp.reshape(-1, self.input_size)
        input_shape = moe_inp.shape
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        self.compute_gate(moe_inp, skip_mask)

        expert_inputs = moe_inp[self.batch_index]
        expert_outputs = self.run_experts_merged(expert_inputs, expert_mode="all")
        # expert_outputs = DistParallelLinear.apply(expert_inputs, self.expert_size, self.input_weight, self.local_rank, self.local_group, self.output_weight)

        expert_outputs = expert_outputs * self.batch_gates[:, None]
        moe_outp = torch.zeros(input_shape, device=expert_outputs.device, dtype=expert_outputs.dtype)
        moe_outp.index_add_(0, self.batch_index, expert_outputs)

        return moe_outp.reshape(original_shape), self.gate.loss

    # def forward(self, moe_inp, skip_mask=None):
    #     y, loss = self.map(moe_inp, skip_mask)
    #     y = self.activation(y)
    #     y = self.reduce(y)
    #     return y, loss

    def map(self, moe_inp, skip_mask=None):
        if skip_mask is not None:
            assert moe_inp.size()[:-1] == skip_mask.size(), \
                    "Skip mask should be same shape as `x`"
        bsz, length, emb_size = moe_inp.size()
        assert emb_size == self.input_size
        moe_inp = moe_inp.reshape(-1, self.input_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        self.compute_gate(moe_inp, skip_mask)

        expert_inputs = moe_inp[self.batch_index]
        expert_outputs = self.run_experts_merged(expert_inputs, expert_mode="map")
        # expert_outputs = DistParallelLinear.apply(expert_inputs, self.expert_size, self.input_weight, self.local_rank, self.local_group)

        moe_outp = torch.zeros((bsz * length * self.top_k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        moe_outp.index_add_(0, self.index_sorted_experts, expert_outputs)

        return moe_outp.reshape(bsz, length, self.top_k, -1), self.gate.loss

    def reduce(self, moe_inp, skip_mask=None):
        bsz, length, k, emb_size = moe_inp.size()
        moe_inp = moe_inp.reshape(-1, emb_size)

        expert_inputs = moe_inp[self.index_sorted_experts]
        expert_outputs = self.run_experts_merged(expert_inputs, expert_mode="reduce")
        # expert_outputs = DistParallelLinear.apply(expert_inputs, self.expert_size, self.output_weight, self.local_rank, self.local_group)

        expert_outputs = expert_outputs * self.batch_gates[:, None]
        moe_outp = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        moe_outp.index_add_(0, self.batch_index, expert_outputs)
        return moe_outp.reshape(bsz, length, self.input_size)


class DistributedExpertDataParallel(DistributedDataParallel):

    def __init__(self, module, *args, **kwargs):
        super().__init__(module, *args, **kwargs)

        self.register_comm_hook(state=None, hook=moe_comm_hook)
        self.init_experts()

    def init_experts(self):
        for m in self.module.modules():
            if isinstance(m, MoE):
                m.init_experts()
                m.sync_experts()

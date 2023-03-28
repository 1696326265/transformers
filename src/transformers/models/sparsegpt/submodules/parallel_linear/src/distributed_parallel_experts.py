import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
from typing import Any, Dict, List, Optional
from torch import Tensor
from torch import distributed as dist


@torch.jit.script
def NewGELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def sech(x):
    return 1/torch.cosh(x)

@torch.jit.script
def NewGELU_backward(x):
    return (torch.tanh((torch.sqrt(2)*(0.044715*torch.pow(x, 3)+x))/torch.sqrt(math.pi))+1)/2+(x*((26829*torch.pow(x, 2))/200000+1)* torch.pow(sech((torch.sqrt(2)*((8943*torch.pow(x, 3))/200000+x))/torch.sqrt(math.pi)), 2))/(torch.sqrt(2)*torch.sqrt(math.pi))


def all_to_all(input, input_size, target_size, group=None, output_buf=None):
    input_list = list(torch.split(input, input_size.tolist(), dim=0))

    if output_buf is None:
        output_buf = torch.empty(
            (target_size.sum(), input.size(1)), 
            device=input.device, dtype=input.dtype
        )
    gathered_inputs = list(torch.split(output_buf, target_size.tolist(), dim=0))

    handle = dist.all_to_all(gathered_inputs, input_list, group=group, async_op=True).get_future()

    def concat(fut):
        return output_buf
    return handle.then(concat)


class DistParallelLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx, input, expert_size, weight, local_rank, 
        local_group=None, output_weight=None
    ):
        local_num_experts = weight.size(0)
        local_size = expert_size.size(0) // local_num_experts

        input_size = expert_size.view(local_num_experts, local_size)
        size_list = input_size.sum(dim=1).tolist()
        input_list = input.split(size_list, dim=0)

        all_target_size = [torch.zeros_like(expert_size) for _ in range(local_size)]
        dist.all_gather(all_target_size, expert_size, group=local_group)
        all_target_size = torch.stack(all_target_size, dim=1)
        target_size = all_target_size[local_rank::local_size]

        output_dim = weight.size(2)
        if output_weight is not None:
            output_dim = output_weight.size(2)
        output_buf = torch.empty(
            (input.size(0), output_dim),
            device=input.device, dtype=input.dtype
        )
        output_buf_list = output_buf.split(size_list, dim=0)
        recv_handle_list = []
        for i in range(local_num_experts):
            # gather inputs
            send_handle = all_to_all(input_list[i], input_size[i], target_size[i], local_group)

            def compute_expert(fut):
                input_i = fut.wait()
                # compute expert i
                output = torch.mm(input_i, weight[i])
                if output_weight is not None:
                    hidden = output
                    h = NewGELU(output)
                    output = torch.mm(h, output_weight[i])
                else:
                    hidden = None
                return output, hidden, input_i

            expert_handle = send_handle.then(compute_expert)

            def recv(fut):
                output_i, hidden_i, input_i = fut.wait()

                recv_handle = all_to_all(output_i, target_size[i], input_size[i], local_group, output_buf_list[i])
                return recv_handle, hidden_i, input_i

            recv_handle = expert_handle.then(recv)
            recv_handle_list.append(recv_handle)


        torch.futures.wait_all(recv_handle_list)

        expert_input_list = []
        hidden_list = []
        for i in range(local_num_experts):
            recv_handle, hidden_i, input_i = recv_handle_list[i].wait()
            recv_handle.wait()
            hidden_list.append(hidden_i)
            expert_input_list.append(input_i)

        if output_weight is not None:
            hidden_list = [output_weight] + hidden_list
        else:
            hidden_list = []

        output = output_buf
        ctx.save_for_backward(expert_size, target_size, weight, *expert_input_list, *hidden_list)
        ctx.local_group = local_group
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        expert_size, target_size, weight = ctx.saved_tensors[:3]
        input_list = ctx.saved_tensors[3:3+target_size.size(0)]
        if len(ctx.saved_tensors) > 3+target_size.size(0):
            hidden_list = ctx.saved_tensors[3+target_size.size(0):]
            output_weight = hidden_list[0]
            hidden_list = hidden_list[1:]

            d_output_weight_buf = torch.empty_like(output_weight)
        else:
            d_output_weight_buf = None

        local_group = ctx.local_group

        local_num_experts = weight.size(0)
        local_size = expert_size.size(0) // local_num_experts

        input_size = expert_size.view(local_num_experts, local_size)
        size_list = input_size.sum(dim=1).tolist()
        grad_list = grad_out.split(size_list, dim=0)

        d_input_buf = torch.empty(
            (expert_size.sum(), input_list[0].size(1)),
            device=input_list[0].device, dtype=input_list[0].dtype)
        d_input_buf_list = d_input_buf.split(size_list, dim=0)
        d_weight_buf = torch.empty_like(weight)

        recv_handle_list = []
        for i in range(local_num_experts):
            send_grad_handle = all_to_all(grad_list[i], input_size[i], target_size[i], local_group)

            def compute_grad(fut):
                gathered_grads = fut.wait()
                if d_output_weight_buf is not None:
                    hidden_grads = torch.mm(gathered_grads, output_weight[i].t())
                    h = NewGELU(hidden_list[i])
                    torch.mm(h.t(), gathered_grads, out=d_output_weight_buf[i])
                    gathered_grads = NewGELU_backward(hidden_list[i]) * hidden_grads

                grad_i = torch.mm(gathered_grads, weight[i].t())
                torch.mm(input_list[i].t(), gathered_grads, out=d_weight_buf[i])
                return grad_i
            
            grad_handle = send_grad_handle.then(compute_grad)

            def recv(fut):
                grad_i = fut.wait()
                recv_handle = all_to_all(grad_i, target_size[i], input_size[i], local_group, d_input_buf_list[i])
                return recv_handle

            recv_handle = grad_handle.then(recv)
            recv_handle_list.append(recv_handle)

        torch.futures.wait_all(recv_handle_list)
        for recv_handle in recv_handle_list:
            recv_handle.wait().wait()

        d_input = d_input_buf
        d_weight = d_weight_buf

        return d_input, None, d_weight, None, None, d_output_weight_buf

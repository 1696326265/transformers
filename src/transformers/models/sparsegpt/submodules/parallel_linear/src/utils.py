import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.distributed as dist

class all_to_all(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, input_size, target_size, group=None):
        input_list = list(torch.split(input, input_size.tolist(), dim=0))

        output_buf = torch.empty(
            (target_size.sum(), input.size(1)), 
            device=input.device, dtype=input.dtype
        )
        gathered_inputs = list(torch.split(output_buf, target_size.tolist(), dim=0))

        dist.all_to_all(gathered_inputs, input_list, group=group)

        ctx.save_for_backward(input_size, target_size)
        ctx.group = group
        return output_buf
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        input_size, target_size = ctx.saved_tensors
        group = ctx.group

        grad_out_list = list(torch.split(grad_out, target_size.tolist(), dim=0))

        # gather grads
        output_buf = torch.empty(
            (input_size.sum(), grad_out.size(1)), 
            device=grad_out.device, dtype=grad_out.dtype
        )
        gathered_grads = list(torch.split(output_buf, input_size.tolist(), dim=0))

        dist.all_to_all(gathered_grads, grad_out_list, group=group)
        
        return output_buf, None, None, None


class all_reduce(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, group=None):
        output = input.clone()
        dist.all_reduce(output, group=group)
        return output
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        return grad_out, None
        
        
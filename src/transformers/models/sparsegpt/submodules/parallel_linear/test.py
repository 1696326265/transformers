import os
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fmoe import DistributedGroupedDataParallel as DGDP

import src.distributed_moe as distributed_moe
import src.moe as moe
import src.fast_moe as fast_moe


def test():
    x = torch.randn(1, 1024, 1024).cuda()
    y = torch.randn(1, 1024, 1024).cuda()

    y1, aux_loss1 = distmoe(x)
    loss1 = (y1 * y).sum() + aux_loss1
    loss1.backward()

    y2, aux_loss2 = standard_moe(x)
    loss2 = (y2 * y).sum() + aux_loss2
    loss2.backward()

    if rank == 0:
        print('Same routing:', torch.allclose(distmoe.module.gates, standard_moe.module.gates))
        print('Same output:', torch.allclose(y1, y2), torch.abs(y1 - y2).max().item())
        print('Same loss:', torch.allclose(loss1, loss2), torch.abs(loss1 - loss2).item())

        grad0 = distmoe.module.gate.w_gate[0].weight.grad
        grad1 = standard_moe.module.gate.w_gate[0].weight.grad
        print('Same gate weight grad:', torch.allclose(grad0, grad1), torch.abs((grad0 - grad1)).max().item())

        for j in range(distmoe.module.local_num_experts):
            grad0 = distmoe.module.input_weight.grad[j]
            grad1 = standard_moe.module.experts.weight.grad[local_rank % local_size + j * local_size]
            print('Same input weight grad: ', j, torch.allclose(grad0, grad1), torch.abs(grad0 - grad1).max().item())

        for j in range(distmoe.module.local_num_experts):
            grad0 = distmoe.module.output_weight.grad[j]
            grad1 = standard_moe.module.output_experts.weight.grad[local_rank % local_size + j * local_size]
            print('Same output weight grad:', j, torch.allclose(grad0, grad1), torch.abs(grad0 - grad1).max().item())
        
        print()


if __name__ == "__main__":
    import os
    import time

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # from .moe import MoE as standard_MoE

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    # print(world_size, local_rank)

    local_size = 4

    distmoe = distributed_moe.MoE(
        input_size=1024, 
        head_size=1024, 
        num_experts=8, 
        top_k=1, 
        local_size=local_size,
        activation=nn.GELU(),
        aux_loss='mi'
    )
    distmoe.to(local_rank)

    standard_moe = moe.MoE(
        input_size=1024, 
        head_size=1024, 
        num_experts=8, 
        top_k=1, 
        activation=nn.GELU(),
        aux_loss='mi'
    )
    standard_moe.to(local_rank)
    standard_moe = DDP(standard_moe, device_ids=[local_rank], gradient_as_bucket_view=True)
    distmoe = DDP(distmoe, device_ids=[local_rank], gradient_as_bucket_view=True)

    distmoe.module.gate.w_gate[0].weight.data = standard_moe.module.gate.w_gate[0].weight.data.clone().contiguous()
    distmoe.module.gate.w_gate[0].bias.data = standard_moe.module.gate.w_gate[0].bias.data.clone().contiguous()
    distmoe.module.gate.w_gate[3].weight.data = standard_moe.module.gate.w_gate[3].weight.data.clone().contiguous()

    if rank < local_size:
        input_expert_weight = standard_moe.module.experts.weight.data
        output_expert_weight = standard_moe.module.output_experts.weight.data
        for i in range(distmoe.module.local_num_experts):
            distmoe.module.input_weight.data[i].copy_(input_expert_weight[i * local_size + distmoe.module.local_rank])
            distmoe.module.output_weight.data[i].copy_(output_expert_weight[i * local_size + distmoe.module.local_rank])

    distmoe.module.sync_experts()

    distmoe.register_comm_hook(state=None, hook=distributed_moe.moe_comm_hook)

    for i in range(6):
        with distmoe.no_sync() and standard_moe.no_sync():
            test()
            dist.barrier()

        # distmoe.zero_grad()
        # standard_moe.zero_grad()
    
    test()
    dist.barrier()

    # parser = argparse.ArgumentParser(prog = 'test MoE')
    # parser.add_argument('--moe_type', type=str, default='moe')
    # parser.add_argument('--num_experts', type=int, default=96)
    # parser.add_argument('--world_size', type=int, default=None)
    # args = parser.parse_args()

    # dist.init_process_group(backend="nccl")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["RANK"])
    # torch.cuda.set_device(local_rank)
    # print(world_size, local_rank)

    # if args.moe_type == 'moe':
    #     moe = moe.MoE(
    #         input_size=512, 
    #         head_size=512, 
    #         num_experts=args.num_experts, 
    #         top_k=4, 
    #         miloss=0.1, 
    #         activation=nn.ReLU(),
    #     )
    #     moe.to(local_rank)
    #     moe = DDP(moe)
    # elif args.moe_type == 'fastmoe':
    #     moe = fast_moe.MoE(
    #         input_size=512, 
    #         head_size=512, 
    #         num_experts=args.num_experts, 
    #         top_k=4, 
    #         miloss=0.1, 
    #         activation=nn.ReLU(),
    #         world_size=args.world_size,
    #     )
    #     moe.to(local_rank)
    #     moe = DGDP(moe)
    # elif args.moe_type == 'distmoe':
    #     moe = distributed_moe.MoE(
    #         input_size=512, 
    #         head_size=512, 
    #         num_experts=args.num_experts, 
    #         top_k=4, 
    #         miloss=0.1, 
    #         activation=nn.ReLU(),
    #         world_size=args.world_size,
    #     )
    #     moe.to(local_rank)
    #     moe = DDP(moe)


    # x = torch.randn(32, 4096, 512).to(local_rank)
    # y, aux_loss = moe(x)
    # loss = y.sum() + aux_loss
    # moe.zero_grad(set_to_none=True)
    # loss.backward()
    # torch.cuda.synchronize()
    # dist.barrier()

    # start_time = time.time()
    # for i in range(20):
    #     x = torch.randn(32, 1024, 512).to(local_rank)

    #     y, aux_loss = moe(x)

    #     loss = y.sum() + aux_loss
    #     moe.zero_grad(set_to_none=True)
    #     loss.backward()
    #     torch.cuda.synchronize()
    #     dist.barrier()
    # end_time = time.time()
    # if rank == 0:
    #     print("Forward time: ", (end_time - start_time))

    # start_time = time.time()
    # for i in range(20):
    #     x = torch.randn(32, 1024, 512).to(local_rank)

    #     mapped, aux_loss = moe.module.map(x)
    #     mapped = moe.module.activation(mapped)
    #     y = moe.module.reduce(mapped)

    #     loss = y.sum() + aux_loss
    #     moe.zero_grad(set_to_none=True)
    #     loss.backward()
    #     torch.cuda.synchronize()
    #     dist.barrier()
    # end_time = time.time()
    # if rank == 0:
    #     print("Map-reduce time: ", (end_time - start_time))

    # print(moe.module.input_experts[0].weight.grad)

    # mapped, loss = moe.map(x)
    # mapped = moe.activation(mapped)
    # reduced = moe.reduce(mapped)

    # print(torch.abs(y - reduced).max())
    # # print(torch.allclose(y, reduced))
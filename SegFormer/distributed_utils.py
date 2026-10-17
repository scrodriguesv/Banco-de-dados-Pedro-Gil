import torch
from torch import nn
import torch.distributed as dist
import numpy as np
import os
import sys


def load_ddp_model(weight_dict):
    return {k.replace("module.", ""): v for k, v in weight_dict.items()}

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def print_only_on_master(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(rank=0, world_size=1, gpu=0):
    if dist.is_initialized():
        dist.destroy_process_group()

    # The rest of your original code follows
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        rank, gpu, world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(gpu)
    dist.barrier()
    print_only_on_master(rank == 0)
    return rank, world_size, gpu

def is_main_device():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        gpu = int(os.environ['LOCAL_RANK'])
    elif torch.cuda.is_available():
        gpu = 0
    return (gpu == 0) and (is_main_process())

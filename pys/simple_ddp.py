import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import torch
from torch.utils.data import DataLoader, Dataset
import os


nnode = 2 # Number of nodes(mahcines)
#gpu_per_node = 4 # Number of GPUs(Processes) per node.
gpu_per_node = [2, 4] # Or you can make a list for number of GPUs in case that each machine has different num of them.
backend = 'gloo' # Backend. Read pytorch.DDP document for more details.
init_method = 'env://' # Current Value is default. Read pytorch.DDP document for more details.
master_addr = '127.0.0.1' # Master address. Replace with the IP address of Node 0.
master_port = '12345' # Master Port.

# Individual node configuration. You have to edit this part for each node(machine)
node = 0 # Node ID. 0 is the Master node.

# Define your model, dataset, and some hyperparameters here. Edit below lines.
nn_model = torch.nn.Module() # Replace with your own model to use.
dataset = Dataset() # Replace with your own dataset.

batch_size = 100 # Replace with your own batch_size# Common configuration values for distributed training.

def train():
    #Write your own code here
    pass

def infer():
    #Write your own code here
    pass

def dist_main(local_rank:int, world_size:int, _rank0:int)->None:
    r'''Entry point for each process.
    The first argument is always local rank which is provided by torch.mp.
    '''
    global_rank = local_rank + _rank0

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(backend, init_method, world_size=world_size, rank=global_rank)

    sampler = DistributedSampler(dataset, world_size, global_rank)
    # Replace the arguments as you wish except the sampler.
    dataloader = DataLoader(dataset, batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    ddp_model = DDP(nn_model.to(local_rank), device_ids=[local_rank])

    # Do Something Here
    # train()...
    # infer()...

    dist.destroy_process_group()

if __name__ == '__main__':
    if isinstance(gpu_per_node, int):
        world_size = nnode * gpu_per_node
        local_size = gpu_per_node
        _rank0 = node * gpu_per_node
    elif isinstance(gpu_per_node, list):
        world_size = sum(gpu_per_node)
        local_size = gpu_per_node[node]
        _rank0 = sum(gpu_per_node[0:node])
    mp.spawn(dist_main, (world_size, _rank0), local_size, join=True, )
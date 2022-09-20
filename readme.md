# Pytorch DDP(DistributedDataParallel) Templates

### This code goes hard, feel free to copy and paste :skull:

Simple Template for Pytorch DDP(DistributedDataParallel).

Multiprocessing for DDP is troublesome so I made a template.

You just need to edit where I mentioned in the codes.

Edite below variables for your own use

### Common Variables

| Name | Required | Description |
| --- | --- | --- |
| nnode | Yes | the number of your machines |
| gpu_per_proc | Yes | the number of GPUs for each node. You can define a list instead of integer. |
| backend | Yes | Backend. Choose among 'nccl', 'gloo', 'mpi'. Read [this](https://pytorch.org/docs/master/distributed.html) for more details. |
| init_method | No | An argument for initializing a process group. Default value is 'env://'. Read 'Initialization' part [here](https://pytorch.org/docs/stable/distributed.html) |
| master_addr | Yes | Master address. Rank 0 process will be the master process, so set this as the IP address of node 0. |
| master_port | Yes | Master port. same detail as above. |
| node | Yes | Node id. Node 0 will broadcast and sync gridents. Should be different for each machine. | 



Examples below based on simple_ddp

```python
nnode = 4
gpu_per_proc = 4 # Every node has 4 GPUs
backend = 'nccl'
master_addr = '162.138.23.15'
master_port = '8888'
```
In this case, 4 Machines have 4 GPUs each using the NCCL backend.

```python
nnode = 4
gpu_per_proc = [4, 4, 2, 2] # Every node has 4 GPUs
backend = 'gloo'
master_addr = '162.138.23.15'
master_port = '8888'
```
Above setting describes that 2 Machines have 4 GPUs, and the others have 2 of them.
In this case you have to set `gpu_per_proc` to 4 in node 0, 1, and 2 in node 2, 3.

```python
nnode = 1
gpu_per_proc = torch.cuda.device_count()
backend = 'nccl'
master_addr = 'localhost'
master_port = '12345'
```
If you are using a single node, then you don't have to type the whole IP address. Just type `'localhost'`.
Also you can just use `torch.cuda.device_count()` for `gpu_per_proc` in a single node environment.
In a multi-nodes environment does require a specific number or list for `gpu_per_proc`
because world size need to be calculated initialize a process group.

```python
nnode = 2
gpu_per_proc = 4
backend = 'nccl'
init_method = 'tcp://162.138.23.15'
```
init_method is an argument for `torch.distributed.init_process_group`.
`torch.distributed` has various ways of initialization, so this part is on your hands.

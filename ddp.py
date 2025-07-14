import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torchvision
import torchvision.transforms as transforms

from helper_cupti_um import setup_cupti_um, free_cupti_um
from managed_alloc import managed_alloc

training_verbose = False

def setup_dataset():
    if training_verbose:
        print("Setting up CIFAR10 dataset")
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    return trainset

def setup_ddp(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if training_verbose:
        print(f"Setting up DDP for rank {rank} out of {world_size} processes")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354"

    # nccl environment variables
    os.environ["NCCL_DEBUG"] = "VERSION"  # Set NCCL debug
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P to avoid issues with some GPUs
    os.environ["NCCL_IB_DISABLE"] = "0"  # Disable Infiniband to avoid issues with some networks

    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        if training_verbose:
            print(f"Initializing Trainer on GPU {gpu_id}")
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        if training_verbose:
            print(f"[GPU{self.gpu_id}] Running batch with {len(source)} samples")
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        if training_verbose:
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        if training_verbose:
            print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        if training_verbose:
            print(f"[GPU{self.gpu_id}] Starting training for {max_epochs} epochs")
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    if training_verbose:
        print("Loading training objects")
    train_set = setup_dataset() # CIFAR10 dataset
    model = torchvision.models.alexnet(num_classes=10) # AlexNet model for CIFAR10
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    if training_verbose:
        print(f"Preparing DataLoader with batch size {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    setup_ddp(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


# [v] TODO: redirect sys.stdout and print results to a file
# TODO: setup the correct paths for cupti-python module and allocator module
# TODO: add more detailed profiling information such as latency, memory usage, execution time, etc.
# [v] TODO: add execution time
if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='diststibuted training for alexnet')
    # parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    # parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    # parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    # args = parser.parse_args()

    print("Device available:", torch.cuda.is_available())

    start = time.perf_counter_ns()

    # change pytorch allocator to managed memory
    managed_alloc()

    # setup CUPTI for Unified Memory profiling
    cupti_setup_start = time.perf_counter_ns() 
    # setup_cupti_um(filename='output.txt') # TODO: argument for filename
    setup_cupti_um() # TODO: argument for filename
    cupti_setup_end = time.perf_counter_ns()

    total_epochs = 1
    save_every = 1
    batch_size = 32
    
    # world_size = torch.cuda.device_count()
    world_size = 2  # For testing, set to just one process
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)

    # free CUPTI resources
    cupti_free_start = time.perf_counter_ns()
    free_cupti_um()
    cupti_free_end = time.perf_counter_ns()

    end = time.perf_counter_ns()

    print(f"Total execution time: {(end - start)} ns")
    print(f"CUPTI setup time: {(cupti_setup_end - cupti_setup_start)} ns")
    print(f"CUPTI free time: {(cupti_free_end - cupti_free_start)} ns")
    print(f"Total execution time without CUPTI overhead: {(end - start - (cupti_setup_end - cupti_setup_start) - (cupti_free_end - cupti_free_start))} ns")
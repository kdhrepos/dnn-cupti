import torch

def managed_alloc():
    # Load the allocator
    managed_alloc = torch.cuda.memory.CUDAPluggableAllocator('./alloc.so', 'managed_alloc', 'managed_free')

    # Swap the current allocator
    torch.cuda.memory.change_current_allocator(managed_alloc)

    print("torch allocator for CUDA has been changed to managed memory.")
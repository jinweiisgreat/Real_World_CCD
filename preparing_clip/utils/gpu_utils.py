import torch
def check_gpu(local=None):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_cached()
    if local != None:
        print(f'Allocated memory:{allocated / 1024**3} GB {local}')
        print(f'Cached memory:{cached / 1024**3} GB {local}')
    else:
        print(f'Allocated memory:{allocated / 1024**3} GB')
        print(f'Cached memory:{cached / 1024**3} GB')
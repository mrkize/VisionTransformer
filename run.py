import os
# os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --set 2")
# os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --nums 27")
# os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --nums 54")
# os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --nums 108")
os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --nums 135")
os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py --nums 162")
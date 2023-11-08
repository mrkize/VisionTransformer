import argparse
import os
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--ordinary_train', action="store_true", help='whether use mask')
parser.add_argument('--epochs', type=int, default=50, help='training epoch')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
parser.add_argument('--model_type', type=str, default='ape', help='model name')
parser.add_argument('--ape', action="store_true", help='use ape')
parser.add_argument('--init', action="store_true", help='init model or finetune')
parser.add_argument('--pe_aug', type=str, default='mask', help='if fill')
parser.add_argument('--img_aug', type=str, default='pbf', help='if fill')
parser.add_argument('--nums', type=int, default=-1, help='if fill')
parser.add_argument('--set', type=int, default=0, help='if fill')
parser.add_argument('--mix_up', action="store_true", help='use Mixup')
parser.add_argument('--n_class', type=int, default=10, help='')
parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--defence", type=str, default='', help="defence method")
parser.add_argument("--istarget", action="store_false", help="defence method")
parser.add_argument('--gpu', type=str, default='cuda:0', help='if fill')
args = parser.parse_args()

nums = [0, 10, 20, 30, 40, 50]
nums_list = [0, 10, 40, 70, 100, 130, 160, 190]
nums_list2 = [0, 5, 10, 15, 20, 25, 30]
nums_list3 = [0, 27, 54, 108, 189, 196]
ls_rate = [0.1, 0.2, 0.3]
if args.nums >= 0:
    pad = " --istarget"
    os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main_cp.py --dataset {} --nums {}".format(args.dataset,args.nums))
    os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main_cp.py --dataset {} --nums {}{}".format(args.dataset, args.nums, pad))
else:
    for i in nums_list:
        pad = " --istarget"
        os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main_cp.py --dataset {} --nums {}".format(args.dataset,i))
        os.system("python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main_cp.py --dataset {} --nums {}{}".format(args.dataset, i, pad))
# args.dataset = "ImageNet10"
# for i in nums_list:
#     pad = " --istarget"
#     os.system("python main.py --dataset {} --nums {}".format(args.dataset,i))
#     os.system("python main.py --dataset {} --nums {}{}".format(args.dataset, i, pad))

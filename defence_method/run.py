import argparse
import datetime
import os
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True)
parser.add_argument('--dataset', type=str, default='cifar10',help='attack dataset')
parser.add_argument('--model', type=str, default="all")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_class', type=int, default=100)
parser.add_argument('--noise_repeat', type=int, default=1)
parser.add_argument('--head_fusion', type=str, default='mean')
parser.add_argument('--discard_ratio', type=float, default=0)
parser.add_argument('--addnoise', action='store_true', default=False)
parser.add_argument('--metric', type=str, default="cos-sim")
parser.add_argument('--atk_method', type=str, default="base_e")
parser.add_argument('--lr', type=float, default=0.004)
parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument('--noise_nums', type=int, default=10)
parser.add_argument('--adaptive', action='store_true', default=False)
args = parser.parse_args()

args = parser.parse_args()
nums =  ["000", "138", "276", "413", "551", "689", "827", "964"]
nums_2 = ["000", "051", "153", "255", "357", "459", "561", "663", "765", "867", "969"]
nums_3 = ["000", "051", "204", "357", "510", "663", "816", "969"]
nums_4 = ["000", "051", "357", "510", "663", "816", "969"]
model_names = ["pbf_mask_0.{}.pth".format(i) for i in nums]
model_names2 = ["orain_mask_0.{}.pth".format(i) for i in nums_2]
model_names3 = ["pbf_mask_0.{}.pth".format(i) for i in nums_3]
model_names4 = ["pbf_mask_0.{}.pth".format(i) for i in nums_4]
pad = " --adaptive" if args.adaptive else ""
atk_list = ["roll_nn", "last_attn_nn", "out", "roll", "last_attn"]
dataset_list = ["cifar10", "cifar100", "ImageNet100"]
metric_list = ["metric_roll", "metric_last_attn", "metric_out"]
mia_type_list = ["ft_nn", "ft_metric"]
metric = ["Euclid", "CE"]

def mark(mul, name):
    with open("Result.txt", "a") as wf:
        wf.write('-' * 120 + "\n")
        wf.write(str(datetime.datetime.now())[:19] + '\n')
        wf.write("{} model(DP bound {}) has trained!\n".format(name,mul))
# run basic
# for dataset in dataset_list:
#     os.system(" python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 Basic.py --dataset {}".format(dataset))
#     os.system(" python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 Basic.py --dataset {} --istarget".format(dataset))

# run label_smoothing
# ls_rate = [0.1, 0.2, 0.3, 0.4]
# for dataset in dataset_list:
#     for ls in ls_rate:
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 label_smoothing.py --dataset {} --ls {}".format(dataset, ls))
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 label_smoothing.py --dataset {} --ls {} --istarget".format(dataset, ls))

# run RelaxLoss
# alpha = [0.25, 0.5, 0.75, 1]
# for dataset in dataset_list:
#     for alp in alpha:
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 RelaxLoss.py --dataset {} --alpha {}".format(dataset, alp))
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 RelaxLoss.py --dataset {} --alpha {} --istarget".format(dataset, alp))

# run adv_reg
# lambd = [0.5, 1, 2, 3]
# for dataset in dataset_list:
#     for lam in lambd:
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 adv_reg.py --dataset {} --lambd {}".format(dataset, lam))
#         os.system(
#             " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 adv_reg.py --dataset {} --lambd {} --istarget".format(dataset, lam))


# run PEdrop
nums_list = [100, 120, 140, 160]
for dataset in dataset_list:
    for nums in nums_list:
        os.system(
            " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 PEdrop.py --dataset {} --nums {}".format(dataset, nums))
        os.system(
            " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 PEdrop.py --dataset {} --nums {} --istarget".format(dataset, nums))
        

# run DPSGD
noise_mul = [0.5, 1, 1.2, 1.5]
for dataset in dataset_list:
    for mul in noise_mul:
        os.system(
            " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 DPSGD.py --dataset {} --noise_multiplier {}".format(dataset, mul))
        mark(mul, "target")
        os.system(
            " python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 DPSGD.py --dataset {} --noise_multiplier {} --istarget".format(dataset, mul))
        mark(mul, "shadow")

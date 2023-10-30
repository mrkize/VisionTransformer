import argparse
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
for model_name in model_names3:
    os.system("python mia_attn.py --dataset {} --atk_method {} --model {} --adaptive".format(args.dataset, args.atk_method, model_name))
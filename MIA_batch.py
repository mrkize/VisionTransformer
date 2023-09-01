import argparse
import os


parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--maskPE', action = "store_true",
                    help='whether use PE')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch_size')
parser.add_argument('--mia_type', type=str, default='nn-based',
                    help='num of workers to use')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='number of training epochs')
opt = parser.parse_args()


cifar10_ratio = ['0', '0.062', '0.125', '0.188', '0.250', '0.312', '0.375', '0.438', '0.500', '0.562', '0.625', '0.688', '0.750', '0.812', '0.875', '0.938']
ImageNet10_ratio = ['{:.3f}'.format(i/196) for i in range(6,196,12)]
ImageNet10_ratio = ['0']+ImageNet10_ratio
if opt.dataset == 'cifar10':
    ratio = cifar10_ratio
elif opt.dataset == 'cinic10':
    ratio = cifar10_ratio
elif opt.dataset == 'ImageNet10':
    ratio = ImageNet10_ratio
else:
    ratio = ['0'] + ['{:.3f}'.format(i/49) for i in range(3,49,3)]
cmd = 'python MIA.py --mia_type {} --dataset {} --model '.format(opt.mia_type,opt.dataset)
model_name = "ViT_mask_"
for i in ratio:
    commond = cmd + 'ViT' if i=='0' else cmd + model_name + i
    if opt.maskPE:
        commond+=' --maskPE'
    os.system(commond)


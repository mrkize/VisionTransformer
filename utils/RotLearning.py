# from experience_mnist import *
import os
import shutil
import datetime
import argparse
import sys
from trainer import *

sys.path.append( "path" )

config = config()
now = str(datetime.datetime.now())[:19]
now = now.replace(":","")
now = now.replace("-","")
now = now.replace(" ","")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TrajectoryMIA')
    parser.add_argument("--pretrain", action="store_true", help="is training state or not")
    parser.add_argument("--finetune", action="store_true", help="is training state or not")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='target', help=['target', 'shadow', 'distill_target', 'distill_shadow'])
    parser.add_argument('--model', type=str, default='vgg', help=['resnet', 'mobilenet', 'vgg', 'wideresnet'])
    parser.add_argument('--data', type=str, default='cifar10', help=['cinic10', 'cifar10', 'cifar100', 'gtsrb', 'rot_data'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model_distill', type=str, default='vgg', help=['resnet', 'mobilenet', 'vgg', 'wideresnet'])
    parser.add_argument('--target_path', type=str, default='', help="attack other model's path")
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--mia_type', type=str, help=['build-dataset', 'black-box'])

    args = parser.parse_args()
    config.set_subkey('general', 'seed', args.seed)
    src_dir = config.path.data_path

    path = './TransferLearning/'
    if not os.path.exists(path):
        os.makedirs(path)
    dst_dir = path + "/config.yaml"
    shutil.copy(src_dir, dst_dir)
    print(args)
    if args.pretrain:
        model_target, train_loader_target, test_loader_target = get_model(args.model, args.data, istarget=True,
                                                                      not_train=args.not_train, config=config,
                                                                      model_path=path, res_path=path)







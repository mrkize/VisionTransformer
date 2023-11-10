import argparse
import csv

import torch
import tim
from utils import MyConfig
from dataloader import get_loader
import os
config_dict = { 'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'ImageNet100': "config/ImageNet100/"
                }

dataset_class_dict = {
    "cifar10": 10,
    "cifar100": 100,
    "ImageNet100": 100
}



def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument("--istarget", action="store_false")
    opt = parser.parse_args()
    return opt


def predict(model, dataloaders, dataset_sizes, device, is_img=False):
    model.to(device)
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(dataloaders):
        inputs, labels = data.to(device), target.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels = torch.squeeze(labels)
        running_corrects += preds.eq(labels).sum().item()
    acc = 1.0 * running_corrects / dataset_sizes
    return acc

def change_csv(dir, value, split):
    rows = []
    with open(dir, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    for row in rows:
        if row['def_method'] == opt.model:
            row[split] = round(value, 4)

    fieldnames = rows[0].keys()
    with open(dir, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)
config.set_subkey('learning', 'DDP', False)
config.set_subkey('learning', 'DP', False)
config.set_subkey('learning', 'batch_size', 256)
loader, size = get_loader(opt.dataset, config, is_target=opt.istarget)
model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
model.load_state_dict(torch.load("defence_model/{}/".format(opt.dataset) + opt.model))
acc = predict(model, loader["train"], size["train"], device)
print("{} {} train acc: {:.4f}".format(opt.dataset, opt.model, acc))
change_csv("defence_result/{}.csv".format(opt.dataset), acc, "train_acc")
acc = predict(model, loader["val"], size["val"], device)
print("{} {} val acc: {:.4f}".format(opt.dataset, opt.model, acc))
change_csv("defence_result/{}.csv".format(opt.dataset), acc, "val_acc")
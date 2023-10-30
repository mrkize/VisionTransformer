import argparse
import os
import time
import datetime
import numpy as np
import torch.nn.functional as F
import torch
import tim_ff
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import TensorDataset
from tqdm import tqdm
import tim
from dataloader import get_loader
from utils import MyConfig
from vit_rollout import rollout, VITAttentionRollout
from torch import nn
import timm

config_dict = {'cifar10': "config/cifar10/",
                'cifar100': "config/cifar100/",
                'cinic10': "config/cinic10/",
                'Fmnist': "config/fashion-mnist/",
                'ImageNet10': "config/ImageNet10/",
                'ImageNet100': "config/ImageNet100/",
                'cifar10-8': "config/cifar10-8*8/",
                'ImageNet10-8': "config/ImageNet10-8*8/"
                }


dataset_class_dict = {
    "STL10": 10,
    "cifar10": 10,
    "cifar100": 100,
    "cinic10": 10,
    "CelebA": 2,
    "Place365": 2,
    "Place100": 2,
    "Place50": 2,
    "Place20": 2,
    "ImageNet100": 100,
    "ImageNet10": 10,
    "Fmnist": 10
}

wf = open("Result.txt", "a")


def write_spilt():
    wf.write('-'*120 + "\n")


def write_time():
    wf.write(str(datetime.datetime.now())[:19]+'\n')


def write_res(res):
    line = "Acuracy: {:.4f}{}Precision: {:.4f}{}Recall: {:.4f}".format(res[0], " "*5, res[1], " "*5, res[2])
    line += "\n"
    wf.write(line)


def write_config(args):
    line = "{:<16}{:<23}Atk_method: {:<11}metric: {:<12}Adaptive: {}\n".format(
        args.dataset, args.model, args.atk_method, args.metric, args.adaptive)
    wf.write(line)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--dataset', type=str, default='ImageNet10',help='attack dataset')
    parser.add_argument('--model', type=str, default="pbf_mask_0.000.pth")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--noise_repeat', type=int, default=1)
    parser.add_argument('--head_fusion', type=str, default='mean')
    parser.add_argument('--discard_ratio', type=float, default=0)
    parser.add_argument('--addnoise', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default="cos-sim")
    parser.add_argument('--atk_method', type=str, default="last_attn_nn")
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument('--noise_nums', type=int, default=10)
    parser.add_argument('--adaptive', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='cuda:0')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    return args


class classifier(nn.Module):
    def __init__(self, input_dim=384, std=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 32)
        torch.nn.init.normal_(self.fc1.weight.data, 0, std)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        torch.nn.init.normal_(self.fc2.weight.data, 0, std)
        # self.fc3 = nn.Linear(32, 32)
        # torch.nn.init.normal_(self.fc2.weight.data, 0, 0.1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 2)
        torch.nn.init.normal_(self.fc4.weight.data, 0, std)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.batch_norm2(x)
        x = F.relu(self.fc4(x))
        return x


def add_gaussian_noise_to_patches(image_tensor, patch_size=16, num_patches=10, mean=0, stddev=0.25):
    batch_size, channels, height, width = image_tensor.size()
    unfolded = torch.nn.functional.unfold(image_tensor, patch_size, stride=patch_size)
    selected_patches = torch.randperm(unfolded.size(2))[:num_patches]
    noise = torch.randn(batch_size, unfolded.shape[1], num_patches) * stddev + mean
    noise = noise.to(args.device)
    unfolded = unfolded.to(args.device)
    unfolded[:, :, selected_patches] += noise
    unfolded = unfolded.clamp(0, 1)
    image_tensor = torch.nn.functional.fold(unfolded, (height, width), patch_size, stride=patch_size)
    return image_tensor

def calculate_entropy(heatmap, epsilon = 1e-8):
    heatmap = heatmap + epsilon
    probabilities = heatmap / heatmap.sum(-1)[...,None]
    entropy = -torch.sum(probabilities * torch.log2(probabilities), dim=-1)
    return entropy


def CrossEntropy(x, target):
    loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
    return loss


def precision(y_true, y_pred):
    true_positives = torch.sum((y_true == 1) & (y_pred == 1))
    false_positives = torch.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives = torch.sum((y_true == 1) & (y_pred == 1))
    false_negatives = torch.sum((y_true == 1) & (y_pred == 0))
    return true_positives / (true_positives + false_negatives)


def pearson_correlation(x, y, dim):
    mean_x = torch.mean(x, dim=dim)
    mean_y = torch.mean(y, dim=dim)
    cov = torch.sum((x - mean_x.unsqueeze(dim)) * (y - mean_y.unsqueeze(dim)), dim=dim) / x.size(dim)
    std_x = torch.std(x, dim=dim)
    std_y = torch.std(y, dim=dim)
    correlation = cov / (std_x * std_y)
    return correlation


def switch_fused_attn(model):
    for name, block in model.blocks.named_children():
        block.attn.fused_attn = False
    return model


def init_config_model_attn(args):
    config = MyConfig.MyConfig(path=config_dict[args.dataset])
    config.set_subkey('learning', 'DP', False)
    config.set_subkey('learning', 'DDP', False)
    config.set_subkey('learning', 'batch_size', 256)
    args.device = gpu_init(config, args)
    # args.device = torch.device('cpu')
    args.num_class = dataset_class_dict[args.dataset]
    target_model = tim.create_model('vit_small_patch16_224', num_classes=args.num_class)
    target_model.load_state_dict(torch.load(config.path.model_path + args.model))
    shadow_model = tim.create_model('vit_small_patch16_224', num_classes=args.num_class)
    if args.adaptive:
        shadow_model.load_state_dict(torch.load(config.path.model_path + args.model[:-4]+"_shadow.pth"))
    else:
        shadow_model.load_state_dict(torch.load(config.path.model_path + args.model[:-7] + "000_shadow.pth"))
    if "attn" in args.atk_method or "rollout" in args.atk_method:
        target_model = switch_fused_attn(target_model)
        shadow_model = switch_fused_attn(shadow_model)
    # target_model, shadow_model = target_model.to(args.device), shadow_model.to(args.device)
    if config.learning.DDP:
        target_model = torch.nn.parallel.DistributedDataParallel(target_model, device_ids=[args.local_rank],
                                                                 output_device=args.local_rank,
                                                                 find_unused_parameters=True)
        # print('lr:',config.learning.learning_rate)
        shadow_model = torch.nn.parallel.DistributedDataParallel(shadow_model, device_ids=[args.local_rank],
                                                                 output_device=args.local_rank,
                                                                 find_unused_parameters=True)
    elif config.learning.DP:
        target_model=torch.nn.DataParallel(target_model, device_ids=[0,1])
        shadow_model=torch.nn.DataParallel(shadow_model, device_ids=[0,1])

    # target_rollout = None
    # shadow_rollout = None
    if config.learning.DP or config.learning.DP:
        target = target_model.module
        shadow = shadow_model.module
    else:
        target = target_model
        shadow = shadow_model
    target_rollout = VITAttentionRollout(target, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio, device=args.device)
    shadow_rollout = VITAttentionRollout(shadow, head_fusion=args.head_fusion, discard_ratio=args.discard_ratio, device=args.device)
    return config, args, target_model, shadow_model, target_rollout, shadow_rollout


def get_attn(model, dataloaders, attention_rollout, noise=False, out_atk="out"):
    attn_metric = []
    for batch_idx, (inputs, labels) in enumerate(dataloaders):
        inputs = inputs.to(args.device)
        if noise:
            inputs = add_gaussian_noise_to_patches(inputs)
        mask = attention_rollout(inputs, out_atk)
        attn_metric.append(mask)
    attn = torch.cat(attn_metric, dim=0)
    return attn


def get_base_e(dataloaders, attention_rollout):
    attn_metric = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(args.device)
            mask = attention_rollout(inputs, True)[:,:,1:]
            attn_metric.append(mask)
    attn = torch.cat(attn_metric, dim=0)
    return attn


def gpu_init(config, opt):
    if config.learning.DDP:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def find_best_threshold(data, labels):
    sorted_data, indices = torch.sort(data)
    sorted_labels = labels[indices]
    best_accuracy = 0.0
    best_threshold = 0.0
    for i in range(len(sorted_data) - 1):
        if sorted_labels[i+1] != 0:
            continue
        threshold = (sorted_data[i] + sorted_data[i+1]) / 2.0
        predicted_labels = (data <= threshold).to(torch.int32)
        accuracy = (predicted_labels == labels).to(torch.float32).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def calculate_accuracy(data, labels, threshold):
    predicted_labels = (data <= threshold).to(torch.int32)
    accuracy = (predicted_labels == labels).to(torch.float32).mean()
    pre = precision(predicted_labels, labels)
    rec = recall(predicted_labels, labels)
    return accuracy, pre, rec


def get_data(model, attention_rollout, loader, atk_method):
    attn_orain_train = get_attn(model, loader["train"], attention_rollout, noise=False, out_atk=atk_method)
    attn_orain_val = get_attn(model, loader["val"], attention_rollout, noise=False, out_atk=atk_method)
    # train_res = torch.zeros(attn_orain_train.shape[0]).to(args.device)
    # val_res = torch.zeros(attn_orain_val.shape[0]).to(args.device)
    train_res_list = []
    val_res_list = []
    for i in range(args.noise_repeat):
        attn_train= get_attn(model, loader["train"], attention_rollout, noise=True, out_atk=atk_method)
        metric_train_repeat = torch.cosine_similarity(attn_orain_train,attn_train,dim=1)
        # 计算欧式距离
        # metric_train_repeat = torch.norm(attn_orain_train - attn_train, dim=1)
        #计算交叉熵
        # metric_train_repeat = CrossEntropy(attn_orain_train, attn_train)
        # 皮尔逊相关系数
        # metric_train_repeat = pearson_correlation(attn_orain_train, attn_train, -1)
        # train_res += metric_train_repeat
        train_res_list.append(metric_train_repeat)

        attn_val= get_attn(model, loader["val"], attention_rollout, noise=True, out_atk=atk_method)
        metric_val_repeat = torch.cosine_similarity(attn_orain_val,attn_val, dim=1)
        # 计算欧式距离
        # metric_val_repeat = torch.norm(attn_orain_val - attn_val, dim=1)
        # 计算交叉熵
        # metric_val_repeat = CrossEntropy(attn_orain_val, attn_val)
        # 皮尔逊相关系数
        # metric_val_repeat = pearson_correlation(attn_orain_val, attn_val, -1)
        # val_res += metric_val_repeat
        val_res_list.append(metric_val_repeat)

    if "nn" not in atk_method:
        train_res = sum(train_res_list) / args.noise_repeat
        val_res = sum(val_res_list) / args.noise_repeat
    else:
        train_res = torch.stack(train_res_list, dim=-1)
        val_res = torch.stack(val_res_list, dim=-1)
    data = torch.cat([train_res, val_res])
    target = torch.cat([torch.ones(train_res.shape[0], dtype=torch.long), torch.zeros(val_res.shape[0], dtype=torch.long)]).to(args.device)
    # dataset = TensorDataset(data, target)
    # return dataset
    return data, target


def train(model, loader, size, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    since = time.time()
    model.train()
    epoch_acc = 0
    tq = tqdm(range(epochs), ncols=100)
    for epoch in tq:
        # scheduler.step()
        running_corrects = 0
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(loader):
            inputs, labels = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
            running_corrects += preds.eq(target.to(args.device)).sum().item()
        epoch_acc = 1.0 * running_corrects / size
        tq.set_postfix({"acc":epoch_acc, "loss":(sum(epoch_loss)/len(epoch_loss)).item()})
        # print("Train acc {:.5f}.".format(epoch_acc))
    time_elapsed = time.time() - since
    print("Last Training acc {:.5f}.".format(epoch_acc),end="")
    print(" ,used {:.5f}s.".format(time_elapsed))
    return model


def predict(model, loader, size):
    model.eval()
    running_corrects = 0
    TP = 0
    FP = 0
    FN = 0
    for batch_idx, (data, target) in enumerate(loader):
        inputs, labels = data.to(args.device), target.to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += preds.eq(target.to(args.device)).sum().item()
        TP += sum((preds == 1) & (labels == 1))
        FP += sum((preds == 0) & (labels == 1))
        FN += sum((preds == 1) & (labels == 0))
    acc = 1.0 * running_corrects / size
    print("Val acc {:.5f}.".format(acc))
    prec = TP / (TP + FP)
    reca = TP / (TP + FN)
    return acc, prec, reca


def attn_rollout_atk(args):
    config, args, target_model, shadow_model, target_rollout, shadow_rollout = init_config_model_attn(args)
    sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    sha_dataset, sha_target = get_data(shadow_model, shadow_rollout, sha_loader, args.atk_method)
    # sha_dataset, sha_target = sha_dataset.to(args.device), sha_target.to(args.device)
    thr = find_best_threshold(sha_dataset, sha_target)
    # print(thr)
    tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    tar_dataset, tar_target = get_data(target_model, target_rollout, tar_loader, args.atk_method)
    # tar_dataset, tar_target = tar_dataset.to(args.device), tar_target.to(args.device)
    val_acc, pre, rec = calculate_accuracy(tar_dataset, tar_target, thr)
    print("{}".format(args.model))
    print(val_acc)
    pad = "Output"
    if args.atk_method == "rollout":
        pad = "Attn rollout"
    elif args.atk_method == "last_attn":
        pad = "Last_attn"
    print("{} attack acc:{:.4f}\nprecision: {:.4f}\nRecall: {:.4f}".format(pad,val_acc, pre,rec))
    return val_acc, pre, rec


def get_output_data(model, attention_rollout, loader, atk_method):
    attn_orain_train = get_attn(model, loader["train"], attention_rollout, noise=False)
    attn_orain_val = get_attn(model, loader["val"], attention_rollout, noise=False)
    data = torch.cat([attn_orain_train, attn_orain_val], dim=0)
    target = torch.cat([torch.ones(attn_orain_train.shape[0], dtype=torch.long), torch.zeros(attn_orain_val.shape[0], dtype=torch.long)])
    dataset = torch.utils.data.TensorDataset(data,target)
    out_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
    return out_loader, len(dataset)


def attn_rollout_atk_nn(args):
    args.noise_repeat = 3
    args.epochs = 25
    args.lr = 0.002
    config, args, target_model, shadow_model, target_rollout, shadow_rollout = init_config_model_attn(args)
    sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    sha_dataset, sha_target = get_data(shadow_model, shadow_rollout, sha_loader, args.atk_method)
    tra_dataset = torch.utils.data.TensorDataset(sha_dataset, sha_target)
    train_loader = torch.utils.data.DataLoader(tra_dataset, batch_size=256, shuffle=True)
    atk_model = classifier(args.noise_repeat).to(args.device)
    atk_model = train(atk_model, train_loader, len(tra_dataset), args.epochs)
    # print(thr)
    tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    tar_dataset, tar_target = get_data(target_model, target_rollout, tar_loader, args.atk_method)
    val_dataset = torch.utils.data.TensorDataset(tar_dataset, tar_target)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    acc, pre, rec = predict(atk_model, val_loader, len(val_dataset))
    print("{}".format(args.model))
    print(acc)
    pad = "Output"
    if args.atk_method == "rollout":
        pad = "Attn rollout"
    elif args.atk_method == "last_attn":
        pad = "Last_attn"
    print("{} attack acc:{:.4f}\nprecision: {:.4f}\nRecall: {:.4f}".format(pad,acc, pre,rec))
    return acc, pre, rec


def baseline_d(args):
    args.epochs = 100
    args.lr = 0.21
    config, args, target_model, shadow_model, target_rollout, shadow_rollout = init_config_model_attn(args)
    sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    atk_train_loader, atk_loader_size = get_output_data(shadow_model, shadow_rollout, sha_loader, args.atk_method)
    atk_model = classifier().to(args.device)
    atk_model = train(atk_model, atk_train_loader, atk_loader_size, args.epochs)
    tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    atk_val_loader, val_loader_size = get_output_data(target_model, target_rollout, tar_loader, args.atk_method)
    acc, pre, rec = predict(atk_model, atk_val_loader, val_loader_size)
    print("Baseline_D attack acc:{:.4f}".format(acc))
    return acc, pre, rec


def get_compare(model, dataloaders):
    sim_metric = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(args.device)
            mask = model.forward_features(inputs, None)[:,1:,1:]
            center_patch = mask[:,int(mask.shape[1]/2),:].unsqueeze(1)
            similarity = torch.cosine_similarity(mask, center_patch, dim=-1).mean(-1)
            sim_metric.append(similarity)
    sim = torch.cat(sim_metric, dim=0)
    return sim


def get_sim_compare_data(model, loader):
    mem_sim = get_compare(model, loader["train"])
    no_mem_sim = get_compare(model, loader["val"])
    data = torch.cat([mem_sim, no_mem_sim], dim=0)
    target = torch.cat([torch.ones(mem_sim.shape[0], dtype=torch.long), torch.zeros(no_mem_sim.shape[0], dtype=torch.long)])
    return data, target.to(args.device)



def baseline_e(args):
    config, args, target_model, shadow_model, target_rollout, shadow_rollout = init_config_model_attn(args)
    sha_loader, sha_size = get_loader(args.dataset, config, is_target=False)
    sha_dataset, sha_target = get_sim_compare_data(shadow_model, sha_loader)
    thr = find_best_threshold(sha_dataset, sha_target)
    tar_loader, tar_size = get_loader(args.dataset, config, is_target=True)
    tar_dataset, tar_target = get_sim_compare_data(target_model, tar_loader)
    val_acc = calculate_accuracy(tar_dataset, tar_target, thr)
    print("{}".format(args.model))
    print("{} attack acc:{:.4f}".format("baseline-e",val_acc))
    return val_acc

torch.random.manual_seed(1333)
rollout_list = ["last_attn", "out", "rollout"]
args = get_args()
if args.atk_method in rollout_list:
    res = attn_rollout_atk(args)
elif args.atk_method == "base_d":
    res = baseline_d(args)
else:
    res = attn_rollout_atk_nn(args)
write_time()
write_config(args)
write_res(res)
write_spilt()


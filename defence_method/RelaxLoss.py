import sys
sys.path.append("..")
import argparse
from PIL import Image

import tim
from dataloader import get_loader
import torch
from utils import MyConfig
import os
import tim
import torch
import torch.nn as nn
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
import torch
from reprlib import recursive_repr
import torch.nn.functional as F


def get_opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=50, help='training epoch')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset and config')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument("--local-rank", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--istarget", action="store_false", help="defence method")
    opt = parser.parse_args()
    return opt




def gpu_init(opt):
    if config.learning.DDP:
        torch.distributed.init_process_group(backend="nccl")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 main.py
save_path = { 'cifar10': "../defence_model/cifar10/DP/",
              'cifar100': "../defence_model/cifar100/DP/",
              'ImageNet100': "../defence_model/ImageNet100/DP/"}


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


config_dict = { 'cifar10': "./config/cifar10/",
                'cifar100': "./config/cifar100/",
                'ImageNet100': "./config/ImageNet100/",
                }


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


class partial:
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = "func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func

        self = super(partial, cls).__new__(cls)

        self.func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args,
                                          self.keywords or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
                (kwds is not None and not isinstance(kwds, dict)) or
                (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.keywords = kwds


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def Relaxloss_train(model, loader, size, scheduler, optimizer, config, opt):
    scalar = GradScaler()
    since = time.time()
    for epoch in range(config.learning.epochs):
        if opt.local_rank == 0:
            print('Epoch {}/{}'.format(epoch, config.learning.epochs - 1))
        # train
        model.train()

        crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        crossentropy = nn.CrossEntropyLoss()
        softmax = nn.Softmax(dim=1)
        num_classes = opt.n_class
        alpha = opt.alpha
        upper = 1

        losses = AverageMeter()
        losses_ce = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        scheduler.step()
        # bar = Bar('Processing', max=len(loader['train']))
        for batch_idx, (data, target) in enumerate(loader['train']):
            inputs, targets = data.to(opt.device), target.to(opt.device)
            unk_mask = None
            dataload_time.update(time.time() - time_stamp)
            with autocast():
                outputs = model(inputs, unk_mask=unk_mask)
                loss_ce_full = crossentropy_noreduce(outputs, targets)
                loss_ce = torch.mean(loss_ce_full)

                if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
                    loss = (loss_ce - 1).abs()
                else:
                    if loss_ce > alpha:  # normal gradient descent
                        loss = loss_ce
                    else:  # posterior flattening
                        pred = torch.argmax(outputs, dim=1)
                        correct = torch.eq(pred, targets).float()
                        confidence_target = softmax(outputs)[torch.arange(targets.size(0)), targets]
                        confidence_target = torch.clamp(confidence_target, min=0., max=upper)
                        confidence_else = (1.0 - confidence_target) / (num_classes - 1)
                        onehot = one_hot_embedding(targets, num_classes=num_classes)
                        soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, num_classes) \
                                    + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, num_classes)
                        loss = (1 - correct) * crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
                        loss = torch.mean(loss)

                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                optimizer.zero_grad()
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()

            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

        print('train{} acc:{:.3f}'.format(epoch,top1.avg), end=' ')

        # test
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader['val']):
                inputs, labels = data.to(opt.device), target.to(opt.device)
                unk_mask = None
                outputs = model(inputs, unk_mask=unk_mask)
                _, preds = torch.max(outputs, 1)
                loss = crossentropy(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                test_corrects += preds.eq(target.to(opt.device)).sum().item()
                epoch_loss = test_loss / size['val']
                epoch_acc = 1.0 * test_corrects / size['val']
            # if epoch_acc > 0.7:
            #     torch.save(model.state_dict(), config.path.model_path + '0.0008Relaxlossshadow0.5_30' + '.pth')
            print('val{} acc:{:.3f}'.format(epoch,epoch_acc))
    return model


def train_model(opt, config, data_loader, data_size, is_target = True):
    model = tim.create_model('vit_small_patch16_224', num_classes=opt.n_class)
    model.load_state_dict(torch.load('../vit_small_patch16_224_{}.pth'.format(opt.n_class)))
    model = model.to(opt.device)
    if config.learning.DDP:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    elif config.learning.DP:
        model=torch.nn.DataParallel(model, device_ids=[0,1], output_device=opt.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config.learning.learning_rate,
                                      betas=(config.learning.beta1, config.learning.beta2),
                                      weight_decay=config.learning.weight_decay
                                 )
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    model = Relaxloss_train(model, data_loader, data_size, base_scheduler, optimizer, config, opt)
    pad = "" if is_target else "_shadow"
    if config.learning.DDP or config.learning.DP:
        if opt.local_rank == 0:
            torch.save(model.module.state_dict(), '../defence_model/{}/RelaxLoss/RL_{}{}.pth'.format(opt.dataset, opt.alpha, pad))
    else:
        torch.save(model.state_dict(), '../defence_model/{}/RelaxLoss/RL_{}{}.pth'.format(opt.dataset, opt.alpha, pad))
    return


opt = get_opt()
opt.n_class = dataset_class_dict[opt.dataset]
# torch.random.manual_seed(3407)
config_path = config_dict[opt.dataset]
config = MyConfig.MyConfig(path=config_path)
opt.device = gpu_init(opt)
print("relaxloss dataset : {}  para: {}".format(opt.dataset, opt.alpha))
# config.set_subkey('learning', 'epochs', 0)
data_loader, data_size = get_loader(opt.dataset, config, is_target=opt.istarget)
train_model(opt, config, data_loader, data_size, is_target=opt.istarget)
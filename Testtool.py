import math
import time

import torch
import torchvision
from torch import nn


def gain_data(target_model, shadow_model, target_loader, target_size, shadow_loader, shadow_size, withPE, config, device):
    target_mem = []
    shadow_mem = []
    patches_to_im = torch.nn.Fold(
        output_size=(config.patch.fold_size, config.patch.fold_size),
        kernel_size=(config.patch.fold_patch_size, config.patch.fold_patch_size),
        stride=config.patch.fold_patch_size
    )

    if config.learning.DP:
        target_model = target_model.module
        shadow_model = shadow_model.module
    target_patch_emb = target_model.to_patch_embedding.to(device).requires_grad_(False)
    shadow_patch_emb = shadow_model.to_patch_embedding.to(device).requires_grad_(False)
    target_PE = target_model.pos_embedding.detach().to(device)
    shadow_PE = shadow_model.pos_embedding.detach().to(device)
    for idx, phase in enumerate(['val', 'train']):
        for batch_idx, (data, target) in enumerate(target_loader[phase]):
            inputs, labels = data.to(device), target.to(device)
            outputs = target_patch_emb(inputs)
            if withPE:
                outputs += target_PE[1:target_patch_emb.num_patches+1]
            outputs = patches_to_im(outputs.transpose(1, 2))
            for i in range(len(outputs)):
                target_mem.append([outputs[i], idx])
        for batch_idx, (data, target) in enumerate(shadow_loader[phase]):
            inputs, labels = data.to(device), target.to(device)
            outputs = shadow_patch_emb(inputs)
            if withPE:
                outputs += shadow_PE[1:shadow_patch_emb.num_patches+1]
            outputs = patches_to_im(outputs.transpose(1, 2))
            for i in range(len(outputs)):
                shadow_mem.append([outputs[i], idx])
    target_mem_loader = torch.utils.data.DataLoader(target_mem,
                                                    batch_size=config.learning.batch_size,
                                                    shuffle=True,
                                                    # num_workers=config.general.num_workers
                                                    )
    shadow_mem_loader = torch.utils.data.DataLoader(shadow_mem,
                                                    batch_size=config.learning.batch_size,
                                                    shuffle=True,
                                                    # num_workers=config.general.num_workers
                                                    )
    atk_loader = {'train': shadow_mem_loader, 'val': target_mem_loader}
    atk_size = {'train': shadow_size['train']+shadow_size['val'] , 'val': target_size['train']+target_size['val']}
    return atk_loader, atk_size


def audit_data(atk_loader, atk_size, config, device):

    atk_model = torchvision.models.resnet18(pretrained=False)
    num_ftrs = atk_model.fc.in_features
    atk_model.fc = torch.nn.Linear(num_ftrs, 2)
    atk_model = atk_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(atk_model.parameters(), lr=config.learning.atk_learning_rate,momentum=config.learning.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.learning.atk_epochs, eta_min=config.learning.min_lr)
    best_acc = 0
    since = time.time()
    for epoch in range(config.learning.atk_epochs):
        print('Epoch {}/{}'.format(epoch, config.learning.atk_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                atk_model.train()  # Set model to training mode
                scheduler.step()
            else:
                atk_model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for batch_idx, (data, target) in enumerate(atk_loader[phase]):
                inputs, labels = data.to(device), target.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = atk_model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = torch.squeeze(labels)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += preds.eq(target.to(device)).sum().item()
            epoch_loss = running_loss / atk_size[phase]
            epoch_acc = 1.0 * running_corrects / atk_size[phase]
            if phase == 'train':
                print('train acc:{:.3f}'.format(epoch_acc), end=' ')
            else:
                print('val acc:{:.3f}'.format(epoch_acc))
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
    time_elapsed = time.time() - since
    print("best acc:{:.3f}".format(best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return epoch_acc, best_acc

def conf(target_model, loader, device):
    target_model.eval()  # Set model to evaluate mode
    maxlist = []
    # Iterate over data.
    for batch_idx, (data, target) in enumerate(loader):
        inputs, labels = data.to(device), target.to(device)
        outputs = torch.nn.functional.softmax(target_model(inputs),dim=1)
        out, preds = torch.max(outputs, 1)
        maxlist.append(out.cpu().detach())
        # max = out.max(0)[0]
        # min = out.min(0)[0]
        # if max>max_conf:
        #     max_conf = max
        # if min<min_conf:
        #     min_conf = min
    # print("min_conf:{} max_conf:{}".format(min_conf, max_conf))
    max = torch.cat(maxlist, dim=0)
    mean, var = max.mean(), max.var()
    print("mean:{} var:{}".format(max.mean(), max.var()))
    return mean, var

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[0,:x.size(1), :]
        return self.dropout(x)
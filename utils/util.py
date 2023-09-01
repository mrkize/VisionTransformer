from dataloader import *
import random
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensor_shuffle(x):
    x_s = x[:,torch.randperm(x.shape[1]),:]
    return x_s


def shf_test(model_pos, model_nopos, dataset, PE, isval= False):
    soft = nn.Softmax(1)
    cro = nn.CrossEntropyLoss()
    sum = 0
    loss_sum = 0
    cross_pe, cross = 0, 0
    for batch_idx, (data, target) in enumerate(dataset):
        inputs, labels = data.to(device), target.to(device)
        res_1 = model_pos(inputs)
        res_2 = model_pos(inputs, shuffle=True)
        # loss_1 = cro(res_1,labels)
        # loss_2 = cro(res_2,labels)
        # print('loss change with pos:',cro(res_1,res_2).item())x

        # img_val = torch.unsqueeze(data[i][0], 0).to(device)

        res_3 = model_pos(inputs, pos=False)
        res_4 = model_pos(inputs, shuffle=True, pos=False)
        # loss_3 = cro(res_3,labels)
        # loss_4 = cro(res_4,labels)
        # print('loss change without pos:',cro(res_3,res_4).item())
        cross_pe += F.cross_entropy(soft(res_1), soft(res_2)).item()
        cross += F.cross_entropy(soft(res_3), soft(res_4)).item()
        # if F.cross_entropy(soft(res_1), soft(res_2)).item() > F.cross_entropy(soft(res_3), soft(res_4)).item():
        #     sum += 1
        # if abs(loss_1-loss_2) > abs(loss_3-loss_4):
        #     loss_sum += 1
        # elif isval:
        #     print('loss_1:', loss_1.item(), 'loss_2:', loss_2.item(), 'loss_1-loss_2:', abs(loss_1-loss_2).item())
        #     print('loss_3:', loss_3.item(), 'loss_4:', loss_4.item(), 'loss_3-loss_4:', abs(loss_3-loss_4).item())
        #     print('---------------------------------------------------------')

    return sum / (batch_idx+1), loss_sum/(batch_idx+1), cross_pe/(batch_idx+1),cross/(batch_idx+1)

def pos_emb_shuffle_test(root_dir):
    train_data, train_loader = model_dataloader(root_dir=root_dir, spilt='train')
    val_data, val_loader = model_dataloader(root_dir=root_dir, spilt='val')
    data_loader = {'train': train_loader, 'val': val_loader}
    data_size = {"train": len(train_data), "val": len(val_data)}

    # model_pos, ret_para = train(data_loader, data_size, config, PE=True)
    # model_nopos, ret_para = train(data_loader, data_size, config, PE=False)
    model_pos = torch.load('./results/Pos/VIT_pos.pth').to(device)
    model_nopos = torch.load('./results/NoPos/VIT_nopos.pth').to(device)
    b = model_pos.pos_emb()
    train_gap, train_loss_gap, cross_pe_train, cross_train = shf_test(model_pos, model_nopos, train_loader, b)
    print('training data cross with PE:', cross_pe_train)
    print('training data cross without PE:', cross_pe_train)
    val_gap, val_loss_gap, cross_pe_val, cross_val = shf_test(model_pos, model_nopos, val_loader, b)
    print('val data cross with PE:', cross_pe_val)
    print('val data cross without PE:', cross_pe_val)
    # print('train gap:', train_gap)
    # print('val gap:', val_gap)
    # print('train loss gap:', train_loss_gap)
    # print('val loss gap:', val_loss_gap)
    return
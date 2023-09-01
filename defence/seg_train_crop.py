import numpy as np
import torch
from torch import nn
import sys
sys.path.append('.')
from .seg_train import LabelSmoothingCrossEntropy

CROP_SIZE = 128


def get_crop(img, lbl, random=True, x=0, y=0):
    if random:
        x = np.random.randint(0, img.shape[2] - CROP_SIZE, size=1)[0]
        y = np.random.randint(0, img.shape[3] - CROP_SIZE, size=1)[0]

    img = img[:,:,x:x+CROP_SIZE, y:y+CROP_SIZE]
    lbl = lbl[:,x:x+CROP_SIZE, y:y+CROP_SIZE]

    return img, lbl
    

def crop_stiching(img, lbl, seg_model, paser):
    # PREDICTED MASK CROPS STICHING
    pred_full = torch.zeros(lbl.shape).to(paser.device)
    for x in range(0,img.shape[3],CROP_SIZE):
        for y in range(0,img.shape[3],CROP_SIZE):
            data_crop, _ = get_crop(img, lbl, random=False, x=x, y=y)
            pred_full[:,x:x+CROP_SIZE,y:y+CROP_SIZE] = seg_model(data_crop)[:,0,:,:]

    pred_full = pred_full.view(img.shape[0],1,img.shape[2],img.shape[3])

    return pred_full


def train_segmentation_model_crop(model, dataloader, val_dataloader, epochs, config, paser):
    criterion = LabelSmoothingCrossEntropy()
    model = model.to(paser.device)

    opt = torch.optim.Adam(model.parameters(),
                           lr=config.learning.learning_rate,
                           betas=(config.learning.beta1, config.learning.beta2),
                           weight_decay=config.learning.weight_decay
                           )
    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = img.to(paser.device), lbl.to(paser.device)
            img, lbl = get_crop(img, lbl)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred, lbl)
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))

        if epoch % 10 == 0: validate_segmentation_model_crop(model, val_dataloader, paser)

    validate_segmentation_model_crop(model, val_dataloader,paser)

    return model, train_loss


def validate_segmentation_model_crop(model, dataloader, paser):
    criterion = nn.CrossEntropyLoss()

    val_loss_data = []
    running_corrects = 0
    size = 0
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(paser.device), lbl.to(paser.device)
            size += img.shape[0]
            pred = model(img)
            _, preds = torch.max(pred, 1)
            running_corrects += preds.eq(lbl).sum().item()
            # PREDICTED MASK CROPS STICHING
            pred_full = crop_stiching(img, lbl, model, paser)

            loss = criterion(pred_full, lbl)
            val_loss_data.append(loss.item())

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)
    acc = 1.0 * running_corrects / size
    print('Validation loss:', round(val_loss, 4))
    print('Validation acc:', acc)
    
    return val_loss, acc
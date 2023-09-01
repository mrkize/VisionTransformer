import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 < smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        y_hat = torch.softmax(x, dim=1)
        cross_loss = self.cross_entropy(y_hat, target)
        smooth_loss = -torch.log(y_hat).mean(dim=1)
        loss = self.confidence * cross_loss + self.smoothing * smooth_loss
        return loss.mean()

    def cross_entropy(self, y_hat, y):
        return - torch.log(y_hat[range(y_hat.shape[0]), y])


def validate_segmentation_model(model, dataloader, paser):
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
            loss = criterion(pred, lbl)
            val_loss_data.append(loss.item())
            running_corrects += preds.eq(lbl).sum().item()

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)
    acc = 1.0 * running_corrects / size
    print('Validation loss:', round(val_loss,4))
    print('Validation acc:', acc)
    
    return val_loss
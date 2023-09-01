import torch
import opacus
import numpy as np
import warmup_scheduler
from torch import nn
import sys
sys.path.append('.')
from .seg_train import validate_segmentation_model, LabelSmoothingCrossEntropy
import torch.nn.functional as F
from opacus import GradSampleModule


def get_dp_model(model, dataloader, config, delta_inv, paser):

    model = opacus.validators.ModuleValidator.fix(model).to(paser.device)
    print('Opacus validation:', opacus.validators.ModuleValidator.validate(model, strict=True))

    opt = torch.optim.Adam(model.parameters(),
                           lr=config.learning.learning_rate,
                           betas=(config.learning.beta1, config.learning.beta2),
                           weight_decay=config.learning.weight_decay
                           )
    privacy_engine = opacus.PrivacyEngine()

    model, dp_opt, dp_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=dataloader,
        target_delta=1/delta_inv,
        target_epsilon=8.5,
        epochs=config.learning.epochs,
        max_grad_norm=2.0,
    )

    return model, dp_opt, dp_dataloader


def train_segmentation_model_dp(delta_inv, model, dataloader, val_dataloader, epochs, config, paser):

    model, opt, dataloader = get_dp_model(model, dataloader, config, delta_inv, paser)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                                T_max=config.learning.epochs,
                                                                eta_min=config.learning.min_lr)
    exp_lr_scheduler = warmup_scheduler.GradualWarmupScheduler(opt, multiplier=1.,
                                                        total_epoch=config.learning.warmup_epoch,
                                                        after_scheduler=base_scheduler)
    criterion = LabelSmoothingCrossEntropy()

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            exp_lr_scheduler.step()
            if img.shape[0] == 0: continue

            img, lbl = img.to(paser.device), lbl.to(paser.device)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred, lbl)
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        train_loss = np.sum(np.array(train_loss_data))/len(train_loss_data)
        print('Training loss:', round(train_loss,4))

        if epoch % 10 == 0: validate_segmentation_model(model, val_dataloader,paser)

    validate_segmentation_model(model, val_dataloader, paser)

    return model, train_loss
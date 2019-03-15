# -*- coding: utf-8 -*-

import numpy as np

import torch

import tqdm


def train_epoch(train_loader, model, loss_fn, optimizer, scheduler=None, tqdm_module=tqdm.tqdm, disable_tqdm=False):
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in tqdm_module(enumerate(train_loader), total=len(train_loader), leave=False, disable=disable_tqdm):
        if not (type(data) in [list, tuple]):
            data = (data,)

        data = (d.cuda() for d in data)
        target = target.type(torch.float).flatten().cuda()

        optimizer.zero_grad()
        outputs = model(*data).flatten()

        loss = loss_fn(outputs, target)

        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()

        optimizer.step()

        if scheduler:
            scheduler.batch_step()

    total_loss /= (batch_idx + 1)
    return total_loss


def test_epoch(val_loader, model, loss_fn):
    with torch.no_grad():
        model.eval()

        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            if not (type(data) in [list, tuple]):
                data = (data,)

            data = (d.cuda() for d in data)
            target = target.type(torch.float).flatten().cuda()

            outputs = model(*data).flatten()
            loss = loss_fn(outputs, target)

            val_loss += loss.item()

    val_loss /= (batch_idx + 1)
    return val_loss


def predict(model, loader):
    res = list()

    with torch.no_grad():
        model.eval()

        for (data, target) in loader:
            if not (type(data) in [list, tuple]):
                data = (data,)

            data = (d.cuda() for d in data)
            outputs = model(*data).cpu().numpy()
            res.append(outputs)

    return np.vstack(np.array(res))
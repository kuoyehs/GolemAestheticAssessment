# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

# import lrs
import tensorboardX
from torch.nn import MSELoss

from data_loader import AVADataset

from model import EMDLoss, NIMA

from tensorboardX import SummaryWriter


def getName(prefix):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(prefix, current_time + '_' + socket.gethostname())
    return log_dir


writer = SummaryWriter(getName("/data/output/"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_acc(label: torch.Tensor, pred: torch.Tensor):
    dist = torch.arange(10).float().to(device)
    p_mean = (pred.view(-1, 10) * dist).sum(dim=1)
    l_mean = (label.view(-1, 10) * dist).sum(dim=1)
    p_good = p_mean > 5
    l_good = l_mean > 5
    acc = (p_good == l_good).float().mean()
    return acc


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor()])

    trainset = AVADataset(csv_file=config.train_csv_file, root_dir=config.train_img_path, transform=train_transform)
    valset = AVADataset(csv_file=config.val_csv_file, root_dir=config.val_img_path, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=config.val_batch_size,
                                             shuffle=False, num_workers=config.num_workers)

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    if config.warm_start:
        model.load_state_dict(torch.load(os.path.join(config.ckpt_path, 'epoch-%d.pkl' % config.warm_start_epoch)))
        print('Successfully loaded model epoch-%d.pkl' % config.warm_start_epoch)

    if config.multi_gpu:
        model.features = torch.nn.DataParallel(model.features, device_ids=config.gpu_ids)
        model = model.to(device)
    else:
        model = model.to(device)

    conv_base_lr = config.conv_base_lr
    dense_lr = config.dense_lr
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
    )
    # optimizer = optim.Adam(model.parameters())
    writer.add_text("optimizer", str(optimizer))

    # # send hyperparams
    # lrs.send({
    #     'title': 'EMD Loss',
    #     'train_batch_size': config.train_batch_size,
    #     'val_batch_size': config.val_batch_size,
    #     'optimizer': 'SGD',
    #     'conv_base_lr': config.conv_base_lr,
    #     'dense_lr': config.dense_lr,
    #     'momentum': 0.9
    #     })

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if config.train:
        # for early stopping
        count = 0
        step = 0
        init_val_loss = float('inf')
        train_losses = []
        val_losses = []
        emd_loss_func = EMDLoss()
        mse_loss_func = MSELoss()
        for epoch in range(config.warm_start_epoch, config.epochs):
            model.train()
            # lrs.send('epoch', epoch)
            batch_emb_losses = []
            for i, data in enumerate(train_loader):
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                outputs = model(images)
                step += 1
                outputs = outputs.view(-1, 10, 1)

                optimizer.zero_grad()

                emd_loss_value = emd_loss_func(labels, outputs)
                dist = torch.arange(10).float().to(device)
                var_loss_value = mse_loss_func(labels.var(dim=1), outputs.var(dim=1))
                p_mean = (outputs.view(-1, 10) * dist).sum(dim=1)
                l_mean = (labels.view(-1, 10) * dist).sum(dim=1)
                mean_loss_value = mse_loss_func(p_mean, l_mean)

                loss = emd_loss_value + var_loss_value + mean_loss_value

                batch_emb_losses.append(emd_loss_value.item())

                loss.backward()
                # import ipdb; ipdb.set_trace()
                optimizer.step()

                # lrs.send('train_emd_loss', loss.item())
                writer.add_scalar('train/emd_loss', emd_loss_value, step)
                writer.add_scalar('train/accuracy', compute_acc(labels, outputs), step)

                print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size + 1,
                    emd_loss_value.data[0]))

            avg_loss = sum(batch_emb_losses) / (len(trainset) // config.train_batch_size + 1)
            train_losses.append(avg_loss)
            writer.add_scalar('train/avg_loss', avg_loss, step)
            print('Epoch %d averaged training EMD loss: %.4f' % (epoch + 1, avg_loss))

            # exponetial learning rate decay
            if (epoch + 1) % 10 == 0:
                conv_base_lr = conv_base_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                dense_lr = dense_lr * config.lr_decay_rate ** ((epoch + 1) / config.lr_decay_freq)
                optimizer = optim.SGD([
                    {'params': model.features.parameters(), 'lr': conv_base_lr},
                    {'params': model.classifier.parameters(), 'lr': dense_lr}],
                    momentum=0.9
                )

                # optimizer = optim.Adam(model.parameters())
                # writer.add_text("optimizer", str(optimizer))

                # send decay hyperparams
                # lrs.send({
                #     'lr_decay_rate': config.lr_decay_rate,
                #     'lr_decay_freq': config.lr_decay_freq,
                #     'conv_base_lr': config.conv_base_lr,
                #     'dense_lr': config.dense_lr
                #     })

            # do validation after each epoch
            batch_val_losses = []
            val_acc = []
            for data in val_loader:
                model.eval()
                images = data['image'].to(device)
                labels = data['annotations'].to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                step += 1
                outputs = outputs.view(-1, 10, 1)
                val_loss = emd_loss_func(labels, outputs)
                batch_val_losses.append(val_loss.item())
                val_acc.append(compute_acc(labels, outputs))
            avg_val_loss = sum(batch_val_losses) / (len(valset) // config.val_batch_size + 1)
            val_losses.append(avg_val_loss)

            writer.add_scalar('val/emb_loss', avg_val_loss, step)
            writer.add_scalar('val/accuracy', np.mean(val_acc), step)
            # lrs.send('val_emd_loss', avg_val_loss)

            print('Epoch %d completed. Averaged EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))

            # Use early stopping to monitor training
            if avg_val_loss < init_val_loss:
                init_val_loss = avg_val_loss
                # save model weights if val loss decreases
                print('Saving model...')
                torch.save(model.state_dict(), os.path.join(config.ckpt_path, 'epoch-%d.pkl' % (epoch + 1)))
                print('Done.\n')
                # reset count
                count = 0
            elif avg_val_loss >= init_val_loss:
                count += 1
                if count == config.early_stopping_patience:
                    print(
                        'Val EMD loss has not decreased in %d epochs. Training terminated.' % config.early_stopping_patience)
                    break

        print('Training completed.')

        if config.save_fig:
            # plot train and val loss
            epochs = range(1, epoch + 2)
            # plt.plot(epochs, train_losses, 'b-', label='train loss')
            # plt.plot(epochs, val_losses, 'g-', label='val loss')
            # plt.title('EMD loss')
            # plt.legend()
            # plt.savefig('./loss.png')

    if config.test:
        # compute mean score
        test_transform = val_transform
        testset = AVADataset(csv_file=config.test_csv_file, root_dir=config.test_img_path, transform=val_transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False,
                                                  num_workers=config.num_workers)

        mean_preds = []
        std_preds = []
        for data in test_loader:
            image = data['image'].to(device)
            output = model(image)
            output = output.view(10, 1)
            predicted_mean, predicted_std = 0.0, 0.0
            for i, elem in enumerate(output, 1):
                predicted_mean += i * elem
            for j, elem in enumerate(output, 1):
                predicted_std += elem * (j - predicted_mean) ** 2
            mean_preds.append(predicted_mean)
            std_preds.append(predicted_std)
        # Do what you want with predicted and std...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--train_img_path', type=str, default='/data/data/images/')
    parser.add_argument('--val_img_path', type=str, default='/data/data/images/')
    parser.add_argument('--test_img_path', type=str, default='/data/data/images/')
    parser.add_argument('--train_csv_file', type=str, default='/data/data/train.csv')
    parser.add_argument('--val_csv_file', type=str, default='/data/data/val.csv')
    parser.add_argument('--test_csv_file', type=str, default='/data/data/test.csv')

    # training parameters
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--conv_base_lr', type=float, default=1e-3)
    parser.add_argument('--dense_lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_freq', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)

    # misc
    parser.add_argument('--ckpt_path', type=str, default='/data/output/')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--warm_start', type=bool, default=False)
    parser.add_argument('--warm_start_epoch', type=int, default=0)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--save_fig', type=bool, default=False)

    # config = parser.parse_args()
    config, unknown = parser.parse_known_args()
    writer.add_text("Config", str(config))

    main(config)

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.transforms as transforms

from losses import FocalLoss, mIoULoss
from model import UNet
from dataloader import AerialDataset


def fetch_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./dataset', help='path to your dataset')
    parser.add_argument('--out_dir', type=str, default='./saved_models', help='path to your output directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch', type=int, default=2, help='batch size')
    parser.add_argument('--loss', type=str, default='focalloss', help='focalloss | iouloss | crossentropy')
    return parser.parse_args()


def cal_accuracy(label, predicted):
    """
    Calculate accuracy
    @param label: label tensor
    @param predicted: predicted tensor
    @return: float seg_acc - accuracy
    """
    seg_acc = (label.cpu() == torch.argmax(predicted, axis=1).cpu()).sum() / torch.numel(label.cpu())
    return seg_acc


def plot_losses(losses, out_fig='./loss_plots.png'):
    """
    Plot training and validation losses
    @param list losses: [epoch, mean_training_loss, mean_validation_loss]
    @param str out_fig: output figure
    @return:
    """
    if isinstance(losses, list):
        losses = np.array(losses)

    plt.figure(figsize=(12, 8))
    plt.plot(losses[:, 0], losses[:, 1], color='b', linewidth=4)
    plt.plot(losses[:, 0], losses[:, 2], color='r', linewidth=4)
    plt.title(args.loss, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.grid()
    plt.legend(['training', 'validation'])  # using a named size
    plt.savefig(out_fig)


if __name__ == '__main__':
    args = fetch_args()

    color_shift = transforms.ColorJitter(.1, .1, .1, .1)
    # blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    dataset = AerialDataset(args.data,
                         train=True,
                         transform=transforms.Compose([color_shift]))  #, blurriness]))

    print('Number of images : ' + str(len(dataset)))

    test_num = int(0.1 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num],
                                                                generator=torch.Generator().manual_seed(101))
    print(f'Training: {len(train_dataset)}\nValidation: {len(test_dataset)}')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.loss == 'focalloss':
        criterion = FocalLoss(gamma=3.0/4).to(device)
    elif args.loss == 'iouloss':
        criterion = mIoULoss(n_classes=6).to(device)
    elif args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        print('Loss function not found!')

    model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    losses = []
    scheduler_counter = 0

    for epoch in range(args.num_epochs):
        # training
        model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):

            pred_mask = model(x.to(device))
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(cal_accuracy(y,pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch + 1,
                    args.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1

        # testing
        model.eval()
        val_loss_list = []
        val_acc_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                pred_mask = model(x.to(device))
            val_loss = criterion(pred_mask, y.to(device))
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(cal_accuracy(y, pred_mask).numpy())

        print(' epoch {} - loss : {:.5f} - accuracy : {:.2f} - val loss : {:.5f} - val accuracy : {:.2f}'.format(
            epoch + 1,
            np.mean(loss_list),
            np.mean(acc_list),
            np.mean(val_loss_list),
            np.mean(val_acc_list)))

        losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        if is_best:
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, 'epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0

        plot_losses(losses, out_fig=os.path.join(args.out_dir, 'loss_plots.png'))

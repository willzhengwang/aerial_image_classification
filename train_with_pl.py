# import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tensorboard import program

from dataloader import AerialDataset, NAME_CLASSES
from seg_module import train_model

if __name__ == '__main__':
    data_set = AerialDataset(root='/quobyte/dev_rnd/zwang/deep_learning/aerial_imagery/dataset', train=True)
    pl.seed_everything(42)
    n_train = int(len(data_set) * 0.9)
    train_set, val_set = torch.utils.data.random_split(data_set, [n_train, len(data_set) - n_train])
    # print(len(train_set))
    # img, label = train_set[0]
    # img = torch.mul(img, 255).byte()
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(img.permute(1, 2, 0))
    # axarr[1].imshow(label)
    # plt.show()

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(f"Number of GPUs: {len(available_gpus)}")

    unet_model, unet_results = train_model(
        "UNet",
        train_loader,
        val_loader,
        gpus=len(available_gpus) - 2,
        max_epochs=150,
        model_hparams={"num_classes": len(NAME_CLASSES), "input_channels": 3},
        loss="focalloss",  # crossentropy
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3})

    print("unet_model", unet_results)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'saved_models/UNet/lightning_logs'])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    print('Testing is done!')

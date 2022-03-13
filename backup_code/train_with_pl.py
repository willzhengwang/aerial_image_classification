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

    # model networks - model parameters
    model = "R2AttU_Net"
    if model == 'UNet':
        batch_size = 4
        model_params = {"num_classes": len(NAME_CLASSES), "input_channels": 3}
    elif model in "R2U_Net":
        batch_size = 1  # batch size - 2 out of memory
        model_params = {"output_ch": len(NAME_CLASSES)}
    elif model == "AttU_Net":
        batch_size = 2
        model_params = {"output_ch": len(NAME_CLASSES)}
    elif model == "R2AttU_Net":
        batch_size = 1
        model_params = {"output_ch": len(NAME_CLASSES)}
    # Notes: with the same number of epochs, UNet and AttU_Net have much better performance than R2U_Net and R2AttU_Net.

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=batch_size)
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print(f"Number of GPUs: {len(available_gpus)}")

    unet_model, unet_results = train_model(
        model,
        train_loader,
        val_loader,
        gpus=max(len(available_gpus) - 1, 0),
        max_epochs=80,
        model_hparams=model_params,
        loss="crossentropy",  # "focalloss"
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3})
    print(model, unet_results)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'saved_models/{}/lightning_logs'.format(model)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    print('Testing is done!')

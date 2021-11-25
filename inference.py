import argparse
import logging
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import UNet


def get_output_files(input_files):
    def _generate_output_file(input_file):
        split = os.path.splitext(input_file)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_output_file, input_files))


def mask_to_image(mask: np.ndarray):
    """

    @param mask:
    return:
    """
    BGR_classes = {'Water': [41, 169, 226],
                        'Land': [246, 41, 132],
                        'Road': [228, 193, 110],
                        'Building': [152, 16, 60],
                        'Vegetation': [58, 221, 254],
                        'Unlabeled': [155, 155, 155]}  # in BGR
    bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

    if mask.ndim == 2:
        return (mask * 255).astype(np.uint8)
    elif mask.ndim == 3:
        vis_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
        cls_mask = np.argmax(mask, axis=0)
        for i, class_name in enumerate(bin_classes):
            rows, cols = np.where(cls_mask == i)
            vis_mask[rows, cols, :] = BGR_classes[class_name]
        return vis_mask


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def predict_img(net,
                img_arr,
                device,
                out_threshold=0.5):
    """

    @param net: network object
    @param img_arr: numpy image array. [rows, cols, channels]
    @param device:
    @param out_threshold:
    return: masks - [classes, rows, cols] for the predicted results
    """

    net.eval()
    img = torch.from_numpy(np.moveaxis(img_arr / 255.0, -1, 0))
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = net(img.to(device=device, dtype=torch.float32))

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        full_mask = probs.cpu().squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def fetch_args():
    parser = argparse.ArgumentParser(description="Label each pixel of input images with a corresponding class")
    parser.add_argument('--model', '-m', default='../../models/MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    # parser.add_argument('--scale', '-s', type=float, default=0.5,
    #                     help='Scale factor for the input images')

    return parser.parse_args()


if __name__ == '__main__':
    args = fetch_args()
    in_files = args.input
    out_files = get_output_files(in_files)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=6, bilinear=True).to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, file_path in enumerate(in_files):
        logging.info(f'\nPredicting image {i+1}/{len(in_files)}: {file_path} ...')

        img = cv2.imread(file_path)
        # TODO: divide image into blocks for processing
        mask = predict_img(net, cv2.resize(img, (512, 512)), device, args.mask_threshold)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            cv2.imwrite(out_filename, cv2.resize(result, (img.shape[1], img.shape[0])))
            logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {file_path}, close to continue...')
        #     plot_img_and_mask(img, mask)

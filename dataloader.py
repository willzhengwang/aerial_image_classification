import cv2
from cv2 import transform
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from scipy import ndimage
from glob import glob

BGR_CLASSES = {'Water': [41, 169, 226],
               'Land': [246, 41, 132],
               'Road': [228, 193, 110],
               'Building': [152, 16, 60],
               'Vegetation': [58, 221, 254],
               'Unlabeled': [155, 155, 155]}  # in BGR

NAME_CLASSES = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']


class AerialDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(AerialDataset, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.img_files = sorted(glob(self.root + '/*/images/*.jpg'))

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cls_mask = np.zeros(mask.shape)  
        cls_mask[mask == BGR_CLASSES['Water']] = NAME_CLASSES.index('Water')
        cls_mask[mask == BGR_CLASSES['Land']] = NAME_CLASSES.index('Land')
        cls_mask[mask == BGR_CLASSES['Road']] = NAME_CLASSES.index('Road')
        cls_mask[mask == BGR_CLASSES['Building']] = NAME_CLASSES.index('Building')
        cls_mask[mask == BGR_CLASSES['Vegetation']] = NAME_CLASSES.index('Vegetation')
        cls_mask[mask == BGR_CLASSES['Unlabeled']] = NAME_CLASSES.index('Unlabeled')
        cls_mask = cls_mask[:, :, 0]

        if self.train:
            if self.transform:
              image = transforms.functional.to_pil_image(image)
              image = self.transform(image)
              image = np.array(image)

            # # 90 degree rotation
            # if np.random.rand()<0.5:
            #   angle = np.random.randint(4) * 90
            #   image = ndimage.rotate(image,angle,reshape=True)
            #   cls_mask = ndimage.rotate(cls_mask,angle,reshape=True)

            # # vertical flip
            # if np.random.rand()<0.5:
            #   image = np.flip(image, 0)
            #   cls_mask = np.flip(cls_mask, 0)
            
            # # horizonal flip
            # if np.random.rand()<0.5:
            #   image = np.flip(image, 1)
            #   cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512,512))/255.0
        cls_mask = cv2.resize(cls_mask, (512,512)) 
        image = np.moveaxis(image, -1, 0)
        
        # for testing
        img0 = read_image(img_path)
        img1 = torch.tensor(image).float()

        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

    def __len__(self):
        return len(self.img_files)
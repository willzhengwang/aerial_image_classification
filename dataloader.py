import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
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
        image = cv2.resize(image, (512,512))/255.0
        cls_mask = cv2.resize(cls_mask, (512,512)) 
        image = np.moveaxis(image, -1, 0).astype(np.float32)
        

        if self.train:
            if self.transform:
              image = transforms.functional.to_pil_image(image)
              cls_mask = transforms.functional.to_pil_image(cls_mask)
              image = self.transform(image)
              cls_mask = self.transform(cls_mask)
        
        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

    def __len__(self):
        return len(self.img_files)
    
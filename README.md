# aerial_image_classification
Semantic segmentation of aerial images with Pytorch - UNet

## Dataset

The used dataset is available at [Semantic segmentation of aerial imagery](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery).

The dataset consists of 72 aerial images of Dubai acquired by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes:

Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

## Train
```
!python train.py --data <dataset_path> --num_epochs 10 --batch 2 --loss focalloss
```

## Inference
```
!python inference.py --model <model_path> --input <input_files> 
```
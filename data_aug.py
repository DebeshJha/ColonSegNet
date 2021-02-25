
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from utils import create_dir
from data import load_data

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    RandomCrop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    ChannelShuffle,
    CoarseDropout
)

def augment_data(images, masks, save_path, augment=True):
    """ Performing data augmentation. """
    size = (512, 512)
    crop_size = (448, 448)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        image_name = x.split("/")[-1].split(".")[0]
        mask_name = y.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        if x.shape[0] >= size[0] and x.shape[1] >= size[1]:
            if augment == True:
                ## Crop
                x_min = 0
                y_min = 0
                x_max = x_min + size[0]
                y_max = y_min + size[1]

                aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
                augmented = aug(image=x, mask=y)
                x1 = augmented['image']
                y1 = augmented['mask']

                # Random Rotate 90 degree
                aug = RandomRotate90(p=1)
                augmented = aug(image=x, mask=y)
                x2 = augmented['image']
                y2 = augmented['mask']

                ## ElasticTransform
                aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                augmented = aug(image=x, mask=y)
                x3 = augmented['image']
                y3 = augmented['mask']

                ## Grid Distortion
                aug = GridDistortion(p=1)
                augmented = aug(image=x, mask=y)
                x4 = augmented['image']
                y4 = augmented['mask']

                ## Optical Distortion
                aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
                augmented = aug(image=x, mask=y)
                x5 = augmented['image']
                y5 = augmented['mask']

                ## Vertical Flip
                aug = VerticalFlip(p=1)
                augmented = aug(image=x, mask=y)
                x6 = augmented['image']
                y6 = augmented['mask']

                ## Horizontal Flip
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x, mask=y)
                x7 = augmented['image']
                y7 = augmented['mask']

                ## Grayscale
                x8 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                y8 = y

                aug = RGBShift(p=1)
                augmented = aug(image=x, mask=y)
                x9 = augmented['image']
                y9 = augmented['mask']

                aug = ChannelShuffle(p=1)
                augmented = aug(image=x, mask=y)
                x10 = augmented['image']
                y10 = augmented['mask']

                aug = CoarseDropout(p=1, max_holes=10, max_height=32, max_width=32)
                augmented = aug(image=x, mask=y)
                x11 = augmented['image']
                y11 = augmented['mask']

                aug = GaussNoise(p=1)
                augmented = aug(image=x, mask=y)
                x12 = augmented['image']
                y12 = augmented['mask']

                images = [
                    x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12
                ]
                masks  = [
                    y, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12
                ]

            else:
                images = [x]
                masks  = [y]

            idx = 0
        for i, m in zip(images, masks):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            if len(images) == 1:
                tmp_image_name = f"{image_name}.jpg"
                tmp_mask_name  = f"{mask_name}.jpg"
            else:
                tmp_image_name = f"{image_name}_{idx}.jpg"
                tmp_mask_name  = f"{mask_name}_{idx}.jpg"

            image_path = os.path.join(save_path, "image/", tmp_image_name)
            mask_path  = os.path.join(save_path, "mask/", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def main():
    np.random.seed(42)

    path = "/home/nikhilroxtomar/lab/DATA/Kvasir-SEG/"
    # path = "/media/nikhil/ML/ml_dataset/Kvasir-SEG/"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    print("Train: ", len(train_x))
    print("Valid: ", len(test_x))

    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    augment_data(train_x, train_y, "new_data/train/", augment=False)
    augment_data(test_x, test_y, "new_data/test/", augment=False)

if __name__ == "__main__":
    main()

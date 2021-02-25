
import os
import numpy as np
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".jpg" for name in data]
    masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
    return images, masks

def load_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)

class KvasirDataset(Dataset):
    """ Dataset for the Kvasir-SEG dataset. """
    def __init__(self, images_path, masks_path, size):
        """
        Arguments:
            images_path: A list of path of the images.
            masks_path: A list of path of the masks.
        """

        self.images_path = images_path
        self.masks_path = masks_path
        self.height = size[0]
        self.width = size[1]
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image and mask. """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        """ Resizing. """
        image1 = cv2.resize(image, (self.width, self.height))
        # image2 = cv2.resize(image, (self.width//2, self.height//2))
        # image3 = cv2.resize(image, (self.width//4, self.height//4))
        mask = cv2.resize(mask, (self.width, self.height))

        """ Proper channel formatting. """
        image1 = np.transpose(image1, (2, 0, 1))
        # image2 = np.transpose(image2, (2, 0, 1))
        # image3 = np.transpose(image3, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        """ Normalization. """
        image1 = image1/255.0
        # image2 = image2/255.0
        # image3 = image3/255.0
        mask = mask/255.0

        """ Changing datatype to float32. """
        image1 = image1.astype(np.float32)
        # image2 = image2.astype(np.float32)
        # image3 = image3.astype(np.float32)
        mask = mask.astype(np.float32)

        """ Changing numpy to tensor. """
        image1 = torch.from_numpy(image1)
        # image2 = torch.from_numpy(image2)
        # image3 = torch.from_numpy(image3)
        mask = torch.from_numpy(mask)

        return image1, mask

    def __len__(self):
        return self.n_samples

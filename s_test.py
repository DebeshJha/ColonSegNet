import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch

from model import CompNet
from utils import create_dir, seeding, make_channel_last
from crf import apply_crf

def load_paths(kvasir_path, file_path):
    with open(file_path, "r") as f:
        data = f.read().strip()

    images_path = []
    masks_path = []

    for line in data.split("\n"):
        image_name = line.split("/")[-1]
        image_path = os.path.join(kvasir_path, "images", image_name)
        images_path.append(image_path)

        mask_path = os.path.join(kvasir_path, "masks", image_name)
        masks_path.append(mask_path)

    return images_path, masks_path

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("figures/sharib/mask_512/")
    create_dir("figures/sharib/mask_original/")
    create_dir("figures/sharib/bbox")

    """ Load dataset """
    kvasir_path = "/media/nikhil/ML/ml_dataset/Kvasir-SEG/"
    file_path = "figures/kvasir_valid.txt"
    images_path, masks_path = load_paths(kvasir_path, file_path)

    """ Hyperparameters """
    size = (512, 512)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Prediction """
    for x, y in tqdm(zip(images_path, masks_path), total=len(images_path)):
        name = x.split("/")[-1].split(".")[0]

        """ Read Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_img = image
        ori_h, ori_w, _ = image.shape

        image = cv2.resize(image, size)
        resize_img = image

        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Reading Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        """ Predicting Mask """
        with torch.no_grad():
            pred = torch.sigmoid(model(image))
            pred = pred[0].cpu().numpy()
            pred = np.squeeze(pred, axis=0)
            pred = pred > 0.5
            pred = pred.astype(np.int32)
            pred = apply_crf(resize_img, pred)
            pred = pred * 255
            pred = np.array(pred, dtype=np.uint8)


        """ Saving the mask """
        save_path = f"figures/sharib/mask_512/{name}.png"
        cv2.imwrite(save_path, pred)

        pred = cv2.resize(pred, (ori_w, ori_h))
        save_path = f"figures/sharib/mask_original/{name}.png"
        cv2.imwrite(save_path, pred)

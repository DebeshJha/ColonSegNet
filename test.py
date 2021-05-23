
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import (
    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)

from model import CompNet
from utils import create_dir, seeding, make_channel_last
from data import load_data
from crf import apply_crf

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jaccard_score(y_true, y_pred, average='binary')
    score_f1 = f1_score(y_true, y_pred, average='binary')
    score_recall = recall_score(y_true, y_pred, average='binary')
    score_precision = precision_score(y_true, y_pred, average='binary', zero_division=1)
    score_acc = accuracy_score(y_true, y_pred)
    score_fbeta = fbeta_score(y_true, y_pred, beta=1.0, average='binary', zero_division=1)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    path = "/../../Kvasir-SEG/"
    (train_x, train_y), (test_x, test_y) = load_data(path)

    """ Hyperparameters """
    size = (512, 512)
    checkpoint_path = "files/checkpoint.pth"

    """ Directories """
    create_dir("results/mix")
    create_dir("results/mask")

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CompNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Testing """
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in enumerate(zip(test_x, test_y)):
        name = y.split("/")[-1].split(".")[0]

        ## Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)

        image1 = cv2.resize(image, size)
        ori_img1 = image1
        image1 = np.transpose(image1, (2, 0, 1))
        image1 = image1/255.0
        image1 = np.expand_dims(image1, axis=0)
        image1 = image1.astype(np.float32)
        image1 = torch.from_numpy(image1)
        image1 = image1.to(device)

        ## Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        mask1 = cv2.resize(mask, size)
        ori_mask1 = mask1
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = mask1/255.0
        mask1 = np.expand_dims(mask1, axis=0)
        mask1 = mask1.astype(np.float32)
        mask1 = torch.from_numpy(mask1)
        mask1 = mask1.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            pred_y1 = torch.sigmoid(model(image1))
            end_time = time.time() - start_time
            time_taken.append(end_time)
            print("{} - {:.10f}".format(name, end_time))

            """ Evaluation metrics """
            score = calculate_metrics(mask1, pred_y1)
            metrics_score = list(map(add, metrics_score, score))

            """ Predicted Mask """
            pred_y1 = pred_y1[0].cpu().numpy()
            pred_y1 = np.squeeze(pred_y1, axis=0)
            pred_y1 = pred_y1 > 0.5
            pred_y1 = pred_y1.astype(np.int32)
            pred_y1 = apply_crf(ori_img1, pred_y1)
            pred_y1 = pred_y1 * 255
            # pred_y = np.transpose(pred_y, (1, 0))
            pred_y1 = np.array(pred_y1, dtype=np.uint8)

        ori_img1 = ori_img1
        ori_mask1 = mask_parse(ori_mask1)
        pred_y1 = mask_parse(pred_y1)
        sep_line = np.ones((size[0], 10, 3)) * 255

        tmp = [
            ori_img1, sep_line,
            ori_mask1, sep_line,
            pred_y1
        ]

        cat_images = np.concatenate(tmp, axis=1)
        cv2.imwrite(f"results/mix/{name}.png", cat_images)
        cv2.imwrite(f"results/mask/{name}.png", pred_y1)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)

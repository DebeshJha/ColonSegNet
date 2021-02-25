
import os
import time
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

def shuffling(x, y):
    """ Shuffle the dataset. """
    x, y = shuffle(x, y, random_state=42)
    return x, y

def make_channel_last(x):
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 2, 3, 1))
    elif len(x.shape) == 3:
        x = np.transpose(x, (1, 2, 0))
    return x

def make_channel_first(x):
    if len(x.shape) == 4:
        x = np.transpose(x, (0, 3, 1, 2))
    elif len(x.shape) == 3:
        x = np.transpose(x, (1, 2, 0))
    return x

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

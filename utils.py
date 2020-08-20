import os
import random

import numpy as np
import torch


def get_label2idx(labels):
    label2idx = dict()
    for idx, label in enumerate(set(labels)):
        label2idx[label] = idx
    return label2idx


def map_label2idx(koco_dataset, label_name, label2idx=None):
    """Add label key in koco dataset.

    Args:
        koco_dataset (list of dict):
        label_name (str): A string that indicates label (e.g., hate)
        label2idx (dict): String label to integer label mapper

    Returns:
        koco_dataset_with_label (list of dict)
    """
    if label2idx is None:
        labels = [d[label_name] for d in koco_dataset]
        label2idx = get_label2idx(labels)

    index_name = "label_index"
    for i, d in enumerate(koco_dataset):
        koco_dataset[i][index_name] = label2idx[d[label_name]]
    return koco_dataset, label2idx


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_and_ngpus():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    return device, n_gpus


def makedirs(dirpath):
    os.makedirs(dirpath, exist_ok=True)


def read_lines(filepath):
    with open(filepath, encoding="utf-8", mode="r") as f:
        return f.read().strip().split("\n")

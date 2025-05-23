import json
import math
import random
import sys
import time
from tqdm import tqdm
from typing import Any
from itertools import product

import numpy as np
import torch

from define_qlm import get_train_evaluate, evaluate, create_model, setup_dataset

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

quixer_hparams = {
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "QLINSVT",
    "print_iter": 50
}


lstm_hparams = {
    "layers": 2,
    "window": 32,
    "residuals": False,
    "epochs": 3,
    "restart_epochs": 30000,
    "dropout": 0.30,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "print_iter": 50
}


fnet_hparams = {
    "layers": 2,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "FNet",
    "print_iter": 50
}


vas_hparams = {
    "layers": 1,
    "heads": 1,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "VAS",
    "print_iter": 50
}


cdimensions = [96, 128]
qdimensions = [512]


model_map = {
#    "VAS": (vas_hparams, cdimensions),
#    "LSTM": (lstm_hparams, cdimensions),
#    "FNet": (fnet_hparams, cdimensions),
    "QLINSVT": (quixer_hparams, qdimensions)
}

torch.backends.cudnn.deterministic = True

device = torch.device(device_name)
print(f"Running on device: {device}")

train_evaluate = get_train_evaluate(device)


for model_name, meta in model_map.items():
    fix_hyperparams, dimensions = meta
    for dim in dimensions:
        for seed in torch.randint(high=1000000, size=(1,)).tolist():
            fix_hyperparams["model"] = model_name
            fix_hyperparams["dimension"] = dim
            fix_hyperparams["seed"] = seed

            train_evaluate(fix_hyperparams)

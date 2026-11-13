"""Reproducibility and IO helpers."""
from pathlib import Path
import csv
import json
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def select_device(request='auto'):
    if request=='auto': request='cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(request)


def json_save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as f: json.dump(data,f,indent=2,ensure_ascii=False)


def csv_save(path,rows):
    rows=list(rows)
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def torch_load(path):
    try:return torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:return torch.load(path,map_location='cpu')


def autocast(device,enabled):
    return torch.autocast(device_type='cuda',dtype=torch.float16,enabled=enabled) if device.type=='cuda' else __import__('contextlib').nullcontext()


def grad_scaler(device,enabled):
    try:return torch.amp.GradScaler('cuda',enabled=enabled and device.type=='cuda')
    except (TypeError,AttributeError):return torch.cuda.amp.GradScaler(enabled=enabled and device.type=='cuda')

import torch
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
import numpy as np
import xgboost as xgb
import os
import sys
import pickle

from torch.cuda.amp import autocast, GradScaler
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

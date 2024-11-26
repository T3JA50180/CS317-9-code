import torch
import matplotlib.pyplot as plt
import torchmetrics
import torchvision
import numpy as np
import xgboost as xgb
import os
import sys
import pickle
import seaborn as sns
import torch.multiprocessing as mp
import h2o
import pandas as pd

from h2o.estimators import H2OGeneralizedLinearEstimator
from torch.cuda.amp import autocast, GradScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from torch import nn
from tqdm.auto import tqdm
from timeit import default_timer as timer

# from torchmetrics.classification import MulticlassConfusionMatrix
from xgboost import XGBClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.set_default_device(device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

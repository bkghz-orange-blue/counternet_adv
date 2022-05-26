    # copy essential imports from fastai2
# https://github.com/fastai/fastai/blob/master/fastai/imports.py

import matplotlib.pyplot as plt,numpy as np,pandas as pd,scipy
from typing import Union,Optional,Dict,Callable,Any,List
import io,operator,sys,os,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,random,time,math,bz2,typing,numbers,string,yaml
import multiprocessing,threading,urllib,tempfile,concurrent.futures,matplotlib,warnings,zipfile

# import pytorch-related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

# misc.
from pprint import pprint
import logging
from tabulate import tabulate
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from fastcore.utils import in_jupyter
from abc import ABC, abstractmethod, ABCMeta
from copy import copy, deepcopy
import wandb

# pytorch-lightening
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import LightningLoggerBase

from pytorch_lightning.utilities.seed import seed_everything


pl_logger = logging.getLogger('lightning')
seed_everything(seed=31)

warnings.filterwarnings("ignore", category=UserWarning)

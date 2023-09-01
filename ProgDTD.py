import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision import datasets, transforms
import seaborn as sns
import pandas as pd
import pytorch_lightning as pl
from copy import copy
import torchvision
import random
from dahuffman import HuffmanCodec
import cv2
import json
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchmetrics import Accuracy
from torch.utils.data import Dataset
import os
from skimage import io
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import sys
import logging
from torch.nn.parameter import Parameter
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Function
from typing import Dict, List, Optional, Sequence, Tuple
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN as GeneralizedDivisiveNormalization
from compressai.models.google import get_scale_table
from torch import Tensor
import PIL
import torch.optim as optim
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from tqdm import tqdm

device = 'cuda:0'
print(torch.cuda.is_available())
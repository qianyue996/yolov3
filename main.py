import json
import os
from collections import defaultdict
from tqdm import tqdm
import cv2 as cv
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from utils.tools import cvtColor

from utils.dataloader import YOLODataset

ds = YOLODataset()
dl = DataLoader(ds,batch_size=1)

for item in dl:
    a,b = item

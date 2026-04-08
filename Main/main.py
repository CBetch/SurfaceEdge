import os
from transformers import RobertaTokenizerFast
import itertools
from torch.utils.data import Dataset as D 
import random
import torch.nn.functional as F
import torch
from collections import Counter
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image
import json
from typing import Callable
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    

if __name__ == '__main__':
    main()
import pycocotools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import logging
from random import randint

import math

x = [torch.rand(3) for i in range(3)]

print(x)

y = {n: torch.rand(3) for n in range(3)}

print(y)

z = [randint(0,3) for i in range(15)]
print(z)
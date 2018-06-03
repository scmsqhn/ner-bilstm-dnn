#!/bin/bash

import tensorflow as tf
import padnas as pd
import sys
import os
from tensorflow.python.ops import variable_scope as vs
import pdb
import gensim
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import time
import os
import jieba
import collections
import sklearn.utils
from sklearn.utils import shuffle
import myconfig as config
import sklearn as sk
import tensorflow as tf
from tensorflow.contrib import rnn
import json
import arctic
from arctic import Arctic
import pymongo
CURPATH = os.path.dirname(os.path.realpath(__file__))
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
sys.path.append("/home/distdev")
print(CURPATH)
print(PARPATH)
import bilstm
from bilstm import datahelper
from bilstm.datahelper import Data_Helper


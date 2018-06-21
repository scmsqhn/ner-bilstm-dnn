#!coding=utf8
import sys
import os
from tensorflow.python.ops import variable_scope as vs
import pdb
import gensim
import traceback
import numpy as np
import pandas as pd
import re
import time
import os
import jieba
import collections
import sklearn.utils
from sklearn.utils import shuffle
import sklearn as sk
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import json
#import arctic
#from arctic import Arctic
import pymongo

sys.path.append("/home/distdev/addr_classify")
sys.path.append("/home/distdev/BilstmGit")

CURPATH = os.path.dirname(os.path.realpath(__file__))
PARPATH = os.path.dirname(CURPATH)
sys.path.append(PARPATH)
sys.path.append(CURPATH)
print(CURPATH)
print(PARPATH)

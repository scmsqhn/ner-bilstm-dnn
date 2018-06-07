#
import os
import sys
import pymysql
import datetime
import pymysql
import traceback
import pandas as pd
import xgboost as xgb
import numpy as np
import re
from sklearn.manifold import TSNE
from gensim import corpora
import gensim

conn = pymysql.Connect(
          host = "117.107.241.69",
          port = 3310,
          user = "production",
          passwd = "V2aBPgBwb8EuPkSe",
          db = "zhaiquanyujing",
          )

cursor = conn.cursor()

gen = os.walk("./bond")
p,d,f = gen.__next__()

for fl in f:
    if fl.split(".")[1] == 'csv':
        df = pd.read_csv(os.path.join('./bond',fl))
        t = re.findall("result(.+)_",fl)
        print(df.head())

#sql_ = "INSERT INTO resultTable VALUES('', '%s', CURTIME(), '%s');"%(i,str(format(dict_res[i],'.9e')))
#cursor.execute('')

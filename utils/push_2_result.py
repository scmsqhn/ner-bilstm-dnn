#encoding=utf8
import os
import sys
import pymysql
import datetime
import traceback
import pandas as pd
import numpy as np
import re

conn = pymysql.Connect(
          host = "117.107.241.69",
          port = 3310,
          user = "production",
          passwd = "V2aBPgBwb8EuPkSe",
          db = "zhaiquanyujing",
          charset='utf8'
          )
cursor = conn.cursor()

gen = os.walk("./bond")
p,d,f = gen.__next__()
for fl in f:
    if fl.split(".")[1] == 'csv':
        df = pd.read_csv(os.path.join('./bond',fl))
        t = re.findall("result(.+)_",fl)[0]
        print(df.head())
        sc=len(list(set(df.loc[:,'1'])))+1
        for i in df.index:
            comp = df.ix[i,1]
            score = (df.ix[i,2]+1)/sc
            #sql= "INSERT INTO resultTable VALUES('', '%s', CURTIME(), '%s');"%(i,str(format(dict_res[i],'.9e')))
            sql = "INSERT INTO resultTable VALUES('', '%s', str_to_date(%s, '%%Y-%%m-%%d'), '%s');"%(comp, t, score)
            print(sql)
            cursor.execute(sql)
        conn.commit()

sql = "SELECT * from resultTable;"
print(sql)
cursor.execute(sql)
for i in cursor:
    print(i)




#encoding=utf8
import pandas as pd
import pymongo
import pdb

def get_risk_comp_classify():
  f = open("./risk_comp_classify_tmp.txt","a+")
  df = pd.read_csv("/home/siyuan/bond_risk/bond_risk_sec/pic/bond/result2018-04-26_19:31:15.csv")
  conn = pymongo.MongoClient("mongodb://10.6.5.32:27017")
  db = conn['myDB']
  coll = db['middleTable']
  ll = list(set(df['1']))
  for c in ll:
    cns = df[df['1']==c]['0']
    f.write("*"*30)
    f.write("this is class %s"% str(c))
    f.write("*"*30)
    for cn in cns:
      f.write("="*30)
      f.write("\n>>: "+str(c)+"/"+cn)
      f.write("="*30)
      for i in coll.find({"compname":cn},{"label"}):
        print(i['label'])
        f.write("\n>>"+i['label'])
  f.close()

get_risk_comp_classify()

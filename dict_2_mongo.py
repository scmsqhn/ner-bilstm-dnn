import pymongo
import json
import pdb

cont = ""
with open("dct.json","r")as f:
    cont=f.read()
items = json.loads(cont)
mongo=pymongo.MongoClient('mongodb://127.0.0.1:27017')
col=mongo['myDB']['traindata']

cnt=0
for i in items:
    #pdb.set_trace()
    col.insert(items[i])
    cnt+=1
    if cnt%1000==1:
        print(items[i])
col.find_one()
col.count()

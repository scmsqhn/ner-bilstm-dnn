import pymongo
import json
import pdb
lst = {}
mongo = pymongo.MongoClient('mongodb://127.0.0.1:27017')
col = mongo['myDB']['ner_addr_crim_sample']
cnt = 0
for i in col.find():
    dct = {}
    try:
        dct['addrcrim'] = ",".join(i['addrcrim'])
    except:
        continue
    dct['text'] = i['text']
    cnt+=1
    lst[str(cnt)] = dct
    #pdb.set_trace()

with open("dct.json","a+")as f:
    f.write(json.dumps(lst))


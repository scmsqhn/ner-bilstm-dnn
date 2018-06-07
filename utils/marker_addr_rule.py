#/bin/bash
import sys,pdb
sys.path.append("/home/siyuan")
import bilstm
from bilstm import addr_classify
import pymongo
sys.path.append("/home/siyuan/svn/algor")
ac = addr_classify.Addr_Classify(['我爱北京天安门'])
coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
cnt = 0
for item in coll.find():
    try:
        item['_id']
    except:
        continue
    if item['text'] == "":
        continue
    wds = ""
    try:
        wds = ac.run(item['text'])
    except:
        continue
    crim = [i for i in wds['crim']]
    live = [i for i in wds['live']]
    reg = [i for i in wds['reg']]
    if '' in live:
        live.remove('')
    if '' in crim:
        crim.remove('')
    if '' in reg:
        reg.remove('')
    #pdb.set_trace()
    coll.update_one({"_id":item["_id"]},{"$set":{"addrcrim":crim}})
    coll.update_one({"_id":item["_id"]},{"$set":{"addrlive":live}})
    coll.update_one({"_id":item["_id"]},{"$set":{"addrreg":reg}})
    if cnt%100==1:
        print("\n> we handle the ", cnt, item['text'])


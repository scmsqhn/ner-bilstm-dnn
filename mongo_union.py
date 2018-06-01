#coding=utf-8
import pymongo

conn=pymongo.MongoClient("mongodb://127.0.0.1:27017")

coll = conn['myDB']['ner_addr_crim_sample']

for i in conn['myDB']['gz_gongan_alarm_1617'].find():
    coll.insert({"text":i['反馈内容']})

for i in conn['myDB']['gz_gongan_case'].find():
    coll.insert({"text":i['jyaq']})

for i in conn['myDB']['original_data'].find():
    coll.insert({"text":i['casdetail']})

print("\n> finish", coll.count())

#coding=utf-8
import pymongo

conn=pymongo.MongoClient("mongodb://127.0.0.1:27017")

print(conn['myDB'])
print("===")
coll = conn['myDB']['ner_addr_crim_sample']
print([i for i in coll.find()[:10]])

print("===")
print(conn['myDB']['gz_gongan_alarm_1617'].find_one())
print("===")

print(conn['myDB']['gz_gongan_case'].find_one())
print("===")

print(conn['myDB']['original_data'].find_one())
print("===")


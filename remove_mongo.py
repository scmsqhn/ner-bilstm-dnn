import pymongo
import json
import pdb

mongo=pymongo.MongoClient('mongodb://127.0.0.1:27017')
col=mongo['myDB']['traindata']
col.remove_many()



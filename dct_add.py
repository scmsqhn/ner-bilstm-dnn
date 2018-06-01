import pymongo
import json
import pdb
import gensim
import jieba
dct = gensim.corpora.Dictionary
global mydct
mydct = dct.load("./model/my.dct.bak")

lst = {}
mongo = pymongo.MongoClient('mongodb://127.0.0.1:27017')
col = mongo['myDB']['ner_addr_crim_sample']
col1 = mongo['myDB']['traindata']
col2 = mongo['myDB']['original_data']

def run(col):
  global mydct
  cnt = 0
  for i in col.find():
    cnt+=1
    j=str(i)
    words = list(set(jieba.cut(j)))
    chars = list(j)
    l = []
    l.extend(words)
    l.extend(chars)
    mydct.add_documents([l])

run(col)
run(col1)
run(col2)
mydct.save('./model/my.dct.bak')


#/usr/bin/bash
#/coding = 'utf-8'

import numpy as np
import sys
import jieba
words = jieba.cut("我爱北京天安门")
import os
sys.path.append('.')
sys.path.append('..')
import requests
import json
import random
import traceback
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
from db_inf import *

sys.path.append("/home/siyuan/iba/extcode/word2vec")
sys.path.append("/home/siyuan/iba/extcode/")

import word2vec
from word2vec import _gensim_word2vec as word2vec

#url1="http://127.0.0.1:27719/predict/110/phoneNum"
#url2="http://127.0.0.1:27719/predict/110/identifier"
#url3="http://127.0.0.1:27719/predict/110/weixin"
#url4="http://127.0.0.1:27719/predict/110/carinfo"
url15="http://60.247.77.171:8888/beijing/predict/110/name_parse"
url7="http://60.247.77.171:8888/beijing/predict/110/addrclassify"
url74="http://60.247.77.171:8888/beijing/predict/110/loc"
url159="http://60.247.77.171:8888/guizhou/algor/namextr"
#url2="https://113.204.229.74:18100/guizhou/method/carinfo"
#url3="https://113.204.229.74:18100/guizhou/loc/predict"
#url1="http://127.0.0.1:27719/predict/110/carinfo"
#url2="http://127.0.0.1:27719/predict/110/phoneNum"
#url3="http://127.0.0.1:27719/predict/110/ner/carinfo"
#url1="http://127.0.0.1:5556/predict/110/phoneNum"
#url1="http://127.0.0.1:23578/ner/carinfo"
#url1="http://0.0.0.0:27719/ner/carinfo"
#url2="http://113.204.229.74:27719/ner/carinfo"
#url3="http://127.0.0.1:27719/ner/phoneNum"
#url3="http://113.204.229.74:27719/method/weixin"
headers = {'content-type': 'application/json', "Accept": "application/json"}
#f = open('./data/ner_sample_mini.txt', 'r')

columns=["carinfo", "creditCard", "identifier", "phoneNum", 'criminalphone', "qq", "weibo", "wx","momo","nickname","web","mail",'qqname','wxname','criminalidentifier','criminalqq','criminalwx']

import json
def _post(url, f):
  l1 = f.readline()
  l2 = ""
  if "微信" in l1:
    l2 = l1
    print(l1)
  else:
    return

  data = {
        'messageid': "12",
        'clientid': "13",
        'text':[l2],
        #'funcname':columns[random.randint(0,len(columns)-1)],
        'encrypt':'false',
        }
  print(url)
  #print(data['funcname'])
  print(data['text'])
  res = requests.post(url,data=json.dumps(data),headers=headers,verify=False,timeout=200)
  print('>>>> res')
  print(res)
  print(res.status_code)
  print(res.headers)
  print(res.text)

def _post_baidu():
    res = requests.get(url="https://www.baidu.com/",verify='false')

import name_parse
from name_parse.predict import  parse_name
def feed_cont(db):
    #f=open("name_addr_cont.json","a+")
    _id = 0
    #dicLst = []
    print("\n>> feed_cont()")
    sav_coll = db.get_collection("name_addr_classify")
    coll = db.get_collection("gz_gongan_alarm_1617")
    sec_coll = db.get_collection("gz_gongan_case")

    for i in coll.find({},{"反馈内容":1}):
        i = i['反馈内容']
        b0 = i
        print(i)
        i = re.sub("\（[^（）]+\）","",i) #将括号内的词汇去掉
        #i = re.sub("\([^（）+]\)","",i) #将括号内的词汇去掉
        print(i)
        b1=i
        ##pdb.set_trace()

        dct = {}
        print(i,type(i))
        #print(parse_name(i))
        #print(url_post(url74, i))
        try:
          dct['addr'] = url_post(url74,i)
        except:
          continue
        dct['name'] = ",".join(parse_name(i))
        dct['id'] = _id
        dct['cont'] = i
        _id+=1
        print(sav_coll.insert(dct))
        #dicLst.append(dct)
        if False:#_id>1:
        #if _id>10:
            break

    #for i in sec_coll.find():
    for i in sec_coll.find({},{"处警简要情况":1}):
        print(i,type(i))
        ##pdb.set_trace()
        i = i['处警简要情况']
        dct = {}
        #print(i,type(i))
        #print(parse_name(i))
        #print(url_post(url74, i))
        try:
          dct['addr'] = url_post(url74,i)
        except:
          continue
        dct['id'] = _id
        dct['cont'] = i
        dct['name'] = ",".join(parse_name(i))
        _id+=1
        print(sav_coll.insert(dct))
        #dicLst.append(dct)
        if False:#_id>3:
        #if _id>20:
            break
    #pd.DataFrame(dicLst).to_csv("name_addr_cont.csv")
    #dic_str=json.dumps(dct)
    #dicLst.append(dct)
    #f.write(dic_str)
    #f.close()

def url_post(url, cont):
  data = {
        'messageid': "12",
        'clientid': "13",
        'text':[cont],
        #'funcname':columns[random.randint(0,len(columns)-1)],
        'encrypt':'false',
        }
  res = requests.post(url,data=json.dumps(data),headers=headers,verify=False,timeout=200)
  ##pdb.set_trace()
  return json.loads(res.text)["result"]

def _post_baidu():
    res = requests.get(url="https://www.baidu.com/",verify='false')

def round_test():
  while(1):
    try:
      _post(url74, f)
      _post(url159, f)
    except:
      f = open('./data/ner_sample_mini.txt', 'r')
      traceback.print_exc()

words_2d = []

def get_name_addr_cont(db):
    print("\n>> get_name_addr_cont")
    pass
    coll= db['name_addr_classify'] 
    conts = coll.find({},{"cont":1,"_id":0,'name':1,'addr':1})
    sentences = []
    for i in conts:
        sentences.append(i['cont'])
        #print("cont",i['cont'])
        #print("name",i['name'])
        #print("addr",i['addr'])
    #pdb.set_trace()
    for sentence in sentences:#[:1000]:
        #print(sentence)
        words = jieba.cut(sentence)
        _words = list(words)
        #print(_words)
        words_2d.append(_words)

    #pdb.set_trace()
    w2v_cls_instance = word2vec.wd2vec()
    w2v_cls_instance._word2vec(words_2d)
    print("the model save into the path:", w2v_cls_instance.modelpath)
    print("*"*30,"\n>> now we build the word2vec for this docs cluster")
    print(words_2d)
    pdb.set_trace()

    dictionary, corpus = w2v_cls_instance.gen_dictionary(words_2d)
    pdb.set_trace()
    corpus, dic = w2v_cls_instance.load_dict_copus()
    #pdb.set_trace()
    tfidf, index = w2v_cls_instance.tfidf_index(corpus)
    #pdb.set_trace()
    tfidf.save('./tfidfmodel') 
    #pdb.set_trace()
    return tfidf, corpus,dic ,words_2d, w2v_cls_instance

from arctic import Arctic
import quandl

def init_arctic(name='mongodb'):
    if name == 'mongodb':
        store = Arctic('mongodb://10.6.5.32')
        store.initialize_library('db_for_train')
        library = store['db_for_train']
        return library

        #apl = quandl.get("WIKI/AAPL", authtoken="your token here")
        #library.write('AAPL', aapl, metadata={'source': 'Quandl'})
        #item = library.read('AAPL')
        #aapl = item.data
        #metadata = item.metadata

if __name__ == "__main__":
  from db_inf import *
  pass
  library = init_arctic()
  print("\n>> test_request.py")
  mongoclient = pymongo.MongoClient("mongodb://10.6.5.32:27017")
  db=mongoclient['myDB']
  train_data_coll = db['tmp_train_adr_classify']
  #coll=db['name_addr_cont_guiyang']
  #cnt=coll.find().count()
  #print(cnt)
  #feed_cont(db)
  tfidf_model,corpus,dic,words_2d, w2v_cls_instance= get_name_addr_cont(db)
  vocab_word2vec, model_word2vec = w2v_cls_instance.spot_vec()
  train_data_for_add_classify = []
  for i in range(len(corpus)):
      ti = tfidf_model[corpus[i]]
      ti_lst = []
      kw_lst = []
      for m in ti:
          ti_lst.append(m[1])
      isImportant = pd.Series(ti_lst).describe()['75%']
      m0max = 0.0
      m0maxid = 0.0
      for m in ti:
          if m[1]>isImportant:
              if m[1]>m0max:
                  m0max=m[1]
                  m0maxid = m[0]
              kw_lst.append(dic[m[0]])
      if len(kw_lst)<3:
          isImportant = pd.Series(ti_lst).describe()['25%']
      for m in ti:
          if m[1]>isImportant:
              if m[1]>m0max:
                  m0max=m[1]
                  m0maxid = m[0]
              kw_lst.append(dic[m[0]])
      kw_lst.append(dic[m0maxid])
      output_arr = []
      #if len(kw_lst)<2:
      #    print(len(kw_lst))
      #    raise Exception("there is a len less than 3")

      if len(kw_lst)>50:
          print(len(kw_lst))
          kw_lst = kw_lst[:50]
          #raise Exception("there is a len more than 32")

      for i in kw_lst:
          try:
            output_arr.append(vocab_word2vec[i])
          except KeyError:
            pass
            continue
      if len(output_arr)<10:
          for i in range(10-len(kw_lst)):
              output_arr.append(np.array([0]*100))
      train_data_for_add_classify.append(output_arr)
  library.write('train_data_for_addr_classify', train_data_for_add_classify, metadata={'source': 'Quandl'})
  item = library.read('train_data_for_addr_classify')
  #print(item.metadata)
  #print(item.data)

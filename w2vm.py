#!coding=utf-8
import gensim
import gensim.models
import pymongo
import jieba
import pdb
def new_w2vm():
    sentences = []
    collections = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['ner_addr_crim_sample']
    for i in collections.find():
        words = list(jieba.cut(i['text']))
        sentences.append(words)
    collections = pymongo.MongoClient("mongodb://127.0.0.1:27017")['myDB']['original_data']
    for i in collections.find():
        words = list(jieba.cut(i['casdetail']))
        sentences.append(words)
    print("\n>sentences len", len(sentences))
    pdb.set_trace()
    w2vmodel = gensim.models.word2vec.Word2Vec(sentences=sentences,window=7,min_count=1,size=128)
    w2vmodel.save("./model/w2vm")
    print("\n>save ok")

def load_w2vm(filepath = './model/w2vm'):
    model = gensim.models.word2vec.Word2Vec.load(filepath)
    print("\n>load ok")
    return model

if __name__ == "__main__":
    pass
    new_w2vm()

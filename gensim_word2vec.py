#encoding=utf-8
# gensim for word2 vec

from gensim.models import word2vec
from gensim.models import Word2Vec  
import data_helper
import jieba
from gensim import corpora

class wd2vec(object):
    def __init__(self):
        pass
        self.modelpath = "gensim_word2vec.model"
        self.txtfilepath = "/home/siyuan/bond_risk/bond_risk_sec/beijing110_cont.txt"
        self.model = ""
        self.texts = ""
        self.sentences = list()
        self.corpus = ""
        self.tfidfmodel = ""
        self.load_txt()
        self.word2vec()
        self.texts_bind()
        self.gen_dictionary()
        self.tfidf()

    def texts_bind(self):
        for i in self.sentences:
            self.texts+=i

    def load_txt(self):
        with open(self.txtfilepath, "r") as f:
            lines = f.readlines()
            print(lines[:3])
            self.sentences = lines[:10000]
            #return lines

    def word2vec(self):
        self.model = Word2Vec(self.sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)  
        self.model.save(self.modelpath)

    def load_model(self):
        self.model = Word2Vec.load(self.modelpath) 
        return self.model

    def most_similar(self, ch):
        return self.model.most_similar(ch)

    def similar(self, ch, ch_):
        return self.model.similarity(ch, ch_)

    def char2vec(self, ch):
        return self.model[ch]

    def gen_dictionary(self):
        dictionary = corpora.Dictionary(self.texts)
        corpus = [dictionary.doc2bow(text) for text in self.texts]
        # print corpus[0] # [(0, 1), (1, 1), (2, 1)]
        return corpus

    def tfidf(self):
        self.tfidfmodel = gensim.models.TfidfModel(self.corpus)

    def tfidf_sent(self, sent):
        return self.tfidf_sent[sent]



md = wd2vec()
a = md.similar("你","我")
b = md.most_similar("你")
c = md.char2vec("他")
print(a,b,c)

def data_clear():
    _l = list()
    for sent in md.sentences:
        _d = jieba.cut(sent, HMM=True)
        for i in _d:
            _l.append(sub(i))
    return "".join(_l)
        

def sub(s): 
    s = re.sub("“",'',s)
    s = re.sub("”",'',s)
    s = re.sub("‘","",s)
    s = re.sub("’","",s)
    s = re.sub("，","",s)
    s = re.sub("。","",s)
    s = re.sub("：","",s)
    s = re.sub("！","",s)
    s = re.sub("？","",s)
    s = re.sub("（","",s)
    s = re.sub("）","",s)
    s = re.sub("：","",s)
    s = re.sub("，","",s)
    s = re.sub("[ ]+", "",s)
    s = re.sub("\(", "",s)
    s = re.sub("\)", "",s)
    s = re.sub("\[", "",s)
    s = re.sub("\]", "",s)
    return s

clear_texts = data_clear()



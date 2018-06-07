#coding=utf8
import jieba
import jieba.posseg
import pandas as pd
import collections
import dmp.gongan.ssc_dl_ner
import dmp.gongan.ssc_dl_ner.data_utils
import gensim.corpora as corpora
from gensim import corpora,similarities,models  
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging
import sys
import random
from gensim.models.doc2vec import  TaggedDocument
import pdb
# Sentence should be a link 
class Sentence(object):

    def __init__(self, sent):
       self.df = ""
       self.posLst = []
       self.wordLst = []
       self.cursor = 0 
       words_lst = list(jieba.posseg.cut(sent))
       for word in words_lst:
           self.posLst.append(list(word)[1])
           self.wordLst.append(list(word)[0])
       self.df = self.init_dataframe()

    def next(self):
       self.cursor+=1
       return (self.wordLst[self.cursor], self.posLst[self.cursor])

    def last(self):
       self.cursor-=1
       return (self.wordLst[self.cursor], self.posLst[self.cursor])

    def get(self, num):
       if num>len(self.posLst):
           num = len(self.posLst)
       if num<0:
           num=0
       return (self.wordLst[num], self.posLst[num])

    def lenth(self):
       return len(self.wordLst)

    def init_dataframe(self):
        df = pd.DataFrame()
        df['word'] = self.wordLst
        df['pos'] = self.posLst
        return df

    def get_dataframe(self):
        return self.df

    def pos_count(self):
        return collections.Counter(self.posLst)
        
    def pos_search(self, pos):
        target_words = []
        for i in range(len(self.wordLst)):
             if self.posLst[i]==pos:
                 target_words.append(self.wordLst[i])
        return target_words
    #def pos_count(self):
    #    return collections.Counter(self.posLst)


    def vpn_pvn_chunk(self):
        text = ""
        for pr in range(self.lenth):
            w = self.wordLst[pr]
            f = self.posLst[pr]
            item = "%s/%s "%(w,f)
            text+=item
        chunk_lst = []
        chunk_lst.extend(list(re.findall("( \w+/v \w+/p \w+/ns [^v]+/[xn] )", text)))
        chunk_lst.extend(list(re.findall("( \w+/p \w+/n \w+/ns [^v]+/[xn] )", text)))
        return chunk_lst

    def x_chunk(self):
        words_arrs = []
        l = len(self.wordLst)
        words_arr = []
        for i in range(l):
            w = self.wordLst[i]
            f = self.posLst[i]
            if f=="x":
                words_arr.append(w)
                words_arrs.append(words_arr)
                words_arr = []
            else:
                words_arr.append(w)
        words_arrs.append(words_arr)
        return words_arrs


class Doc_Classify(object):

    def __init__(self, doc_lst):
        print("\n> now we new a doc classify")
        self.similarity = ""
        self.tfidf_model = ""
        self.corpus_tfidf = ""
        self.sentenceLst=[] # sentenceLst is set of wordLst
        self.document = [] # document is set of Sentence Instance
        self.panel = ""
        self.cursor = 0
        self.low_dim = ""# low dim of word vec
        self.word_classify = "" # dbscan classify result
        for line in doc_lst:
            #pdb.set_trace()
            #print(line)
            line = dmp.gongan.ssc_dl_ner.data_utils.full_to_half(line)
            sent = Sentence(line)
            if len(sent.wordLst)>0:
              self.document.append(sent)
              self.sentenceLst.append(sent.wordLst)
        self.gen_dictionary_corpus()
        #self.tfidf()
        #self.lsi_model()
        #self.init_panel()
        #self.model_word2vec_doc = ""
        #self.init_similarity()
        #self.test_text_samples = []
        self.init_word_transfer_prob()
        self.init_char_transfer_prob()
        self.init_pos_transfer_prob()
        #self._trans2wordpos()

    def lenth(self):
        return len(self.document)

    def next(self):
       self.cursor+=1
       return (self.document[self.cursor])

    def last(self):
       self.cursor-=1
       return (self.document[self.cursor])

    def get(self,num):
       self.cursor=num
       return (self.document[self.cursor])

    def init_panel(self):
        self.panel = pd.Panel()
        for sent in self.document:
            self.panel.add(sent.df)

    def get_panel(self):
        return self.panel
        
    def pos_count(self):
        base_pos_count_sent = {}
        for sent in self.document:
            pos_count_sent = sent.pos_count()
            for k in pos_count_sent:
               if k in base_pos_count_sent:
                  base_pos_count_sent[k]+=pos_count_sent[k]
               else:
                  base_pos_count_sent[k]=pos_count_sent[k]
        return base_pos_count_sent

    def pos_search(self,pos):
        target_words = []
        for sent in self.document:
            words_sent = sent.pos_search(pos)
            target_words.extend(words_sent)
        return target_words

    def word2vec(self):
        self.model_word2vec_doc = gen_w2v_model(self.sentenceLst)
        return self.model_word2vec_doc

    def tfidf(self):
        self.tfidf_model = gensim.models.TfidfModel(self.corpus)  
        self.corpus_tfidf = self.tfidf_model[self.corpus]  
        return self.tfidf_model, self.corpus_tfidf

    def tsne(self):
        self.low_dim = gensim_util.tsne(self.model_word2vec_doc)
        return self.low_dim

    def dbscan(self):
        self.word_classify = gensim_util.dbscan(self.low_dim)
        return self.word_classify

    def gen_dictionary_corpus(self):
        self.dictionary = corpora.Dictionary(self.sentenceLst)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.sentenceLst]
    def init_similarity(self):
        #l = (self.lenth()//3) if (self.lenth()//3)<2 else 2
        #print("\n> l=%s"%l)
        self.similarity = similarities.Similarity('Similarity-tfidf-index', self.corpus_tfidf, num_features=900) 
        self.similarity.num_best = 5

    def similarities(self, sentence):
        words = list(jieba.cut(sentence))
        words_corpus = self.dictionary.doc2bow(words)
        words_corpus_tfidf = self.tfidf_model[words_corpus]
        return self.similarity[words_corpus_tfidf]

    def lsi_model(self):
        self.lsi = gensim.models.LsiModel(self.corpus_tfidf) # is as generator 
        self.corpus_lsi = self.lsi[self.corpus_tfidf]  # generate a model of thie doc
        self.similarity_lsi_model=similarities.Similarity('Similarity-LSI-index', self.corpus_lsi, num_features=900,num_best=5)  
        self.similarity_lsi_model.save("./similarity_lsi_model.model")
        

    def sent_get_tfidf(self, sent):
        words = list(jieba.cut(sent))
        words_doc2bow = self.dictionary.doc2bow(words)
        words_bow_tfidf = self.tfidf_model[words_doc2bow]
        return words_bow_tfidf

    def similarities_lsi(self, sent):
        words = list(jieba.cut(sent))
        words_doc2bow = self.dictionary.doc2bow(words)
        words_bow_tfidf = self.tfidf_model[words_doc2bow]
        self.simi_lsi= self.lsi[words_bow_tfidf]
        return self.simi_lsi, self.similarity_lsi_model[self.simi_lsi]

    def save_test_doc(self, doc_lst):
        self.test_text_samples = doc_lst

    def sentenceLst4doc2vec(self):
        cnt = 0
        x_train = []
        for sent in self.sentenceLst:
            cnt+=1
            document = TaggedDocument(sent, tags=[cnt])  
            x_train.append(document)  
        return x_train

    def init_doc2vec_model(self):
        x_train = self.sentenceLst4doc2vec()
        self.doc2vec_model = Doc2Vec(x_train, min_count=1, window=10, size=100, sample=1e-4, negative=5, dm=1, workers=7)
        for epoch in range(22):
            self.doc2vec_model.train(x_train, total_examples=self.doc2vec_model.corpus_count, epochs=1)  
            self.doc2vec_model.alpha -= 0.002
            self.doc2vec_model.min_alpha = self.doc2vec_model.alpha
        self.doc2vec_model.save("./doc2vec.model")
        return self.doc2vec_model

    def doc2vec_predict(self,sent):
        self.doc2vec_model = Doc2Vec.load("./doc2vec.model")  
        inferred_vector_dm = self.doc2vec_model.infer_vector(sent)  
        doc_vec_sims = self.doc2vec_model.docvecs.most_similar([inferred_vector_dm], topn=10)  
        return doc_vec_sims

    def init_word_transfer_prob(self):
        df = pd.DataFrame()
        dq = collections.deque(maxlen=2)
        for senInstance in self.document:
            #print(senInstance.wordLst)
            dq.append(senInstance.wordLst[0])
            for word in senInstance.wordLst[1:]:
                dq.append(word)
                try:
                    df.loc[dq[0], dq[1]]+=1
                except KeyError:
                    df.loc[dq[0], dq[1]]=1
        self.word_transfer_prob = df

    def init_char_transfer_prob(self):
        df = pd.DataFrame()
        dq = collections.deque(maxlen=2)
        for senInstance in self.document:
            char_str = "".join(senInstance.wordLst)
            dq.append(char_str[0])
            for char in char_str[1:]:
                dq.append(char)
                try:
                    df.loc[dq[0], dq[1]]+=1
                except KeyError:
                    df.loc[dq[0], dq[1]]=1
        self.char_transfer_prob = df
                      
    def init_pos_transfer_prob(self):
        df = pd.DataFrame()
        dq = collections.deque(maxlen=2)
        for senInstance in self.document:
            dq.append(senInstance.posLst[0])
            for word in senInstance.posLst[1:]:
                dq.append(word)
                try:
                    df.loc[dq[0], dq[1]]+=1
                except KeyError:
                    df.loc[dq[0], dq[1]]=1
        self.pos_transfer_prob = df

    def trans2wordpos(self, sent):
        text = ""
        assert type(sent) == str
        pair_lst = list(jieba.posseg.cut(sent))
        l = len(pair_lst)
        for i in pair_lst:
            word_pos = "%s/%s "%(i.word, i.flag)   
            text+=word_pos
        text+="\n"
        self.text = text
        return self.text

    def _trans2wordpos(self):
        text = ""
        for sentence in self.document:
            l = sentence.lenth()
            for i in range(l):
                word_pos = "%s/%s "%(sentence.wordLst[i], sentence.posLst[i])   
                text+=word_pos
            text+="\n"
        self.text = text
        return self.text

    def vpn_pvn_chunk(self,sent):
        text = sent
        chunk_lst = []
        for i in re.findall("( \w+/v \w+/p \w+/ns [^v]+/[xn] )", text):
            chunk_lst.append(i)
        for i in re.findall("( \w+/p \w+/n \w+/ns [^v]+/[xn] )", text):
            chunk_lst.append(i)
        items = []
        for item in chunk_lst:
            s = re.findall("(.+?)/\w+ ", item)   
            items.append("".join(s)) 
        return items

    def cut_word_vpn_pvn_chunk(self,sent):
        text = ""
        pair_lst = jieba.posseg.cut(sent)
        for pr in pair_lst:
            w = pr.word
            f = pr.flag
            item = "%s/%s "%(w,f)
            text+=item
        #print("sent",sent)
        #print("text",text)
        chunk_lst = []
        for i in re.findall("( \w+/v \w+/p \w+/ns [^v]+/[xn] )", text):
            chunk_lst.append(i)
        for i in re.findall("( \w+/p \w+/n \w+/ns [^v]+/[xn] )", text):
            chunk_lst.append(i)
        #print("chunk_lst",chunk_lst)
        items = []
        for item in chunk_lst:
            s = re.findall("([\w，\.,。:'\"])/\w+", item)   
            items.append("".join(s)) 
        #print("items",items)
        return items

    def load_lsi_model(self):
        return gensim.models.lsimodel.LsiModel.load("./similarity_lsi_model.model")
                      
if __name__ == "__main__":            
    mini_text_samples = ['2014年12月28日23时许,曾进(女,28岁,汉族,本科文化,个体,电话号码:13765041977,身份证号码:522423198610040023,现住址:云岩区万江小区17栋204号,户籍:贵阳市云岩区)到我所报称今日22时许,在贵阳市云岩区万江小区17栋204号家中,被撬窗入室盗窃,被盗物品:一对戒指、价值200元,一颗黄金戒指、价值1500元,一颗钻戒、价值6990元、一对黄金耳环、价值1500元,一对白金耳环、价值1800元、一颗吊坠、价值400元,一条项链、价值5200元。',
 '2014年12月30日晚上19时许,报警人,胡叶梅,女,1001年5月25日出生,户籍地址:山西省吕梁市临县车赶乡,身份证号码141124199105250188,在延安西路友谊商场一楼艾莱依服装店内被一名女子以顺手牵羊的方式盗窃收银台抽屉内的现金6861元。',
 '2014年11月5日,宁丹丹到我队报称:其卡号6222600580001766228的交通银行储蓄卡于2014年11月2日晚20时14分在青岛市被他人盗刷肆万元人民币(￥40000.00)。',
 '2014年12月30日23时51分许,王永强(男,汉族,出生日期:1977年7月27日,身份证号:522725197707273514,户籍所在地:贵阳市云岩区乌金路278号1栋1单元附2号贵,现住:阳市云岩区百花山登高小区10栋2单元103号,电话:13595057031)来我所报称,其于2014年12月23日停放在百花山登高小区10栋2单元103号窗下的电瓶车电瓶被盗,被盗物品:超威牌电瓶5个,型号不详,购价1200元,购买时间:2013年7月,总损失价值:1200元。',
 '2014年12月30日21时许,我所接到举报称在贵阳市云岩区石洞坡路有人贩卖毒品海洛因,得知后我所民警立即前往布控,2014年12月30日22时30分许,我所民警在贵阳市云岩区石洞坡路丁字路口将贩卖毒品海洛因嫌疑人吴仕涛抓获,并缴获吴仕涛贩卖的毒品海洛因0.1克。',
 '2014年12月30日晚上20时30分左右,我所民警接线人举报,在贵阳市云岩区东山“碧海云天”浴室附近有人贩卖毒品。后我所民警及乔装成线人的巡防队员在线人配合下,在控制下于贵阳市云岩区东山“碧海云天”浴室附近路边抓获两名名正在贩卖毒品冰毒的嫌疑男子。并当场从其中一名男生身上搜出作案工具黑色三星手机一部,又从我所巡防队员手中收缴该两名男子所贩卖的用透明塑料袋封装的毒品冰毒疑似物一小包(经称重,净重0.5克)。经讯问,该黄奎、杨政承认贩卖毒品冰毒的事实。',
 ' 2014年12月27日早上8时许,吴鹏发现其在贵阳市云岩区东山村巫峰组星月浴室旁的租住房家中被溜门入室盗窃一部白色OPPO手机(型号N1,串号及卡号不详,2013年2月份购买,购价3300元人民币,现价值500元人民币)、一部黑色小米3手机(串号不详,卡号:15085973068,2014年3月份购买,购价1999元人民币,现价值1000元人民币)、一部黑色尼采手机(型号不详、串号不详及卡号不详,现价值100元人民币)、一台灰色惠普笔记本电脑(型号:i3,2013年10月份购买,购价3700元人民币,现价值2000元人民币)。',
 '2014年12月10日13时许,报警人庞影(女,1995年08月17日出生,汉族,初中文化程度,户籍所在地重庆市黔江区沙坝乡西泡村3组,现住贵阳市云岩区嘉华酒店112号房,现在无工作,居民身份证号码500239199508172045,联系电话18685475912)报称在延安西路“路尚鞋店”购物时其放在双肩包内的200元现金及一部白色的OPPO手机(购于2014年4月,串号不详,本机号18685475912,购价2400元)被盗。',
 ' 2014年12月30日12时30分许,梅忠心(男,汉族,1990年01月06日生,身份证号码:522124199001064075,户籍:贵州省正安县中观镇鲜光村街上组 ,现住:贵阳市云岩区海马冲16号2号5楼7号,联系电话:18286084407),报称其在贵阳市云岩区海马冲因租房签订虚假合同被诈骗11000元人民币。',
 '2014年12月30日18时13分许,翁顺鸿(男,24岁,身份证号码:520123199005102413,户籍:贵州省修文县谷堡乡大寨村木家寨组,联系电话:13885137444。)在贵阳市云岩区大营坡农贸批发市场被扒窃,被盗走一部白色苹果6手机,串号不详,购于2014年10月,购价:5700元人民币。']
    doc = Doc_Classify(mini_text_samples)
    pass
    

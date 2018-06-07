#coding=utf8
import re
import pandas as pd
import gensim
import gensim.models
import jieba
#words_define = "/home/distdev/addr_classify/words_define.txt"
#jieba.load_userdict(words_define)
import gensim
from gensim.models import word2vec
import sklearn.cluster
from sklearn.cluster import dbscan
import sklearn.manifold
import sys
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import TSNE
import re
import numpy as np
from sklearn.manifold import TSNE
import sys
sys.path.append("/home/siyuan/algor/src/iba")
import dmp.gongan.ssc_dl_ner.data_utils
import dmp.gongan.gz_case_address.predict
import pymongo
import traceback
import pdb

mongo_client = pymongo.MongoClient("mongodb://127.0.0.1")
myDB = mongo_client['myDB']
coll = myDB['gz_gongan_alarm_1617']
data = coll.find()
#print(coll.count())#['反馈内容']
fknr = [i['反馈内容'] for i in data]
#print(len(fknr))
def get_sentences_from_db(collection):
    sentences = []
    for i in collection.find():
        words = list(jieba.cut(i['jyaq']))
        sentences.append(words)
    return sentences

def gen_w2v_model(sentences):
    gz_gongan_case_model_w2v = word2vec.Word2Vec(sentences=sentences, size=100, alpha=0.025, window=5, min_count=3, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,trim_rule=None)
    return gz_gongan_case_model_w2v

def tsne(model):
    arr = []
    for k in model.wv.vocab.keys():
        arr.append(model[k])
    X_embedded = TSNE(n_components=2).fit_transform(arr)
   # print(X_embedded.shape,"\n")
    return X_embedded

def gen_dataframe():
    pass

def dbscan(arr):
    dbscan = sklearn.cluster.DBSCAN(eps=0.2,min_samples=5,metric='euclidean',algorithm='auto',leaf_size=30,p=None)
    y_pred = dbscan.fit_predict(arr)
    return y_pred



text_samples__ = ['2014年12月28日23时许,曾进(女,28岁,汉族,本科文化,个体,电话号码:13765041977,身份证号码:522423198610040023,现住址:云岩区万江小区17栋204号,户籍:贵阳市云岩区)到我所报称今日22时许,在贵阳市云岩区万江小区17栋204号家中,被撬窗入室盗窃,被盗物品:一对戒指、价值200元,一颗黄金戒指、价值1500元,一颗钻戒、价值6990元、一对黄金耳环、价值1500元,一对白金耳环、价值1800元、一颗吊坠、价值400元,一条项链、价值5200元。',\
'2014年12月30日晚上19时许,报警人,胡叶梅,女,1001年5月25日出生,户籍地址:山西省吕梁市临县车赶乡,身份证号码141124199105250188,在延安西路友谊商场一楼艾莱依服装店内被一名女子以顺手牵羊的方式盗窃收银台抽屉内的现金6861元。',\
'2014年11月5日,宁丹丹到我队报称:其卡号6222600580001766228的交通银行储蓄卡于2014年11月2日晚20时14分在青岛市被他人盗刷肆万元人民币(￥40000.00)。',\
'2014年12月30日23时51分许,王永强(男,汉族,出生日期:1977年7月27日,身份证号:522725197707273514,户籍所在地:贵阳市云岩区乌金路278号1栋1单元附2号贵,现住:阳市云岩区百花山登高小区10栋2单元103号,电话:13595057031)来我所报称,其于2014年12月23日停放在百花山登高小区10栋2单元103号窗下的电瓶车电瓶被盗,被盗物品:超威牌电瓶5个,型号不详,购价1200元,购买时间:2013年7月,总损失价值:1200元。',\
'2014年12月30日21时许,我所接到举报称在贵阳市云岩区石洞坡路有人贩卖毒品海洛因,得知后我所民警立即前往布控,2014年12月30日22时30分许,我所民警在贵阳市云岩区石洞坡路丁字路口将贩卖毒品海洛因嫌疑人吴仕涛抓获,并缴获吴仕涛贩卖的毒品海洛因0.1克。',\
'2014年12月30日晚上20时30分左右,我所民警接线人举报,在贵阳市云岩区东山“碧海云天”浴室附近有人贩卖毒品。后我所民警及乔装成线人的巡防队员在线人配合下,在控制下于贵阳市云岩区东山“碧海云天”浴室附近路边抓获两名名正在贩卖毒品冰毒的嫌疑男子。并当场从其中一名男生身上搜出作案工具黑色三星手机一部,又从我所巡防队员手中收缴该两名男子所贩卖的用透明塑料袋封装的毒品冰毒疑似物一小包(经称重,净重0.5克)。经讯问,该黄奎、杨政承认贩卖毒品冰毒的事实。',\
' 2014年12月27日早上8时许,吴鹏发现其在贵阳市云岩区东山村巫峰组星月浴室旁的租住房家中被溜门入室盗窃一部白色OPPO手机(型号N1,串号及卡号不详,2013年2月份购买,购价3300元人民币,现价值500元人民币)、一部黑色小米3手机(串号不详,卡号:15085973068,2014年3月份购买,购价1999元人民币,现价值1000元人民币)、一部黑色尼采手机(型号不详、串号不详及卡号不详,现价值100元人民币)、一台灰色惠普笔记本电脑(型号:i3,2013年10月份购买,购价3700元人民币,现价值2000元人民币)。',\
'2014年12月10日13时许,报警人庞影(女,1995年08月17日出生,汉族,初中文化程度,户籍所在地重庆市黔江区沙坝乡西泡村3组,现住贵阳市云岩区嘉华酒店112号房,现在无工作,居民身份证号码500239199508172045,联系电话18685475912)报称在延安西路“路尚鞋店”购物时其放在双肩包内的200元现金及一部白色的OPPO手机(购于2014年4月,串号不详,本机号18685475912,购价2400元)被盗。',\
' 2014年12月30日12时30分许,梅忠心(男,汉族,1990年01月06日生,身份证号码:522124199001064075,户籍:贵州省正安县中观镇鲜光村街上组 ,现住:贵阳市云岩区海马冲16号2号5楼7号,联系电话:18286084407),报称其在贵阳市云岩区海马冲因租房签订虚假合同被诈骗11000元人民币。',\
'2014年12月30日18时13分许,翁顺鸿(男,24岁,身份证号码:520123199005102413,户籍:贵州省修文县谷堡乡大寨村木家寨组,联系电话:13885137444。)在贵阳市云岩区大营坡农贸批发市场被扒窃,被盗走一部白色苹果6手机,串号不详,购于2014年10月,购价:5700元人民币。']

#db = client['dataframe']
#data = db.read('addr_gz_cont').data['ct'].values
#for i in text_sample:
#for i in coll.find()[20:120]:
addr_dict_lst = []

import pdb
#print(list(data)[1:10])
from dmp.gongan.ssc_dl_ner.data_utils import full_to_half
from doc_classify import Doc_Classify

livein = [\
            ['(?:暂住|现住[地]{0,}|家住|住址|租住[的]{0,}|所住)','[^vmp]+'],\
            ['(?:暂住|现住[地]{0,}|家住|住址|租住[的]{0,}|所住)','[,\)\']'],\
            ['^nsrm','家中'],\
            ['在/p','租房'],\
            ['在/p','房屋'],\
            ['在/p','屋'],\
            ['在/p','室'],\
            ['在/p ','\w*?房/n '],\
            ['租住','室','内'],\
            ['家中','被'],\
            ['住的','宿舍'],\
            ['住','./x'],\
            ['住',',/x'],\
            ['发现', '家中被'],\
            ['位于','家中被盗'],\
         ]

regisin = [
              ['户籍地籍贯/n ','\w+/[^vmp]+ '],\
              ['户籍[地]{0,1}','[,\)、]'],\
              ['户籍[地]{0,1}','[,\)]'],\
              ['籍[地]{0,1}','[,\)、]'],\
              ['户籍/n ./x ','./x '],\
              ['户籍[地]{0.}/n ./x ','./x '],\
              ['\(/x 身份证/n ./x ','\)/x'],\
              #['身份证/n 号码/n ','\w+/[^vmp] '],\
           ]

crim_act = ['盗走','盗','偷走','被盗','抢劫','盗取','抢夺','扒窃',\
'破坏','撬','撬坏','抢','拿走','砸坏','翻窗','抢走','剪断','偷',\
'撬开','砸','敲','撬门','强奸','打开','物品','溜门','损坏','拉开','扒窃']
crim_act_str ="('"+"'|'".join(crim_act)+"')"
#print(crim_act_str)
crimin = [\
             [r'\w+/v 在/p |家中',r'宿舍'],\
             ['在/p','宿舍'],\
             ['在/p','在/p'],\
             ['在/p','在/p','./x'],\
             ['在/p','在/p','在/p'],\
             ['(?:\w+/v 在/p |家中)','[租房屋]'],\
             ['(?:\w+/v 在/p |家中)','(发现|遗失)'],\
             ['(?:\w+/v 在/p |家中)','(?:\w+/[nv] 时/n){0,1} 被\w+/[nv]'],\
             ['(?:\w+/v 在/p |家中)','店[内门]'],\
             ['(?:\w+/v 在/p |家中)','门口'],\
             ['(?:在/p)',' 门口/s 停放/v '],\
             ['(?:在/p)',' 门口/s 被盗/n '],\
             ['(?:在/p)',' 被/p \w+/r '],\
             ['(?:在/p)',' \w+/r 时/n '],\
             ['放在/v ','内/f '],\
             ['在/p ','发现/v '],\
             ['在/p','被'],\
             ['[在去]+','[路上]+'],\
             ['发现','可疑'],\
             ['位于','被'],\
             ['在/p','发现'],\
             ['到','后','发现/v '],\
             ['在/p ','被/p '],\
             ['到/v ','开房/n '],\
             ['位于','被盗'],\
             ['\w+/v 在/p ','门口/s 的/uj '],\
             ['[在于]/p ',crim_act_str],\
             ['途径/n ',crim_act_str],\
             ["途径/n","被"],\
             ['经过/n ',crim_act_str],\
             #['在系','的于家中','[被骗盗抢偷]+'],\
             ['放在/v \w+/\w+ 的/uj ','内/n '],\
             ['放在/v \w+/\w+ 的/uj ','被\w*?/n '],\
             ['在/p \w+/(?:ns|nr|nz) ','被\w*?/n '],\
             ['在/p 其/r ','被\w*?/n '],\
             ['从/p ','到/v '],\
             ['在/p ','被/p ', '诈骗抢夺偷窃盗/v '],\
             ['赶至现场','./x '],\
             ['在/p','被盗'],\
             ['在/p','被'],\
             ['在/p','内','被'],\
             ['发现', '被'],\
             ['案发地', './x'],\
         ]

class Addr_Classify(object):

    def __init__(self, text):
        pass
        #print("\n> new Addr_Classify Instance")
        self.doc = Doc_Classify(text)
        #reg_registin = "%s(.+?)(?=\))|(?=[, ])"%i
        #with open("./apr_resu_kw.txt", "r") as f:
        #    lines = f.readlines()
        #    for line in lines:
        #        words = line.split(',')
        #        if len(words)>1:
        #            crimin.append(line.split(','))
        self.reg_pat_gen()


    def reg_pat_gen(self):
        formartter_2w = [
                 # "%s(.+?)%s",\
                 "(%s(.+?)%s)",\
                ]
        formartter_3w = [
                 "(%s(.+?)%s(.+?)%s)",\
                 #"%s(.+?)%s+.+?%s",\
                ]
        format_dic = {}
        format_dic['2'] = formartter_2w
        format_dic['3'] = formartter_3w
        pass
        self.crimin_lst = []
        self.livein_lst = []
        self.regisin_lst = []
        for i in crimin:
            #print(tuple(i))
            l = str(len(i))
            #print(format_dic[l])
            for j in format_dic[l]:
                reg = j%tuple(i)
                #print(reg)
                self.crimin_lst.append(re.compile(reg))
        for i in livein:
            l = str(len(i))
            #print(format_dic[l])
            #print(tuple(i))
            for j in format_dic[l]:
                reg = j%tuple(i)
                #print(reg)
                self.livein_lst.append(re.compile(reg))
        for i in regisin:
            #print("i> ", tuple(i))
            l = str(len(i))
            #print(format_dic[l])
            for j in format_dic[l]:
                reg = j%tuple(i)
                #print(reg)
                self.regisin_lst.append(re.compile(reg))

    def keyword_reg(livein, lin):
        lin = full_to_half(lin)
        words = list(jieba.posseg.cut(lin))
        text = ""
        for word in words:
            item = "%s/%s "%(word.word, word.flag)
            text+=item
        _text = full_to_half(text)
        addr_livein = []
        for i in livein:
            #reg_livein = "%s"%i
            #print("the rules and the results i text:")
            #print(i)
            #print(text)
            _ = re.findall(i, _text)
            #print("the rules and the results _:")
            #print(_)
            #print("\n",reg_livein, text)
            for m in _:
                if type(m) == tuple:
                    for n in m:
                        addr_livein.append(n)
                else:
                    addr_livein.append(m)
       # print("\n", lin, "\n",  text)
        results = set()
        for i in addr_livein:
            result = "".join(list(re.findall("(\w+)/\w+",i)))
            results.add(result)
        return list(results)

    def key_words_rule_reg_ext(self,sent):
        livein_lst = Addr_Classify.keyword_reg(self.livein_lst, sent)
        regisin_lst = Addr_Classify.keyword_reg(self.regisin_lst, sent)
        crimin_lst = Addr_Classify.keyword_reg(self.crimin_lst, sent)
        return livein_lst, regisin_lst, crimin_lst

    def ext_with_ner_predict(self, lin):
        #print("ext_with_ner_predict")
        #print(lin)
        #pdb.set_trace()
        words = []
        try:
            #print(lin)
            #print([lin])
            #pdb.set_trace()
            words = dmp.gongan.gz_case_address.predict.ner_predict([lin])[0]
        except KeyError:
            traceback.print_exc()
        return words

    def ext_with_posseg_reg(self,sent):
        chunk = self.doc.cut_word_vpn_pvn_chunk(sent)
       # print("\n> chunk:", chunk)
        return chunk

    def filter_add(self, lines):
        clr_lines = []
        for i in lines:
           # print("i=",i)
            if i== "":
                continue
            addr = ""
            words = list(jieba.posseg.cut(i))
            for word in words:
               # print(word.word)
               # print(word.flag)
                if word.flag!="nr" and word.flag!="ns" and word.flag!="m" and word.flag!="n" and word.flag!='v' and word.flag!='p' and word.flag!='j' and word.flag!='nz':
                    continue
                addr+=word.word
           # print("\n>add addr", addr)
            clr_lines.append(addr)
        return list(set(clr_lines))

    def run(self, sent):
        sent = full_to_half(sent)
        add_cls = {}
        add_cls['live'] = set()
        add_cls['reg'] = set()
        add_cls['crim'] = set()
        add_cls['text'] = sent
        lv, rg, cr = list(self.key_words_rule_reg_ext(sent))
        """
        jiazhong beidao
        """
        if sent.find('家中被') != -1:
            cr.extend(lv)
        if sent.find('家中被') != -1:
            lv.extend(cr)
        with open("with_sent_check.txt", "a+") as f:
            f.write("\n>sent:")
            f.write(str(sent))
            f.write("\n>lv:")
            f.write(str(lv))
            f.write("\n>rg:")
            f.write(str(rg))
            f.write("\n>cr:")
            f.write(str(cr))
            f.write("\n")
        tt = []
        tt.extend(lv)
       # print("lv>>>>>>>>>>>>>>>>>>")
       # print(lv)
        tt.extend(rg)
       # print("rg>>>>>>>>>>>>>>>>>>")
       # print(rg)
        tt.extend(cr)
       # print("cr>>>>>>>>>>>>>>>>>>")
       # print(cr)
        _tt = []
        for i in tt:
           if len(i)>0:
               _tt.append(i)
        tt = list(set(_tt))
        #addrs = tt#self.ext_with_ner_predict(sent)
        addrs = self.ext_with_ner_predict(sent)
        addrs = [re.sub("[^\w\d]","",addr) for addr in addrs]
        add_cls['base'] = addrs
       # print("tt", tt, type(tt))
        if len(addrs)>0:
            for addr in addrs:
                base_score = []
                base_sents=[]
                for i in tt:
                    isFlag = False
                   # print(addr,i)
                    _addr = full_to_half(addr)
                    #if _addr.find(i) != -1:
                    #    isFlag=True
                    if i.find(_addr) != -1:
                        isFlag=True
                    if isFlag:
                        if i in lv:
                           # print(_addr, "is in lv ", i)
                            add_cls['live'].add(addr)
                        if i in rg:
                           # print(_addr, "is in lv ", i)
                            add_cls['reg'].add(addr)
                        if i in cr:
                           # print(_addr, "is in lv ", i)
                            add_cls['crim'].add(addr)

        with open("check.txt", "a+") as f:
            f.write(str(add_cls))
            f.write("\n")
        return add_cls

if __name__ == "__main__":
    coll = pymongo.MongoClient("mongodb://127.0.0.1:27017")["myDB"]["original_data"]
    pdb.set_trace()
    text_ = []
    for i in coll.find():
        try:
           cont = i['casdetail']
           print(cont)
           assert type(cont) == str
           text_.append(cont)
        except:
           break
           #continue
    ac = Addr_Classify(text_[:2])
    result_lst=  []
    tcoll = pymongo.MongoClient("mongodb://127.0.0.1:27017")["myDB"]["traindata"]

    for i in text_:
        dct = {}
        #pdb.set_trace()
        # print(text_[i])
        result = ac.run(i)
        for j in result.keys():
           pass# print(j , result[j])

        dct['crim'] = ",".join(result['crim'])
        dct['text'] = cont
        dct['live'] = ",".join(result['live'])
        dct['reg'] = ",".join(result['reg'])
        tcoll.insert(dct)
        #tcoll.insert({"_id":i["_id"]},{"$set":dct})
        #pdb.set_trace()

    #df['crim_rw'] = False
    #df['reg_rw'] = False
    #df['live_rw'] = False
    #df['crim_rw'][df['crim']==df['predict_crim']] = True
    #df[df['crim']!=df['predict_crim']]['crim_rw'] = False
    #df['reg_rw'][df['reg']==df['predict_reg']] = True
    #df[df['reg']!=df['predict_reg']]['reg_rw'] = False
    #df['live_rw'][df['live']==df['predict_live']] = True
    #df[df['live']!=df['predict_live']]['live_rw'] = False
    #per = len(df[(df['live_rw']==True) & (df['live_rw']==True) & (df['live_rw']==True)])/len(df)
    #print("right %s per"%(str(per)))
    #df.to_csv("~/addr_classify/my_check.csv", index=False)



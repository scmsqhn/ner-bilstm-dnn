#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 01:14:21 2017

@author: qin haining 

@project: bond_risk

"""
from sklearn.utils import shuffle
from pandas.io.pytables import HDFStore
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing as pp
from matplotlib.font_manager import FontProperties
import sys,os
import numpy as np
#import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import classification_report
from collections import defaultdict
import traceback
import json
import datetime
import xgboost as xgb
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
sys.path.append('.')
sys.path.append('..')
import train
import  MyDebug
from MyDebug import debug_tool as d
import DataLoader as dl

sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages')
sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages/xgboost-0.6a2')
#sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages/')

global BASEDIR
BASEDIR = ""

SENTI_GROUP = ["债券风险","个人风险","财务风险","经营风险","行业风险","企业风险"]

RISK_LABEL = ['风险企业风险重大事项资产重组', '风险企业风险基本信息变更人员变动', '风险企业风险经营风险收入下降',
       '风险企业风险工商变更清算', '风险行业风险行业调整产能过剩', '企业风险司法诉讼涉诉', '企业风险债券风险债券违约',
       '风险企业风险重大事项关联风险', '企业风险重大事项关联风险', '企业风险股权结构股权收购', '企业风险经营事项股份质押',
       '风险企业风险公司公告股权结构', '风险企业风险重大事项股权变动', '风险企业风险公司公告决议公告', '企业风险经营风险资产恶化',
       '企业风险风险提示股票停牌', '风险企业风险经营风险破产重组', '企业风险管理风险管理层重大波动', '企业风险股权结构股权结构',
       '风险企业风险重大事项控股股东变更', '风险企业风险经营事项股份质押', '企业风险重大事项并购', '企业风险监管处罚违法违规',
       '风险企业风险重大事项收购', '风险个人风险高管风险', '企业风险重大事项资产重组', '企业风险工商变更人员变动',
       '企业风险行业风险产能问题', '风险企业风险重大事项决议公告', '风险企业风险公司公告融资融券', '风险企业风险管理风险管理层重大波动',
       '企业风险对外投资减持股份', '企业风险经营风险价格风险', '企业风险财务流动性风险违约债务', '企业风险股权结构增持股份',
       '风险企业风险风险提示股票停牌', '企业风险管理风险管理人员变更', '风险企业风险监管处罚违法违规', '企业风险管理风险争权夺利',
       '风险行业风险产品把控违法违规', '风险企业风险财务风险负债较多', '风险企业风险重大事项关联交易', '企业风险公司公告决议公告',
       '风险企业风险经营风险违约债务', '风险企业风险风险提示资产重组', '企业风险财务流动性风险亏损', '企业风险重大事项借贷担保',
       '风险企业风险重大事项借贷担保', '企业风险行业风险行业调整', '企业风险股权结构股份回购', '企业风险股权结构股权变动',
       '企业风险重大事项资产收购', '企业风险财务流动性风险负债较多', '企业风险对外投资投资风险', '风险企业风险经营事项提供担保',
       '企业风险管理风险高管风险', '企业风险财务流动性风险应收账款', '企业风险经营风险破产重组']


class Hdfs_stru():

    def __init__(self, hdfs):
        p('Hdfs_Data_Struc init')
        self.hdfs = hdfs
        self._f_dict = {}

    def map_index_columns(self):
        for i in self.hdfs.keys():
            _f_dict[i] = {}
            _f_dict[i]['index'] = _f_dict[i].index.values
            _f_dict[i]['cols'] = _f_dict[i].columns.values

    def get_f_dict(self):
        p(self._f_dict)




def dataparse(x):
    try:
#        print(type(x))
#        print(x)
        return pd.datetime.strptime(str(x)[0:10], '%Y-%m-%d') if not pd.isnull(x) else None
    except:
        print('=== STH IS WRONG !!! ===', x)
        traceback.print_exc()
        if (x != None):
            if (str(x)=='2016-023-0'):
                return pd.datetime.strptime('2016-02-03', '%Y-%m-%d')

            if (str(x)=='0    2017-12-07'):
                return pd.datetime.strptime('2017-12-07', '%Y-%m-%d')
        return None


class Data_helper():
    '''
    this is a tool of Data_helper()
    '''
    def __init__():
        pass
        p('[x] __init__')

def _code(i):
    ret = chardet.detect(i)
    return ret['encoding'], ret['confidence'], ret['language']

def set_base_dir(in_dir):
    try:
        global BASEDIR
        BASEDIR = in_dir
    except:
        p('[x] set_base_dir is wrong')
        traceback.print_exc()

def _rd_csv(filename):
    try:
        return pd.read_csv(filename)
        mgsId = struct.unpack("i",msg[:4])
    except:
        traceback.print_exc()

def _rd_xls(filename):
    try:
        return pd.read_excel(filename)
    except:
        traceback.print_exc()

def read_csv(filename, data_columns):
    try:
        p('[x] read file', os.path.join(BASEDIR, filename))
#        p('data_columns is', type(data_columns))
        file_data = pd.read_csv(os.path.join(BASEDIR, filename), 
            parse_dates=data_columns, 
            infer_datetime_format = True, 
            date_parser=dataparse)
        return file_data
    except:
        traceback.print_exc()
        return ""

def read_clear_file():
    try:
        p('[x] read_clear_file')
        company_df = read_csv('company_df.csv', ['成立日期'])
        bond_df = read_csv('bond_df.csv', ['报告期'])
        level_df = read_csv('level_df.csv', ['评级日期','公告日期'])
        public_sentiment_df = read_csv('public_sentiment_df.csv', ['舆情日期'])
        p(public_sentiment_df.columns)
        bond_fault_df = read_csv('bond_fault_df.csv', ['发生日期'])

        # merge all the data 
        part_1 = pd.merge(company_df, bond_df, how='outer', on='公司名称')
        part_2 = pd.merge(public_sentiment_df, level_df, how='outer', left_on='舆情日期', right_on='公司中文名称')
        part_3 = pd.merge(part_1, part_2, how='outer', left_on='公司名称', right_on='公司中文名称')
        # p(bond_fault_df.columns)
        total = pd.merge(part_3, bond_fault_df, how='outer', left_on='公司名称_x', right_on='发行人')
        #p(total.columns)
        # p(total.shape())
        # p(total.fillna().shape())
        return total
    except:
        traceback.print_exc()

def _less_name(x):
    map_2_norm = {"AAApi":"AAA","A-1+":"AAA+","A-1":"AAA","AAAsf":"AA+","A-2":"A+","AApi":"AA-",u"稳定":"AA-","":"AA-","已偿付":"AA-"}
    if x in map_2_norm:
        return map_2_norm[x]
    return x


def credit_level_2_score(inp):
    if pd.isnull(inp):
        return -1
    x = _less_name(inp)
#    print('level is ', x)
    mudy_level = {"Aaa":100,"Aa1":95,"Aa2":90,"Aa3":85,"A1":80,"A2":75,"A3":70,"Baa1":65,"Baa2":60, "Caa3":10,"Ca":5,"C":0}
    norm_level = {"AAA+":110,"AAA":105,"AAA-":100,"AA+":95,"AA":90,"AA-":85,"A+":80,"A":75,"A-":70,"BBB+":65,"BBB":60,"BBB-":55,"BB+":50,"BB":45,"BB-":40,"B+":35,"B":30,"B-":25,"CCC":20,"CC":15,"C":10, "RD":5,"D":0}
    try:
        return norm_level[x] if x in norm_level else(mudy_level[x])
    except:
        traceback.print_exc()
        return 100

def data_wash(df_input):
    df_output = df_input.drop(['Unnamed: 0_x_x', '公司ID', 'Unnamed: 0_y_x', 'Unnamed: 0_x_y', 'Unnamed: 0_y_y', 'Unnamed: 0'], axis=1)
    p(df_output.columns)
    df_output = df_input.drop(['公司中文名称', '公司名称_x'], axis=1)
    return df_output

def label_2_list(str_):
    lst_label = []
    for item in public_sentiment_2_lst(str_):
        print(item)
        lst_label.append(item)
    return lst_label


def base_label_2_list(lst_label):
    lst_label = []
    for str_ in lst_label:
        print(str_)
        for item in public_sentiment_2_lst(str_):
            print(item)
            lst_label.append(item)
    return lst_label

def public_sentiment_2_lst(cell_input):
        cell_input_txt = re.sub('[\->\[\]\'\" ]', '' , cell_input)
        cell_lst = cell_input_txt.split(',')
        print(cell_lst)
        return cell_lst

def public_sentiment_2_onehot(ser_input):
    '''
    input a columns of sentiment, which contain many sentiment lable like this ['a->b','c->d'], format is str
    '''
    total_label_lst = []
    for i in ser_input:
        i = re.sub('[\->\[\]\'\" ]', '' , str(i))
        #i = re.sub(r'["-",">"," "]',r'',str(i))
        ilst = i.split(',')
        for j in ilst:
            total_label_lst.append(j)
    onehot_matrix_sentiment = pd.get_dummies(list(set(total_label_lst)))
    return onehot_matrix_sentiment

def wr2csv(df_conf, file_name):
	df_conf.to_csv(file_name, index=True, sep=',',encoding = "utf-8")  

def group_by_sentiment(df_data):
    df_data_cp = df_data.copy()
    df_data['']

def calcu_label(self, bond):
    p('[x] calcu_label')
#        p(bond)
    bond_tmp = bond.copy()
#        p(bond_tmp)
    for i in bond_tmp.index:
        itm = bond_tmp.loc[i]
#            p(itm)
        startdate = itm['评级日期']
        enddate = itm['舆情日期']
        p('')
#           p(startdate,enddate)
        bond_tmp_tmp = bond_tmp[bond_tmp["pub_date"]>startdate & bond_tmp["pub_date"]<enddate]
#            p(bond_tmp_tmp)
#            lable_type = bond_tmp.iloc[:,'lable_type']
#          p(lable_type)

def base_count_senti(bond, cols):
    label_dict={}
    label_lst = []
    label_set = []
    for items in bond.loc[:,cols]:
        print(type(items), items)
        items = str(items)
        items = re.sub('[\->\[\]\'\" ]', '' , items)
        for item in items.split(','):
#            p('append',item)
            label_lst.append(item)
    return list(set(label_lst))

def base_count(bond, cols):
    label_dict={}
    label_lst = []
    label_set = []
    for item in bond.loc[:,cols]:
        label_lst.append(item)
    return list(set(label_lst))

def smart_count_label(ser_):
    p("smart_count_label")
    _ = []
    for lst in ser_.values:
        for item in lst:
            _.append(item)
    print('_',_)
    return _

def count_label(bond, cols):
    label_dict={}
    label_lst = []
    label_set = []
    for items in bond.loc[:,cols]:
        label_lst+=list(items)
    label_set = list(set(label_lst))
#        label_set = list(set(label_lst))
    return pd.Series(label_lst).value_counts()

def groupby_sentiment(inbond, cols):
    label_count = count_label(inbond, cols)

    for i in label_count.keys():
        inbond[i] = 0
#        bond['label_sum'] = bond[['评级日期','lable_type','舆情日期']].apply(lambda x : self.calcu_label(x))       
    for i in inbond.index:
#            p("[x] 以此打印index i = ",i)
        val_cnt = pd.Series(inbond.loc[i,'标签类型']).value_counts()
        for j in val_cnt.index:
            if j=="":
                continue
            try:
#                p('bond.loc[i,j] = ', i, j, val_cnt[j])
                inbond.loc[i,j] = val_cnt[j]
            except KeyError:
                p('KeyError')
                continue
    return inbond

def _groupby_sentiment(inbond, cols):
    p(inbond)
    lst_set_label = base_count_senti(inbond, '标签类型')
    p('lst_set_label', lst_set_label)
    lst_set_comp = base_count(inbond, '企业名称')
    p('lst_set_comp', lst_set_comp)
    _out_df  = pd.DataFrame(index=lst_set_comp, columns=lst_set_label)

    for i in lst_set_comp:
        _ = inbond[inbond['企业名称']==i]
        _label_lst = []
        _label_lst = smart_count_label(_['标签类型'])
        _label_lst_set = list(set(_label_lst))
        for x in _label_lst_set:
            _out_df.loc[i, x] = _label_lst.count(x)
        _out_df.loc[i, '企业名称'] = i
        _out_df.loc[i, '发布日期'] = pd.Series(list(set(_['发布日期']))).sort_values().values[-1]
        p('发布日期', _out_df.loc[i, '发布日期'])
    return _out_df






def groupby_sentiment(inbond, cols):
    label_count = count_label(inbond, cols)
    for i in label_count.keys():
        inbond[i] = 0
#        bond['label_sum'] = bond[['评级日期','lable_type','舆情日期']].apply(lambda x : self.calcu_label(x))       
    for i in inbond.index:
#            p("[x] 以此打印index i = ",i)
        val_cnt = pd.Series(inbond.loc[i,'标签类型']).value_counts()
        for j in val_cnt.index:
            if j=="":
                continue
            try:
#                p('bond.loc[i,j] = ', i, j, val_cnt[j])
                inbond.loc[i,j] = val_cnt[j]
            except KeyError:
                p('KeyError')
                continue
    return inbond

def base_groupby_sentiment(inbond, cols):
    label_count = count_label(inbond, cols)
    for i in label_count.keys():
        inbond[i] = 0
#        bond['label_sum'] = bond[['评级日期','lable_type','舆情日期']].apply(lambda x : self.calcu_label(x))       
    for i in inbond.index:
#            p("[x] 以此打印index i = ",i)
        val_cnt = pd.Series(inbond.loc[i,'标签']).value_counts()
        for j in val_cnt.index:
            if j=="":
                continue
            try:
                p('bond.loc[i,j] = ', i, j, val_cnt[j])
                inbond.loc[i,j] = val_cnt[j]
            except KeyError:
                p('KeyError')
                continue
    return inbond



def means_shift(X):
    '''
    ##产生随机数据的中心
    centers = df_matrix.sample(n=4)
    ##产生的数据个数
    n_samples=len(df_matrix.index)
    ##生产数据
    X, _ = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6, 
                      random_state =0)
    '''
    ##带宽，也就是以某个点为核心时的搜索半径
    p(X)
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10)
    ##设置均值偏移函数
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ##训练数据
    ms.fit(X)
    ##每个点的标签
    labels = ms.labels_
    p(labels)
    ##簇中心的点的集合
    cluster_centers = ms.cluster_centers_
    ##总共的标签分类
    labels_unique = np.unique(labels)
    ##聚簇的个数，即分类的个数
    n_clusters_ = len(labels_unique)
    p("number of estimated clusters : %d" % n_clusters_)
    return cluster_centers, labels,n_clusters_

def show_means_shift(X, cluster_centers, labels, n_clusters_):
    pass
    ##绘图
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    base_center = np.array(cluster_centers[0].shape)

    for k, col in zip(range(n_clusters_), colors):
        ##根据lables中的值是否等于k，重新组成一个True、False的数组
        my_members = labels == k
        cluster_center = cluster_centers[k]
        p('cluster_center:', cluster_center)
        ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
#        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def load_data():

    import pandas as pd
    label_vec = pd.read_csv('/home/siyuan/data/sentimention_label_vec.csv')
    comp_name_ = label_vec['Unnamed: 0']
    label_vec_ = label_vec.drop(['Unnamed: 1'],axis=1)
    label_vec_arr_ = np.array(label_vec_)
    sentimention_vec_arr = label_vec_arr_[:,1:]
    comp_vec_arr = label_vec_arr_[0]
    return comp_vec_arr, sentimention_vec_arr, comp_name_

def prt_2_file(txt_):
    pass
    with open('msg_output_doc.txt','a+') as f:
        f.writelines(txt_)
        f.writelines('\n')


def p(*i):
#    pass
    print(i)
    #sprt_2_file(str(i))

def pr(*i):
    pass
#    print(i)


def get_hdfs(f_name='/home/siyuan/data/bond_risk.h5'):
    store = HDFStore(f_name)
    hdfs_stru = Hdfs_stru(store)
    return hdfs_stru

def _load_hdfs(f_name='/home/siyuan/data/bond_risk.h5'):
    return HDFStore(f_name)

def load_hdfs(f_name='/home/siyuan/data/bond_risk.h5'):
    store = HDFStore(f_name)
    comp_56_label_ = store['obj1']
    level_df = store['label_vec_arr_el_df']
    company_df = store['company_df']
    bond_df = store['bond_df']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    bond_fault_df = store['bond_fault_df']
    sentimention_label_vec = store['sentimention_label_vec']
    public_sentiment_df = store['public_sentiment_df']
    total_df = store['total_df']
    total_fault_sentimention = store['total_fault_sentimention']
    total_sentimention = store['total_sentimention']    
    return store, comp_56_label_ ,level_df, company_df, bond_df, bond_fault_df, sentimention_label_vec,public_sentiment_df,total_df,total_fault_sentimention,total_sentimention

def _path(f_name):
    return os.path.join('/home/siyuan/data', str(f_name))

def sav_2_hdfs(store, *f_names):                                                                                                                                                                                                    
    cnt = 0
    for i in list(f_names[0][0]):
        p('csv cnt=', cnt)
        cnt+=1
        p('sav_2_hdfs', i)
        store.hdfs[i] = _rd_csv(_path(i))
    for j in list(f_names[0][1]):
        cnt+=1
        p('xls cnt=', cnt)
        store.hdfs[j] = _rd_xls(_path(j))
    return 0

def scan_dat_file(file_dir='/home/siyuan/data'):   
    L = []
    X = []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:
            if os.path.splitext(file)[1] == '.csv': 
                L.append(file)
            if os.path.splitext(file)[1] == '.xls': 
                X.append(file)
    return L, X

def pick_label(str_x):
    if pd.isnull(str_x):
        return ""

    p('str_x = ', str_x)
    p('type str_x = ', type(str_x))
    label_lst = []
    item = str_x
    item = re.sub('[\->\[\]\'\" ]', '' , item)
    for j in item.split(','):
        p(j)
        label_lst.append(j)
    p('label_lst = ', label_lst)
    return list(set(label_lst))

def lst_2_set(lst_x):
    p('lst_x', lst_x)
    lst_y = []
    str_y = ""
    for item in lst_x:
        str_y+=item
        str_y+=","
#        p('item = ', item)
    label_lst = pick_label(str_y)
    p(str(list(set(label_lst))))
    return str(list(set(label_lst)))

def ser_str_2_date(ser):
    for dates in ser:
        dates = pd.datetime.strptime(str(dates)[0:10], '%Y-%m-%d') if not pd.isnull(dates) else None
    return ser


def str_2_date(dates):
    return (pd.datetime.strptime(str(dates)[0:10], '%Y-%m-%d') if not pd.isnull(dates) else None)

def days_before_base(base, days):
    if pd.isnull(base):
        base = datetime.datetime.now()
    days = datetime.timedelta(days=days)
    return base-days

def is_dat_between(d_target, date1, date2):
    if pd.isnull(d_target):
        return 0
    if(d_target>date1):
        if(d_target<date2):
            return 1
    else:
        return 0

def is_dat1_less_dat2(date1, date2):
    if pd.isnull(date1) or pd.isnull(date2):
        return 1
    if(date1<date2):
        return 1
    if(date1>date2):
        return 0


def onehot_2_int(_onehot_lst):
    _lst = [str(i) for i in _onehot_lst]
    _str_lst = ''.joins(_lst)
    for i in _str_lst:
        pass

def first_risk(ser_date):
    ser_date_2 = [(str_2_date) for i in ser_date]
    return ser_date_2.sort_values().values[-1]

def label_one_hot(_df5):
    _df5['标签'] = _df5['标签类型_x'] + "," + _df5['标签类型_y'] + "," + _df5['标签类型']
    _df5['标签'] = _df5['标签'].apply(lambda x : pick_label(x))                                                                                                                                                                                                                                                                                                                                                                                 
    _df5 = _df5.drop(['标签类型','标签类型_x','标签类型_y'], axis=1)
    return _df5

def prepare_date_beat(_df5):
    bond_fault_df = pd.read_csv('bond_fault_df.csv')
    bond_fault_df['label'] = 1
    _df7['mask'] = 0
    _df7['发生日期'] = _df7['发生日期'].apply(lambda x : dataparse(x))
    _df7['发布日期'] = _df7['发布日期'].apply(lambda x : dataparse(x))

def prepare_data_(_df5):
    sentimention_groups = _df5.groupby(['企业名称','发布日期'])
    bond_fault_df = hdfs['bond_fault_df.csv']
    bond_fault_df['label'] = 1
    _df7 = groupby_sentiment(_df5, '标签')
    _df7['mask'] = 0
    _df7['发生日期'] = _df7['发生日期'].apply(lambda x : dataparse(x))
    _df7['发布日期'] = _df7['发布日期'].apply(lambda x : dataparse(x))

    ser_risk_date = _df7['发生日期'].groupby(_df7['企业名称']).min()
    ser_sentimention_date = _df7['发布日期'].groupby(_df7['企业名称']).max()
#
#     _df7.apply(lambda x : )
    '''
    # pick all label to list de-repeat
    _df6 = groupby_sentiment(_df5, '标签')
    print(bond_fault_df.columns)
    _df7 = pd.merge(_df6, bond_fault_df, how='inner', left_on='企业名称', right_on='发行人')
    _df7['mask'] = 0
    _df7['发生日期'] = _df7['发生日期'].apply(lambda x : dataparse(x))
    _df7['发布日期'] = _df7['发布日期'].apply(lambda x : dataparse(x))
    ser_risk_date = _df7['发生日期'].groupby(_df7['企业名称']).min()
    ser_sentimention_date = _df7['发布日期'].groupby(_df7['企业名称']).max()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    pr('ser_risk_date = ', ser_risk_date)
    '''
    for i in _df7.index:
        p(_df7.ix[i, '企业名称'] ,'is ondy')
        d_sentimetion = str_2_date(_df7.loc[i,'发布日期'])
        comp_name = _df7.loc[i,'企业名称']
        lst_comp_lost = []
        try:
            if comp_name in ser_risk_date.keys():
                d_risk = ser_risk_date[comp_name]
            elif comp_name in ser_sentimention_date.keys():
                d_risk = ser_sentimention_date[comp_name]
            else:
                d_risk = datetime.datetime.now()
                pr('comp_name is NULL')
                if comp_name in _df7.loc[:,'企业名称'].values:
                    pr(comp_name, 'has sentimention')
                else:
                    pr(comp_name, 'do not has sentimention')
                    lst_comp_lost.append(comp_name)
        except:
            traceback.print_exc()

        d_aft = days_before_base(d_risk, 1)
        d.pre = days_before_base(d_aft, 361)
        if is_dat_between(d_sentimetion, d.pre, d_aft):
            _df7.loc[i, 'mask'] = 1
        else:
            _df7.loc[i, 'mask'] = 0
    pr(str(lst_comp_lost), 'lost sentimention')
    _df8 = _df7[_df7['mask']==1]
#    _df8['企业名称'] = _df8['企业名称'].apply(lambda x : _df8['发行人'] if pd.isnull(x) else x)
#    _df8 = _df8.fillna(0)
    p('_df8.columns', _df8.columns)
    _df9 = _df8.drop(['标签', 'enterprise_id', 'Unnamed: 0', 'mask', '发行人'], axis=1)
    #_df10 = pd.get_dummies(_df9)
    label_cnt_dict = {}

    for label in _df9.columns:
        try:
            if label in ['企业名称', '发布日期', '发生日期', None]:
                continue
            p('sum of comp&label is =', _df9[label].groupby(_df9['企业名称']).sum())
            label_cnt_dict[label] = _df9[label].groupby(_df9['企业名称']).sum()
            p(label_cnt_dict[label])
        except:
            p(label, 'is error ')
            traceback.print_exc()
    _df10 = pd.DataFrame(label_cnt_dict)

    return(_df7, _df8 ,_df9, _df10, label_cnt_dict,lst_comp_lost)

def base_prepare_data_(_df5):
    # pick all label to list de-repeat
    _df6 = groupby_sentiment(_df5, '标签')
    bond_fault_df = hdfs['bond_fault_df.csv']
    bond_fault_df['label'] = 1
    print(bond_fault_df.columns)
    _df7 = pd.merge(_df6, bond_fault_df, how='inner', left_on='企业名称', right_on='发行人')
    _df7['mask'] = 0
    _df7['发生日期'] = _df7['发生日期'].apply(lambda x : dataparse(x))
    _df7['发布日期'] = _df7['发布日期'].apply(lambda x : dataparse(x))
    ser_risk_date = _df7['发生日期'].groupby(_df7['企业名称']).min()
    ser_sentimention_date = _df7['发布日期'].groupby(_df7['企业名称']).max()
    pr('ser_risk_date = ', ser_risk_date)
    for i in _df7.index:
        d_sentimetion = str_2_date(_df7.loc[i,'发布日期'])
        p('d_sentimetion = ', d_sentimetion)
#        p("_df7.loc[i,'发生日期']", _df7.loc[i,'发生日期'])
#        p("ser_risk_date[_df7.loc[i,'发生日期']]",ser_risk_date[_df7.loc[i,'发生日期']])
#        _df8['发生日期'] = _df8['发生日期'].apply(lambda x : datetime.datetime.now() if pd.isnull(x) else x)
        
        comp_name = _df7.loc[i,'企业名称']
        lst_comp_lost = []

        #pr('comp_name', comp_name)
        try:
            if comp_name in ser_risk_date.keys():
                d_risk = ser_risk_date[comp_name]
            elif comp_name in ser_sentimention_date.keys():
                d_risk = ser_sentimention_date[comp_name]
            else:
                pr('comp_name is NULL')
                if comp_name in _df7.loc[:,'企业名称'].values:
                    pr(comp_name, 'has sentimention')
                else:
                    pr(comp_name, 'do not has sentimention')
                    lst_comp_lost.append(comp_name)
        except:
            traceback.print_exc()

        pr(str(lst_comp_lost), 'lost sentimention')
        _df7.loc[i,'发生日期'] = d_risk
        d_aft = days_before_base(d_risk, 1)
        d.pre = days_before_base(d_aft, 361)
        if is_dat_between(d_sentimetion, d.pre, d_aft):
            _df7.loc[i, 'mask'] = 1
        else:
            _df7.loc[i, 'mask'] = 0
    _df8 = _df7[_df7['mask']==1]
#    _df8['企业名称'] = _df8['企业名称'].apply(lambda x : _df8['发行人'] if pd.isnull(x) else x)
#    _df8 = _df8.fillna(0)
    p('_df8.columns', _df8.columns)
    _df9 = _df8.drop(['标签', 'enterprise_id', 'Unnamed: 0', 'mask', '发行人'], axis=1)
    #_df10 = pd.get_dummies(_df9)
    label_cnt_dict = {}

    for label in _df9.columns:
        try:
            if label in ['企业名称', '发布日期', '发生日期', None]:
                continue
            p('sum of comp&label is =', _df9[label].groupby(_df9['企业名称']).sum())
            label_cnt_dict[label] = _df9[label].groupby(_df9['企业名称']).sum()
            p(label_cnt_dict[label])
        except:
            p(label, 'is error ')
            traceback.print_exc()
    _df10 = pd.DataFrame(label_cnt_dict)

    return(_df9, _df10, label_cnt_dict)
def zero_or_not(_x):
    if _x == 0:
        return 0
    elif _x > 0:
        return 1
    else:
        return 0

#    elif _x > 0:
#        return 1

# import sys, getopt

# def main(argv):
#     inputfile = ""
#     outputfile = ""
#     try:
#         # 这里的 h 就表示该选项无参数，i:表示 i 选项后需要有参数
#         opts, args = getopt.getopt(argv, "hi:o:dt",["infile=", "outfile="])
#     except getopt.GetoptError:
#         sys.exit(2)

#     for opt, arg in opts:
#         if opt == "-h":
#             sys.exit()
#         elif opt in ("-i", "--infile"):
#             inputfile = arg
#         elif opt in ("-o", "--outfile"):
#             outputfile = arg

#         elif opt in ("-d"):
#             from pandas.io.pytables import HDFStore
#             import re
#             import numpy as np
#             import pandas as pd
#             import matplotlib as mpl
#             import matplotlib.pyplot as plt
#             import seaborn as sns
#             import sklearn
#             from sklearn import preprocessing as pp
#             from matplotlib.font_manager import FontProperties
#             import sys,os
#             import numpy as np
#             #import xgboost as xgb
#             from sklearn import preprocessing
#             from sklearn.metrics import classification_report
#             from collections import defaultdict
#             import traceback
#             import json
#             import datetime
#             import xgboost as xgb

#             from sklearn.datasets.samples_generator import make_blobs
#             from sklearn.cluster import MeanShift, estimate_bandwidth
#             import numpy as np
#             import matplotlib.pyplot as plt
#             from itertools import cycle  ##python自带的迭代器模块
#             sys.path.append('.')
#             sys.path.append('..')
#             import train

#             sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages')
#             sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages/xgboost-0.6a2')
#             #sys.path.append('/home/siyuan/anaconda3/envs/p36/lib/python3.6/site-packages/')

#             sample_rate = 0.01
#             Hdfs = get_hdfs()
#             sav_2_hdfs(Hdfs, tuple(scan_dat_file()))
#             hdfs = _load_hdfs()                                                                                                                                                                                                         
#             _df1 = Hdfs.hdfs['56家债券企业2017-12-26.xls']
#             _df2 = Hdfs.hdfs['全部违约企业-舆情标签信息.csv']
#             _df3 = Hdfs.hdfs['舆情.csv']                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
#             _df4 = pd.merge(_df1, _df2, how='outer', on=['企业名称','发布日期']) 
#             _df5 = pd.merge(_df4, _df3, how='outer', on=['企业名称','发布日期']) 

#             _df5_train = _df5.sample(frac=sample_rate)
#             _ = label_one_hot(_df5_train)
#             _p1, datas, _1 = prepare_data_(_)
#             datas['label'] = datas['label'].fillna(0)
#             i = int(datas.shape[0]*0.7)    

#             train_data = datas.iloc[:i,:]
#             test_data = datas.loc[i:,:]

#             train_Y = train_data['label'].apply(lambda x : zero_or_not(x)) #.fillna(0)
#             train_X = train_data.drop(['label'],axis=1)#.fillna(0)

#             test_Y = test_data['label'].apply(lambda x : zero_or_not(x)) #.fillna(0)
#             test_X = test_data.drop(['label'],axis=1)#.fillna(0)

#         elif opt in ("-t"):
#             train.run(train_X, train_Y,test_X, test_Y, num_round=2)

def read_csv():
    _csv1 = pd.read_csv('/home/siyuan/data/全部违约企业-舆情标签信息.csv')#(5480, 3)
    _csv2 = pd.read_csv('/home/siyuan/data/舆情.csv')#(1490073, 3)
    _csv3 = pd.read_excel('/home/siyuan/data/56家债券企业2017-12-26.xls')#(761, 4)
    _csv4 = pd.read_csv('/home/siyuan/data/舆情_01.csv')#(761, 4)
#    _csv5 = pd.read_csv('/home/siyuan/data/zq_zt.csv')#(761, 4)
    return _csv1, _csv2, _csv3 , _csv4#, _csv5

def data_clear():
    _f1, _f2, _f3 , _f4 = read_csv()#, _f5= read_csv()
    _f3 = _f3.drop(['enterprise_id'],axis=1)
    _csv = pd.concat([_f1,_f2,_f3,_f4])
    _csv = _csv.dropna()
    return _csv#, _f5
'''
def handle_credit(_df_, _f5):
    p(_df_.columns)
    p(_f5.columns)
    p('merger the sentimention and credit msg!')
    _df_['credit_recent'] = -1
    _df_['credit_ago'] = -1
    _ = pd.merge(_df_,_f5,how='left', on='企业名称')
    _ = _[_['日期']!=None]
    _['日期'] = _['日期'].apply(lambda x : dataparse(x))
    _ = _[_['发布日期']>_['日期']]
    for i in list(set(_['企业名称'])):
        _cp = _[_['企业名称']==i]
        credit_list = pd.Series(list(set(_cp['日期']))).sort_values().values
        if len(credit_list)>1:
            p('len > 1')
            recent = pd.Series(list(set(_cp['日期']))).sort_values().values[-1]
            _[_['企业名称']==i]['credit_recent'] = _cp[_cp['日期']==recent]['评级']
            ago = pd.Series(list(set(_cp['日期']))).sort_values().values[-2]
            _[_['企业名称']==i]['credit_ago'] = _cp[_cp['日期']==ago]['评级']
        if len(credit_list)>0:
            p('len > 0')
            recent = pd.Series(list(set(_cp['日期']))).sort_values().values[-1]
            _[_['企业名称']==i]['credit_recent'] = _cp[_cp['日期']==recent]['评级']
    return _
'''

def credit_recent(x, _df_, _f5):
  try:
    _ = _df_[_df_['企业名称']==x]
    _c = _f5[_f5['企业名称']==x]
    date = _['发布日期'].min()
    _date_c = _c[_c['日期'] < date]
    p(_date_c, type(_date_c))
    recent =  _date_c.sort_values('日期').dropna()
    if(len(recent)>0):
        return recent.iloc[-1,:]['评分']
    return np.NaN

  except:
    traceback.print_exc()
    return np.NaN



def credit_ago(x, _df_, _f5):
  try:
    _ = _df_[_df_['企业名称']==x]
    _c = _f5[_f5['企业名称']==x]
    date = _['发布日期'].min()
    _date_c = _c[_c['日期'] < date]
    p(_date_c, type(_date_c))
#    recent =  zq_zt.sort_values('日期').dropna().iloc[-1,:]['评分']
    ago = _date_c.sort_values('日期').dropna()
    if(len(ago)>1):
        return ago.iloc[-2,:]['评分']
    return np.NaN
  except:
    traceback.print_exc()
    return np.NaN




def handle_credit_simply(_df_, _f5):
    _df_ = _df_.copy()
    cnt = 0
    _df_['credit_recent'] = -1
    _df_['credit_ago'] = -1
    _f5['日期'] = _f5['日期'].apply(lambda x : dataparse(x))
    _f5['评分'] = _f5['评级'].apply(lambda x : credit_level_2_score(x))
    _df_['credit_recent'] = _df_['企业名称'].apply(lambda x : credit_recent(x, _df_, _f5))
    _df_['credit_ago'] = _df_['企业名称'].apply(lambda x : credit_ago(x, _df_, _f5))
    p(_df_['credit_recent'])
    return _df_

    # =====  divider for highlight


def handle_credit_smart(_df_, _f5):
    #d.p('handle_credit_smart(_df_, _f5):')

    _df_ = _df_.copy()
    #d.p(_df_.columns)
    #d.p(_f5.columns)
    cnt = 0
    _df_['credit_recent'] = -1
    _df_['credit_ago'] = -1
    _df_['credit_trend'] = -1
    # _df_['credit_recent'] = pd.NaT
    # _df_['credit_ago'] = pd.NaT
    # _df_['credit_trend'] = pd.NaT
    _f5['日期'] = _f5['日期'].apply(lambda x : dataparse(x))
    _f5['评分'] = _f5['评级'].apply(lambda x : credit_level_2_score(x))
    for i in list(set(_df_['企业名称'])):
        cnt+=1
        date_line = dataparse(_df_[_df_['企业名称']==i]['发布日期'].max())
        _cp = _f5[(_f5['公司名称']==i) & (date_line>_f5['日期'])]
        early_moment  = _cp['评级'].groupby(_cp['日期']).keys.min()
        p(early_moment)
        if early_moment<date_line:
            date_line = early_moment
        p(date_line)
        p('_cp contain: ')
        p(_cp)
        p('this is', cnt)
#        early_moment  = _cp['评级'].groupby(_cp['日期']).keys.min()
#        _['发布日期']
        max_ = pd.Series(_cp['评分'].groupby(_cp['日期']).max())
        print(max)
        if len(max_)>1:
            _df_[_df_['企业名称']==i]['credit_ago'].apply(lambda x : max_.sort_index()[-1])
            _df_[_df_['企业名称']==i]['credit_recent'].apply(lambda x : max_.sort_index()[-2])
            _df_[_df_['企业名称']==i]['credit_trend'] = _df_[_df_['企业名称']==i]['credit_recent']-_df_[_df_['企业名称']==i]['credit_ago']
            p('match 2')
            break
        elif len(max_)>0:
            _df_[_df_['企业名称']==i]['credit_ago'].apply(lambda x : max_[-1])
            _df_[_df_['企业名称']==i]['credit_trend'] = pd.NaT
            p('match 1')
            break
        else:
            _df_[_df_['企业名称']==i]['credit_trend'] = pd.NaT
            pass
            p('match nothing')
    _df_.to_csv('temp.csv')
    return _df_

def pickmax1(x):
    print(x)
    return x.max()[-1]

def pickmax2(x):
    print(x)
    return x.max()[-2]

def data_load(date_pre=180, date_aft=0):
    p("date_pre, date_aft", date_pre, date_aft)
    cnt = 1
    _csv = data_clear()
    #_senti_credit = pd.merge(_csv,_f5,how='outer', on='企业名称')
    #return _senti_credit
    group_comp_date = _csv.groupby(['企业名称','发布日期']).groups.keys()
    lset_group_comp = list(set(_csv.groupby(['企业名称']).groups.keys()))
    lset_group_date = list(set(_csv.groupby(['发布日期']).groups.keys()))

    dict_comp_date = _csv['发布日期'].groupby(_csv['企业名称']).max()
    _df_ = pd.DataFrame(index=[], columns=['发布日期','企业名称','标签类型'])
    for i in dict_comp_date.keys():
        _date = days_before_base(dataparse(dict_comp_date[i]), date_pre)
#        print(_date, type(_date))
        _ = _csv[_csv['企业名称']==i]
        _['发布日期'] = _['发布日期'].apply(lambda x : dataparse(x))
        _ = _[_['发布日期'] > _date]
        _df_ = pd.concat([_df_, _])
        cnt +=1

        if cnt%10 == 1:
            print('######')
            print('      ')
            print(i)
            print('      ')
            print('######')

        if cnt%100==1:
            print('[x]', cnt, 'comp has be handled')

#        if cnt > 500:
#            break

    p('comp name is [no fault concat]: ', _df_['企业名称'])

    bond_fault_df = pd.read_csv('/home/siyuan/data/bond_fault_df.csv')
    bond_fault_df['label'] = 1
    _df_['label']=0


    bond_fault_df['企业名称'] = bond_fault_df['发行人']
    #bond_fault_df['发布日期'] = bond_fault_df['发生日期']
    _df_ = _df_[(True-_df_['企业名称'].isin(list(set(bond_fault_df['企业名称']))))]
    print(_df_.shape)

    for i in bond_fault_df['企业名称']:
        _ = _csv[_csv['企业名称']==i]
        _date_ = bond_fault_df['发生日期'].groupby(bond_fault_df['企业名称']).min()[i]
        # !!! caution some company data will lost here because data-filter below can not catch them
        _date_1 = days_before_base(dataparse(_date_), date_pre)
        _date_2 = days_before_base(dataparse(_date_), date_aft)
        _['发布日期'] = _['发布日期'].apply(lambda x : dataparse(x))
        _ = _[_['发布日期']>_date_1]
        _ = _[_['发布日期']<_date_2]

        _df_ = pd.concat([_df_, _])
    _set = set(bond_fault_df['企业名称'])
    print('fault_company:' ,_set)
    _df_['label'] = _df_['企业名称'].apply(lambda x : isBad(x, _set))
    print(_df_.shape)#
    _df_.to_csv('_df_.csv')
    print('* ALL THE LABLE COUNT IS ', list(_df_['label'].values).count(1))
    p(_df_['企业名称'])
    #p(_df_)
    return _df_

def isBad(x, _set):
    if x in _set:
        print(x, 'is in ', _set)
        return 1
    else:
        print(x, 'is not in ', _set)
        return 0

def cut_str_2_lst(_df9):

    # label_cnt_dict = {}
#     for label in _df9.columns:
#         try:
# #            if label in ['企业名称', '发布日期', '发生日期', None]:
# #                continue
#             p('sum of comp&label is =', _df9[label].groupby(_df9['企业名称']).sum())
#             label_cnt_dict[label] = _df9[label].groupby(_df9['企业名称']).sum()
#             p(label_cnt_dict[label])
#         except:
#             p(label, 'is error ')
#             traceback.print_exc()
#     _df10 = pd.DataFrame(label_cnt_dict)
    _df9['标签类型'] = _df9['标签类型'].apply(lambda x : label_2_list(x))
    return _df9




def label_check(x, _fault):
    if x in set(_fault['企业名称']):
        return 1
    else:
        return 0

def get_zq_zt(file_path='/home/siyuan/data/zq_zt.csv'):
    pass
    return pd.read_csv(file_path)

def date_check(x, _fault):
    try:
        if x in set(_fault['发行人']):
            return _fault[_fault['发行人']==x]['发生日期'].min()
    except:
        traceback.print_exc()
        p('this comp is not fault ')

def sava_age_dat_2_csv(pre, aft):
    da_60_0  = cut_str_2_lst(data_load.pre,aft)
    da_60_0.to_csv('%d_%d_senti_data.csv' % (pre,aft))

def calcu_senti_date_2(comp_name, pre, aft):
    my_lst = []
    #assert not (my_lst==None)
    #d.prt_title('calcu_senti_date')
    #d.p('my_lst', type(my_lst))

    if (pre==60, aft==0):

        da_60_0 = cut_str_2_lst(pd.read_csv('60_0_senti_data.csv'))
        _da60 = da_60_0[da_60_0['企业名称']==comp_name]
        for i in _da60['标签类型']:
            if i == None:
                #d.p('i is None')
                continue
            if ',' in i:
                for j in i.split(','):
                    my_lst.append(j)
                    #d.p('my_lst_append', j)
            else:
                my_lst.append(str(i))
                #d.p('my_lst_append', i)
        cnt_60_0 = len(my_lst)
        return cnt_60_0

    if (pre==120, aft==61):
        da_120_61 = cut_str_2_lst(pd.read_csv('120_61_senti_data.csv'))
        _da120 = da_120_61[da_120_61['企业名称']==comp_name]
        for i in _da120['标签类型']:
            if i == None:
                #d.p('i is None')
                continue
            if ',' in i:
                for j in i.split(','):
                    my_lst.append(j)
                    #d.p('my_lst_append', j)
            else:
                my_lst.append(str(i))
                #d.p('my_lst_append', i)
        cnt_120_0 = len(my_lst)
        return cnt_120_0

    if (pre==180, aft==121):
        da_180_121 = cut_str_2_lst(pd.read_csv('180_121_senti_data.csv'))
        _da180 = da_180_121[da_180_121['企业名称']==comp_name]
        for i in _da180['标签类型']:
            if i == None:
                #d.p('i is None')
                continue
            if ',' in i:
                for j in i.split(','):
                    my_lst.append(j)
                    #d.p('my_lst_append', j)
            else:
                my_lst.append(str(i))
                #d.p('my_lst_append', i)
        cnt_180_0 = len(my_lst)
        return cnt_180_0
    else:
        return -1

def calcu_senti_date(comp_name, pre, aft):
    my_lst = []
    #assert not (my_lst==None)
    #d.prt_title('calcu_senti_date')
    #d.p('my_lst', type(my_lst))

    da_60_0  = cut_str_2_lst(data_load.pre,aft)
    da_60_0.to_csv('%d_%d_senti_data.csv' % (pre,aft))
    _ = da_60_0[da_60_0['企业名称']==comp_name]

    for i in _['标签类型']:
        if i == None:
            #d.p('i is None')
            continue
        if ',' in i:
            for j in i.split(','):
                my_lst.append(j)
                #d.p('my_lst_append', j)
        else:
            my_lst.append(str(i))
            #d.p('my_lst_append', i)
    cnt_60_0 = len(my_lst)
    for i in my_lst:
        d.p(i)
    d.v(cnt_60_0)
    return cnt_60_0
    '''
    lst=[]
    da_120_61  = cut_str_2_lst(data_load(120,61))
    for i in da_120_61['标签类型'].dropna():
        if ',' in i:
            for j in i.split(','):
                lst.append(j)
            lst.append(i)
        lst = lst.append(i)
    cnt_120_61 = len(lst)

    lst=[]
    da_180_121  = cut_str_2_lst(data_load(180,121))
    for i in da_180_121['标签类型'].dropna():
        if ',' in i:
            for j in i.split(','):
                lst.append(j)
            lst.append(i)
        lst = lst.append(i)
    cnt_180_121 = len(lst)
    d.v(cnt_60_0)
    d.v(cnt_120_61)
    d.v(cnt_180_121)
    '''

def process_data(threadName, q, mom_dict, cnt):
    global exitFlag
    while not exitFlag:
        #d.p(exitFlag)
        queueLock.acquire()
        if not q.empty():
            resolve = q.get()
            d.v(resolve)
            result = pc.FUNC_DICT[resolve](mom_dict)
            d.v(result)
            queueLock.release()
            #d.p(("%s processing %s" % (threadName, resolve)))
        else:
            cnt-=1 
            #d.p(cnt)
            if cnt<0:
                #exitFlag = 1
                #d.p('we have round 100 now we can leave')
                exitFlag = 1
            queueLock.release()
        time.sleep(1)

def  load_data_diff_senmi(m_dict):
    import DataLoader as dl
    total_data = dl.load_data('bond_risk_total_18-01-17')
    total_credit = dl.load_data('zq_zt')
#            print(_zq_zt.head())
    total_data = handle_credit_smart(total_data, total_credit)
    senmi_data = dl.load_data('senmi_cnt')
    #d.p(total_data.head())
    #d.p(senmi_data.head())
    total_data = pd.merge(total_data, senmi_data, how='left', left_on='企业名称', right_on='Unnamed: 0')
    total_data['sub120_60'] = total_data['120'] - total_data['60']
    total_data['sub180_120'] = total_data['180'] - total_data['120']
    now_str = dl.now_str()
    filename = 'total_data%s'%now_str
    m_dict['total_data'] = filename
    dl.sav_data(total_data, m_dict['total_data'])
    return total_data

def load_data_diff_senmi_direc(m_dict):
    m_dict['total_data'] = 'total_data_risk_18_2'
    return dl.load_data('total_data_risk_18_2')

def senti_group_count():
    _x = pd.read_csv('/home/siyuan/data/_x.csv') 
    bond_fault_df = pd.read_csv('/home/siyuan/data/bond_fault_df.csv') 
    _x_groups = _x.groupby("企业名称").groups
    _x_groups_keys = _x.groupby("企业名称").groups.keys()
    calc_df = pd.DataFrame()
    for i in list(_x_groups_keys):
        cnt_lst = []
        #_x_groups[i]
        print(i)
        #print(list(_x_group[i]))
        bond_fault_df_min = bond_fault_df.groupby(bond_fault_df['发行人']).min()
        time_deadline=0
        try:
            time_deadline = bond_fault_df_min[i]
        except:
            time_deadline = str(datetime.datetime.now().date())
            print('[x] there is no fault happen')
        #d.p('this comp name is ', i )
        dict_3 = calcu_senti_group(i, list(_x_groups[i]), time_deadline)
        for p in dict_3.keys():
            calc_df.loc[i, p] = dict_3[p]
    calc_df.to_csv('sentimention_group_trend.csv')

def calcu_senti_group(i , lst, time_deadline):
    dict_cnt = dict()
    for x in SENTI_GROUP:
        for y in ["60","120","180"]:
            dict_cnt["%s%s"%(x,y)] = 0
    _x = pd.read_csv('/home/siyuan/data/_x.csv') 
    my_lst = []
    #assert not (my_lst==None)
    #d.prt_title('calcu_senti_date')
    #d.p('my_lst', type(my_lst))
    for index in lst:
        t = pd.to_datetime(_x.loc[index]['发布日期'])
        t60 = pd.to_datetime(time_deadline) - datetime.timedelta(60)
        t120 = pd.to_datetime(time_deadline) - datetime.timedelta(120)
        t180 = pd.to_datetime(time_deadline) - datetime.timedelta(180)
        print(t,t60,t120,t180)
        if t<t60:
            items = _x.loc[index].dropna().drop(['发布日期','企业名称','Unnamed: 0','Label'])
            for j in items.keys():
                print(j)
                for m in SENTI_GROUP:
                    if m in j:
                        print(m,"is in", j)
                        dict_cnt["%s60"% m]+=1
        elif t<t120:
            items = _x.loc[1].dropna().drop(['发布日期','企业名称','Unnamed: 0','Label'])
            for j in items.keys():
                print(j)
                for m in SENTI_GROUP:
                    if m in j:
                        print(m,"is in", j)
                        dict_cnt["%s120"% m]+=1
        elif t<t180:
            items = _x.loc[1].dropna().drop(['发布日期','企业名称','Unnamed: 0','Label'])
            for j in items.keys():
                print(j)
                for m in SENTI_GROUP:
                    if m in j:
                        print(m,"is in", j)
                        dict_cnt["%s180"% m]+=1
    return dict_cnt
def main():
#    date_list_ = [[180,0],[90,0],[30,0],[180,90],[90,30],[180,60]]
    if False:
        _x = pd.read_csv('/home/siyuan/data/_x.csv') 
#        _x = _x.iloc[:100,:]
        lst_cmp_name = list(set(_x['企业名称']))
        calc_df = pd.DataFrame(columns=['comp_name', 'senti_60_0', 'senti_120_61', 'senti_180_121'])
        #d.p('there r total ', len(lst_cmp_name), 'comp here' )
        for i in lst_cmp_name:
            #d.p('this comp name is ', i )
            calc_df.loc[i, 'senti_60_0'] = calcu_senti_date_2(i, 60, 0)
            calc_df.loc[i, 'senti_120_61'] = calcu_senti_date_2(i, 120, 61)
            calc_df.loc[i, 'senti_180_121'] = calcu_senti_date_2(i, 180, 121)
        calc_df.to_csv('sentimention_trend.csv')
        #d.prt_title('compelity!')
    else:
      date_list_ = [[180,0]]
      for i in date_list_:
        if False:
            _base = cut_str_2_lst(data_load(i[0], i[1]))
        if False: 
            _base = cut_str_2_lst(pd.read_csv('/home/siyuan/data/_df_.csv'))
            _y = _groupby_sentiment(_base, '标签类型')
            bond_fault_df = pd.read_csv('/home/siyuan/data/bond_fault_df.csv')
            bond_fault_df['违约'] = 1
            bond_fault_df['企业名称'] = bond_fault_df['发行人']
            _fault = bond_fault_df[['企业名称','违约']]
    #        _y['企业名称'] = _y.index
            _y['Label']= _y['企业名称'].apply(lambda x : label_check(x,_fault))
            _y['发布日期']= _y['企业名称'].apply(lambda x : date_check(x,_fault))

            _x = _y#.drop(['企业名称'],axis=1)
            _zq_zt  = pd.read_csv('/home/siyuan/data/zq_zt.csv')
            _x = handle_credit_smart(_x,_zq_zt)
            _senmi_cnt = pd.read_csv('/home/siyuan/data/senmi_cnt.csv')
            _x = pd.merge(_x, _senmi_cnt, how='left', left_on='企业名称', right_on='Unnamed: 0')
            _x['sub120_60'] = _x['120'] - _x['60']
            _x['sub180_120'] = _x['180'] - _x['120']
        if True:
            import DataLoader as dl
            #d.prt_title("thie is for senti_group test now")
            _x = pd.read_csv('/home/siyuan/data/_x.csv') 
            sentimention_group_trend = pd.read_csv('/home/siyuan/data/sentimention_group_trend.csv') 
            for i in ["债券风险120","企业风险120"]:#:sentimention_group_trend.columns:
                try:
                    sentimention_group_trend.loc[:,i] = sentimention_group_trend.dropna().loc[:,i].apply(lambda x : (1.0 / (1.0 + np.exp(-float(x)))))
                except:
                    print(i)
                    continue
            d.v(sentimention_group_trend)
            _zq_zt  = pd.read_csv('/home/siyuan/data/zq_zt.csv')
            total_data = dl.load_data('bond_risk_total_18-01-17')
            total_credit = dl.load_data('zq_zt')
            _x = handle_credit_smart(_x, _zq_zt)
            _senmi_cnt = pd.read_csv('/home/siyuan/data/senmi_cnt.csv')
            _x = pd.merge(_x, _senmi_cnt, how='left', left_on='企业名称', right_on='Unnamed: 0')
            _x = pd.merge(_x, sentimention_group_trend, how='left', left_on='企业名称', right_index=True)
            #d.prt_title("_x final data struct")
            d.v(_x.tail())
            _x['sub120_60'] = _x['120'] - _x['60']
            _x['sub180_120'] = _x['180'] - _x['120']
        p('watting i am shuffle ,for label >6')
        #d.prt_lin()
        try:
            _z = shuffle(_x).drop(['Unnamed: 0_x','Unnamed: 4','Unnamed: 0_y'], axis=1)
        except:
            _z = shuffle(_x)
        try:
           _z = _z.drop(['Unnamed: 0'], axis=1)
           _z = _z.drop(['Unnamed: 0_x'], axis=1)
        except:
           _z = _z
        #d.prt_lin()
        [print(i) for i in _z.columns]
        _z.to_csv('_z.csv')
        [print(i) for i in _z.iloc[1,:]]
        train_, eval_, bst = train.run(_z)
        nm_z = _z.drop(['发布日期'], axis=1)
        i = int(_z.shape[0]*0.5)
        eval_nm_ = _z.iloc[i:,:]
        d.p("eval_.tail()")
        d.p(eval_.tail())
        d.p("eval_nm_")
        d.p(eval_nm_)
        #dict_all = train.judge_modol(eval_, eval_nm_)
        dict_all = train.judge_modol_bst(eval_, eval_nm_, bst)
        train.create_feature_map(train_.columns, "/home/siyuan/data/xgb.fmap")
        bst.save_model('/home/siyuan/xgb.model')
        f_score = train.calc_fscore_bst(bst)
        d.v(f_score)

        for i in dict_all:
            print(i, dict_all[i])

if __name__ == '__main__':
    #main()
    import  bondrisk_tsne_dbscan
    y_pred, df = bondrisk_tsne_dbscan.main2()
    import pdb
    pdb.set_trace()



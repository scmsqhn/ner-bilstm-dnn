#coding=utf-8

# data_helper.py

# sql handler for all mysql of my 
import pdb

#pdb.set_trace()
import load_mysqL_from_localcpk
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
import pymysql
import datetime
import pymysql  
import traceback
import pandas as pd
import xgboost as xgb
import numpy as np
import re


def timeFormat(_strs):
    #print(_strs)
    num = re.findall("\d+", str(_strs))
    #print(num)
    _append=6-len(num)
    [num.append('0') for i in range(_append)]
    return datetime.datetime.strptime(",".join(num), '%Y,%m,%d,%H,%M,%S')

model_labels_2 = ["企业名称","发布日期","Label","credit_recent","credit_ago","credit_trend","60","120","180","债券风险60","债券风险120","债券风险180","个人风险60","个人风险120","个人风险180","财务风险60","财务风险120","财务风险180","经营风险60","经营风险120","经营风险180","行业风险60","行业风险120","行业风险180","企业风险60","企业风险120","企业风险180","sub120_60","sub180_120"]


model_labels = \
["企业风险->知识产权->著作权","行业风险->行业分析->黑天鹅","企业风险->管理风险->泄密","企业风险->基本信息变更->人员变动","企业风险->","企业风险->人力资源->招聘猛增",\
"企业风险->债券风险->主体评级下调","个人风险->高管风险","企业风险->用工->不签协议","行业风险->行业分析->趋势变化","企业风险->债券风险->取消债券发行","企业风险->经营风险->暂停上市","企业风险->风险提示->连续停牌","企业风险->风险提示->重大投资风险","企业风险->知识产权->专利信息","企业风险->重大事项->关联风险","企业风险->公司公告->股权结构","企业风险->司法诉讼->失信被执行人","企业风险->财务流动性->风险亏损","企业风险->知识产权->侵权","企业风险->经营风险->企业破产","企业风险->管理风险->管理层重大波动","企业风险->经",\
"企业风险->工商变更->人员变动","企业风险->重大事项->协议转让","企业风险->重大事项->资产重组","企业风险->司法诉讼->涉诉","行业风险->行业分析->行业动态","企业风险->财务风险->诈骗","企业风险->经营风险->僵尸企业","行业风险->行业分析->行业衰退环境恶化","企业风险->经营风险->破产保护","企业风险->经营风险->资产受限","企业风险->管理风险->管理人员失信经历","企业风险->行业风险趋势变化","企业风险->财务风险->被骗钱","企业风险->知识产权->专利失效","行业风险->产品把控->技术能力","企业风险->税务偷税漏税","企业风险->经营风险->坏账",\
"企业风险->风险提示->交易异常","企业风险->管理风险->管理混乱","企业风险->监管处罚->工商处罚","企业风险->债券风险->债券违约","企业风险->经营风险->融资失败",\
"企业风险->财务风险->财务造假","企业风险->财务风险->债务较多","企业风险->经营风险->企业欠税","企业风险->管理","企业风险->产品风险->假冒产品","企业风险->重大事项->增资扩股","企业风险->监管处罚->违法违规","行业风险->行业调整->市场面临考验","企业风险->经营风险->失信问题","企业风险->事故->交通事故","行业风险->价格变化价格泡沫","行业风险->行业调整->空间有限","企业风险->基本信息变更->换届选举","企业风险->消费者评价->待遇差","","企业风险->监管处罚->上交所询问","企业风险->财务风险->资金链危机","企业风险->重大事项->举牌",\
"企业风险->债券风险->债权变更","企业风险->经营风险->价格战","企业风险->司法诉讼->强制执行","企业风险->经营风险->企业盲目扩大","企业风险->重大事项->回购股份","企业风险->工商变更->经营变更","行业风险->行业调整->产能过剩","企业风险->重大事项->股权激励","企业风险->风险提示->复牌公告","个人风险->高管动向","企业","行业风险->价格变化->价格回落","企业风险->基本信息变更->减资公告","企业风险->重大事项->集中抛售","企业风险->经营风险->资产恶化","企业风险->风险提示->重大事项停牌","企业风险->经营风险->产品滞销","企业风险->用工->工资低",\
"企业风险->公司公告->融资融券","个人风险->核心人员离职","企业风险->人力资源->大批裁员","企业风险->事故->伤亡事故","企业风险->监管处罚->环保处罚","企业风险->经营风险->收入下降","企业风险->用工->员工伤残","企业风险->监管处罚->违法违规",\
"企业风险->重大事项->股票并购","企业风险->重大事项->决议公告","企业风险->风险提","企业风险->产品风险->不合格产品","企业风险->事故->爆炸事故","企业风险->人力资源->离职猛增","企业风险->经营事项->委托理财","企业风险->财务风险->IPO造假","企业风险->重大事项->借贷担保","行业风险->价格变化->市场份额下降","企业风险->财务风险->应收账款","企业风险->重大事项->资产冻结","企业风险->事故->火灾事故","企业风险->重大事项->股权收购","企业风险->司法诉讼失信被执行人","企业风险->财务风险->财产保全","企业风险->财务风险->投资风险",\
"行业风险->行业分析->趋势","企业风险->财务风险->资金损失","企业风险->经营风险->价格风险","企业风险->重大事项->注册资本减少","企业风险->经营风险经营不善","企业风险->重大事项关联风险","行业风险->价格变化销售低迷","企业风险->经营风险->亏损","企业风险->用工->劳资纠纷","企业风险->经营","企业风险->监管处罚->违法违规违约订单","企业风","企业风险->风险提示->停牌公告","企业风险->风险提示->业绩预告","企业风险->经营风险->违约债务","企业风险->管理风险->管理层重大波动","企业风险->重大事项->法人频繁变更","企业风险->监管处罚证监处罚","企业风险->债券风险主体评级下调","企业风险->债券风险->债券风险","企业风险->经营风险价格风险",\
"企业风险->风险提示->短暂停牌","企业风险->重大事项->增持股份","企业风险->产品风险->产品差","企业风险->经营风险->产品质量被曝光","行业风险->行业调整->行业改革","行业风险->产品把控->违法违规","企","企业风险->产品风险->技术落后","企业风险->财务风险->拖欠供应商","企业风险->司法诉讼->企业赔偿","企业风险->经营风险->经营不善","企业风险->经营风险失信问题","企业风险->监管处罚->保监会处罚","企业风险->用工->工作时间长","企业风险->重大事项->关联交易","企业风险->财务风险->资金出逃","企业风险->债券风险->列入评级观察名单","企业风险->工商变更->停业",\
"企业风险->重大事项->减持股份","企业风险->风险提示->资产重组","企业风险->重大事项->资产收购","企业风险->财务风险->负债较多","企业风险->事故->污染事故","企业风险->管理风险->管理人员变更","企业风险->经营事项->证监处罚","企业风险->财务风险->流动性问题","企业风险->监管处罚->违法违规合同违约","企业风险->监管处罚->证监会处罚","企业风险->债券风险->债券暴跌","企业风险->经营风险->业绩亏损","企业风险->债券风险->债券价格大跌","企业风险->重大事项->股价暴跌","企业风险->财务流动性风险负债较多","企业风险->经营风险->关联企业波动","企业风险->经营事项->提供担保",\
"企业风险->重大事项->收购","企业风险->重大事项->并购","企业风险->财务流动性风险违约债务","企业风险->公司公告->股票退市","行业风险->行业分析->灰犀牛","企业风险->经营风险->经营方针变化","企业风险->消费者评价->服务差","企业风险->工商变更->清算","企业风险->事故->沉没事故","企业风险->经营事项->股份质押","企业风险->公司公告->决议公告","企业风险->重大事项->控股股东变更","企业风险->风险提示重大投资风险","企业风险->经营风险->盈利下降","企业风险->事故->其他事故","企业风险->债券风险债券违约","企业风险->财务风险->IPO遇阻","企业风险->风险提示->临时停牌",\
"企业风险->用工->没有保险","企业风险->用工->拖欠工资","企业风险->管理风险->高管风险","企业风险->财务流动性风险->流动性问题","行业风险->行业调整->产能问题","企业风险->债券风险->债项评级调低","企业风险->重大事项->资产并购","企业风险->重大事项->股权变动","企业风险->风险提示->股票停牌","行业风险->行业调整->需求萎缩","企业风险->消费者评价->形象差","企业风险->管理风险->争权夺利","企业风险->股权结构->股权变动","企业风险->经营风险->破产重组","企业风险->公司公告->股票预案","企业风险->债券风险->列入评级观察名单","企业风险->基本信息变更->修订公司章程","企业风险->债券风险->推迟披露债券评级","企业风险->监管处罚->证监处罚","企业风险->监管处罚->上交所问询","企业风险->重大事项->","企业风险->风险提示->债券停牌","行业风险->产品把控->产品质量","行业风险->行业调整->行业调整","个人风险->高管持股变动"]



class Configstormshandong(object):
    ip = "192.168.172.130"
    db = "shandong"
    tb = "shandongtest"
    login = "root"
    pw = "oec!Server"
    port = 3306

class Configooo(object):
    ip = "192.168.172.130"
    db = "test"
    tb = "`239110`"
    login = "root"
    pw = "oec!Server"
    port = 3306



class Config_guangxi(object):
    #shandong_ip = "192.168.172.130"
    #shandong_database = "shandong"
    #shandong_label = "shandongtest"
    #storm_database = "test"
    #storm_table = "239110"
    #shandong_table = "shandongtest"
    ip = "117.107.241.65"
    db = "datacenter_baidumap"
    tb = "merchant_info_guangxi"
    login = "user_baidumap"
    pw = "lqh4kwHnfm5&iinnzGdl"
    port = 3306

class Config_map_shandong(object):

    ip = "117.107.241.65"
    db = "datacenter_baidumap"
    tb = "merchant_info_shandong"
    login = "user_baidumap"
    pw = "lqh4kwHnfm5&iinnzGdl"
    port = 3306

class Config_bond_risk(object):
    ip = "117.107.241.69"
    port = 3310
    db = "zhaiquanyujing"
    mid_tb = "middleTable"
    result_tb = "resultTable"
    tag_tb = "tag_info_t"
    login = "production"
    pw="V2aBPgBwb8EuPkSe"

class ISqlHelper(object):

  def init_db(self):
    raise NotImplemented

  def drop_db(self):
    raise NotImplemented

  def insert(self, value=None):
    raise NotImplemented

  def delete(self, conditions=None):
    raise NotImplemented

  def update(self, conditions=None, value=None):
    raise NotImplemented
  
  def select(self, count=None, conditions=None):
    raise NotImplemented

#------
class SqlHelper(ISqlHelper):
    def __init__(self, Config):
        self.connect = pymysql.connect(  
              host=Config.ip,
              port=Config.port,  
              user=Config.login,
              passwd=Config.pw,  #'lqh4kwHnfm5&iinnzGdl',  #"onfig.pw,
              db=Config.db,
              charset='utf8'  
        )
        self.cursor = self.connect.cursor()
        self.config = Config
        # init
        #conn = psycopg2.connect(database = Config.shandong_database, user=Config.login, password=Config.pw, host=Config.addr_ip, port=22)
        #cursor = conn.cursor()
        # exe sql
        #cursor.execute(sql, values)
        #print('mysql+mysqldb://%s:%s@%s/%s?charset=utf8"' % (Config.shandong_database, Config.login, Config.addr_ip, Config.shandong_table))
        #self.engine = create_engine('mysql+mysqldb://%s:%s@%s/%s?charset=utf8"' % (Config.shandong_database, Config.login, Config.addr_ip, Config.shandong_table))
        #DB_Session = sessionmaker(bind=self.engine)
        #self.session = DB_Session()
  
    def init_db(self):
        BaseModel.metadata.create_all(self.engine)

    def drop_db(self):
        BaseModel.metadata.drop_all(self.engine)

    def insert(self, value):
        proxy = Config(ip=value['ip'], port=value['port'], types=value['types'], protocol=value['protocol'],
        country=value['country'],
        area=value['area'], speed=value['speed'])
        self.session.add(proxy)
        self.session.commit()
    
    def delete(self, conditions=None):
        if conditions:
            conditon_list = []
        for key in list(conditions.keys()):
            if self.params.get(key, None):
                conditon_list.append(self.params.get(key) == conditions.get(key))
                conditions = conditon_list
                query = self.session.query(Config)
                for condition in conditions:
                    query = query.filter(condition)
                    deleteNum = query.delete()
                    self.session.commit()
            else:
                deleteNum = 0
        return ('deleteNum', deleteNum)
    
    def update(self, conditions=None, value=None):

        '''
        conditions的格式是个字典类似self.params
        :param conditions:
        :param value:也是个字典：{'ip':192.168.0.1}
        :return:
        '''
        if conditions and value:
            conditon_list = []
        for key in list(conditions.keys()):
            if self.params.get(key, None):
                conditon_list.append(self.params.get(key) == conditions.get(key))
                conditions = conditon_list
                query = self.session.query(Config)
        for condition in conditions:
            query = query.filter(condition)
            updatevalue = {}
        for key in list(value.keys()):
            if self.params.get(key, None):
                updatevalue[self.params.get(key, None)] = value.get(key)
                updateNum = query.update(updatevalue)
                self.session.commit()
            else:
                updateNum = 0
        return {'updateNum': updateNum}
        
    def selectdescription(self, table=None, count=None,conditions=None):
        sql = "SELECT * FROM %s LIMIT 1;" % table
        self.cursor.execute(sql)  
        return self.cursor.description

    def selectany(self, table=None, count=None, conditions=None):
        sql = "SELECT * FROM %s LIMIT %d;" % (table,count)
        self.cursor.execute(sql)  
        return self.cursor.fetchall()

        '''
        conditions的格式是个字典。类似self.params
        :param count:
        :param conditions:
        :return:
        '''
        '''
        if conditions:
            conditon_list = []
        for key in list(conditions.keys()):
            if self.params.get(key, None):
                conditon_list.append(self.params.get(key) == conditions.get(key))
                conditions = conditon_list
            else:
                conditions = []
        query = self.session.query(Config.ip, Config.port, Config.score)
        if len(conditions) > 0 and count:
            for condition in conditions:
                query = query.filter(condition)
            return query.order_by(Config.score.desc(), Config.speed).limit(count).all()
        elif count:
              return query.order_by(Config.score.desc(), Config.speed).limit(count).all()
        elif len(conditions) > 0:
            for condition in conditions:
                query = query.filter(condition)
            return query.order_by(Config.score.desc(), Config.speed).all()
        else:
            return query.order_by(Config.score.desc(), Config.speed).all()
        '''

    def select_where(self, cols, tb, kw):
        sql = "select %s from %s where keyword='%s';" % (cols, tb, kw)
        #print(sql)
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def select(self, cols, tb):
        sql = "select %s from %s;" % (cols, tb)
        #print(sql)
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def close(self):
       self.cursor.close()
       self.connect.close()

    def execute(self, sql):
        self.cursor.execute(sql)
        #self.connect.commit()
        return self.cursor.fetchall()

def days_shift_window(base, days):
    # datetime.datetime(2018, 3, 4, 16, 42, 4, 308669)
    if pd.isnull(base):
        base = datetime.datetime.now()
    days = datetime.timedelta(days=days)
    return base, base-days
    
def todayStr():
    return str(datetime.datetime.now()).split(" ")[0]

def todayShiftStr(num):
    b_ = datetime.datetime.now()
    d_ = datetime.timedelta(days=num)
    return str(b_-d_).split(" ")[0]

def filter_data_by_time(bond_risk_, day):
    # ex: bond_risk_.execute("select * from middleTable where (to_days(%s) - to_days(date_input) <=180 and to_days(now()) - to_days(date)<=180);"%day)
    # only match the middleTable; 
     data_180_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=180);"%day)
     data_150_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=150);"%day)
     data_120_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=120);"%day)
     data_90_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=90);"%day)
     data_60_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=60);"%day)
     data_30_ = bond_risk_.execute("select * from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(%s) - to_days(date)<=30);"%day)
     k_lst = [ "data_180_", "data_150_", "data_120_", "data_90_", "data_60_", "data_30_"]
     v_lst = [ data_180_, data_150_, data_120_, data_90_, data_60_, data_30_]
     return dict(zip(k_lst, v_lst))

def cnt_label_day(bond_risk_, comp, label, date, n):
    _cntLst = []
    sql_ = 'select cnt from middleTable where (to_days(now()) - to_days(date))<=%s  and label="%s" and compname="%s";'%(date+n,label,comp)
    try:
        cursor= bond_risk_.execute(sql_)
        #cursor = bond_risk_.find({"compname":comp, "label":label})
        for i in cursor:
            if datetime.datetime.now()-timeFormat(i['date'])<(date+n):
                _cntLst.append(i[cnt])
        return  _cntLst
    except:
        #print(sql_)
        traceback.print_exc()
        #print(len(label_cnt))
        return 0.0

def get_label_120(bond_risk_, comps, labels, n):
    dic_com = dict()
    breakcnt = 0
    for c in comps:
        #print(c)
        dic_label = dict()
        for l in labels:
            dic_date = dict()
            for d in [120,60,]:
                #_ = todayShiftStr(d)
                cnt_comp_date_label = cnt_label_day(bond_risk_,c,l,d,n)
                b_ = 0
                if cnt_label_day == 0:
                    b_ =  0
                elif len(cnt_comp_date_label)>0:
                    #print(c,l,d)
                    for i in cnt_comp_date_label:
                        b_+=i
                    #print(cnt_comp_date_label)
                    #print(b_)
                dic_date[d]=b_
            dic_label[l] = dic_date
        dic_com[c] = dic_label   
        breakcnt+=1
        #if breakcnt>10:
        #    break

    return dic_com

def cell_fill(_panel, i, c):
    #print("> cell_fill", i, c)
    data = _panel[i]
    #print(data)
    try:
        if c =="180":
            cnt = data.loc["180",:].sum()
            return cnt
        elif c =="120":
            cnt = data.loc["120",:].sum()
            return cnt
        elif c =="60":
            cnt = data.loc["60",:].sum()
            return cnt
        elif c == "all":
            return data.loc[:,:]
        else:
            cnt = data.loc[180,c]
            if cnt>0:
                pass
                #print(i,c)
                #print(cnt)
            return cnt
    except:
            pass
            #traceback.print_exc()
            return 0

import collections
class TimeCnt():
    def __init__(self):
        self.deque = collections.deque(maxlen=3)
        self.time_mark()
    def time_mark(self):
        _time = datetime.datetime.now()
        self.deque.append(_time)

    def cnt_time(self):
        self.time_mark()
        #print(">耗时: ", (self.deque[-1]-self.deque[-2]).microseconds)

def debug(func):  
    def new_func(a, b):  
        #print("> ",func)
        try:
            result = func(a, b)  
            #print("> ",func, result)
        except:
            traceback.print_exc()
    return new_func  

def time_cnt(func):  
    def new_func(a, b):  
        _TimeCnt = TimeCnt()
        #_TimeCnt.cnt_time()
        result = func(a, b)  
        _TimeCnt.cnt_time()
        #print "result:", result, "used:", (end_tiem - start_time).microseconds, "μs"  
        #return result
    return new_func  

@time_cnt
def prt(a):
    pass
    #print(a)

@debug
def prt(a):
    pass
    #print(a)

def load_model(model_file='/home/siyuan/data/xgb.model'):
    bst = xgb.Booster()
    bst.load_model(model_file)
    return bst

def predict(bst, test_X, test_Y):
    xg_test = xgb.DMatrix(test_X, label=test_Y)
    pred = bst.predict(xg_test)
    return pred

def save_result(pred):
    pass
    #print(pred)

def update_every_day(model_labels_2, model_labels, model_columns_base):
    pass

def group_cnt_key_word(kw,i,_panel):
    num = re.findall("\d+",kw)[0]
    keyword =  re.findall("[^\d]+",kw)[0]
    df =  _panel[i].loc[int(num),:]
    b = 0
    for c in df.keys():
        if keyword in c:
            b+=df[c]
    return b

def set_dummy(df, _is=True):
    if _is:
        df = df.fillna(0.0)
        df = df.astype(np.float64)
        return df
    else:
        df = df.applymap(lambda x : np.NaN if x==0.0 else x)
        df = df.astype(np.float64)
        return df

def get_label_time_window(bond_risk_, from_, to_):
    cursor = bond_risk_.find()
    _labels = []
    _compnames= []
    for i in cursor:
        #print(">>> i", i)
        dateinput = timeFormat(i['date_input'])
        fromnow180 = (datetime.datetime.now()-datetime.timedelta(from_))
        fromnow60 = (datetime.datetime.now()-datetime.timedelta(to_))
        if dateinput<fromnow180:
            if dateinput>fromnow60:
               _labels.append(i['label'])
               _compnames.append(i['compname'])
    #print("_labels", "_compnames")
    #print(_labels, _compnames)
    return _labels, _compnames

def pred_all(df):
    collections.Counter(predict(bst, df.iloc[:1200][(True-df.iloc[:1200]['120'].isin([np.NaN]))], df.iloc[:,1]))

#    try:
#        update_every_day(model_labels_2, model_labels, model_columns_base)
#    except:
#        traceback.print_exc()
#    prt("a","b")
def main(n):
    _tc = TimeCnt()
    _tc.cnt_time()
    #model_labels_2 = model_labels_2
    #model_labels = model_labels
    #model_columns_base = model_columns_base
    #print("> this is sql test")
    #shandong_ = SqlHelper(Config_map_shandong)
    #guangxi_ = SqlHelper(Config_guangxi)
    #storm_shandong_= SqlHelper(Configstormshandong)
    #storm_110_ = SqlHelper(Configooo)
    _tc.cnt_time()
    today = todayStr()

    #from load_mysqL_from_localcpk import load_mongodb_conn
    #bond_risk_ = load_mysqL_from_localcpk.load_mongodb_conn()
    mysql_bond_risk_ = SqlHelper(Config_bond_risk)
    #cursor = bond_risk_.find()
    #for i in cursor:
        #print(i)
        #pdb.set_trace()
    #    timeFormat(i['date_input'])
    label_120_ = mysql_bond_risk_.execute("select label from middleTable where (to_days(now()) - to_days(date)>=%d);"% n)
    compname_120_ = mysql_bond_risk_.execute("select compname from middleTable where (to_days(now()) - to_days(date)>=%d);"% n)
    #label_120_ = mysql_bond_risk_.execute("select label from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(now()) - to_days(date)<=%d);"% n)
    #_tc.cnt_time()
    #compname_120_ = mysql_bond_risk_.execute("select compname from middleTable where (to_days(now()) - to_days(date_input) <=720 and to_days(now()) - to_days(date)<=%d);"% n)
    #label_120_, compname_120_ = get_label_time_window(bond_risk_, n, n-120)
    set_ = set()
    [set_.add(i) for i in label_120_]
    label_lst_ = list(set_)
    _tc.cnt_time()
    set_.clear()
    [set_.add(i) for i in compname_120_]
    compname_lst_ = list(set_)
    _dic = get_label_120(mysql_bond_risk_, compname_lst_, label_lst_, n)
    _tc.cnt_time()
    _panel = pd.Panel(_dic)
    _panel = _panel.fillna(0.0)
    #filter_data_by_time(bond_risk_, today)
    _tc.cnt_time()

    df_4_model = pd.DataFrame(index=compname_lst_, columns = model_labels + model_labels_2)
    df_4_model = df_4_model.fillna(0.0)
    df_4_model = df_4_model.astype(np.float64)
    _index = list(df_4_model.index)
    _columns = model_labels
    _columns_2 = model_labels_2
    #print("> ready to get data")
    _cnt = 0 
    for i in _index:
        _cnt+=1
        if _cnt % 100 ==1:
            pass
            #print(">>>> !!! handle the,", i, _cnt)
        #if _cnt > 300:
            #break
            #print("> handle the,", i, _cnt)
        for c in _columns:
            df_4_model.loc[i,c] = cell_fill(_panel, i,c)
        df_4_model.loc[i, "企业名称"] = i
        df_4_model.loc[i, "发布日期"] = datetime.datetime.now()
        df_4_model.loc[i, "credit_recent"] = 0
        df_4_model.loc[i, "credit_ago"] = 0 
        df_4_model.loc[i, "credit_trend"] = 0

        df_4_model.loc[i, "60"] = _panel[i].loc[60,:].sum()
        df_4_model.loc[i, "120"] = _panel[i].loc[120,:].sum()
        df_4_model.loc[i, "180"] = _panel[i].loc[180,:].sum()
        df_4_model.loc[i, "债券风险60"] = group_cnt_key_word("债券风险60",i,_panel)
        df_4_model.loc[i, "债券风险120"] = group_cnt_key_word("债券风险120",i,_panel)
        df_4_model.loc[i, "债券风险180"] = group_cnt_key_word("债券风险180",i,_panel)
        df_4_model.loc[i, "个人风险60"] = group_cnt_key_word("个人风险60",i,_panel)
        df_4_model.loc[i, "个人风险120"] = group_cnt_key_word("个人风险120",i,_panel)
        df_4_model.loc[i, "个人风险180"] = group_cnt_key_word("个人风险180",i,_panel)
        df_4_model.loc[i, "财务风险60"] = group_cnt_key_word("财务风险60",i,_panel)
        df_4_model.loc[i, "财务风险120"] = group_cnt_key_word("财务风险120",i,_panel)
        df_4_model.loc[i, "财务风险180"] = group_cnt_key_word("财务风险180",i,_panel)
        df_4_model.loc[i, "经营风险60"] = group_cnt_key_word("经营风险60",i,_panel)
        df_4_model.loc[i, "经营风险120"] = group_cnt_key_word("经营风险120",i,_panel)
        df_4_model.loc[i, "经营风险180"] = group_cnt_key_word("经营风险180",i,_panel)
        df_4_model.loc[i, "行业风险60"] = group_cnt_key_word("行业风险60",i,_panel)
        df_4_model.loc[i, "行业风险120"] = group_cnt_key_word("行业风险120",i,_panel)
        df_4_model.loc[i, "行业风险180"] = group_cnt_key_word("行业风险180",i,_panel)
        df_4_model.loc[i, "企业风险60"] = group_cnt_key_word("企业风险60",i,_panel)
        df_4_model.loc[i, "企业风险120"] = group_cnt_key_word("企业风险120",i,_panel)
        df_4_model.loc[i, "企业风险180"] = group_cnt_key_word("企业风险180",i,_panel)
        #df_4_model = df_4_model.applymap(lambda x : np.NaN if x==0 else x)
        df_4_model.loc[i, "sub120_60"] = df_4_model.loc[i, "120"] - df_4_model.loc[i, "60"]
        df_4_model.loc[i, "sub180_120"] = df_4_model.loc[i, "180"] - df_4_model.loc[i, "120"]
        #df_4_model = df_4_model.applymap(lambda x : np.NaN if x==-1 else x)
        #df_4_model = df_4_model.applymap(lambda x : np.NaN if x==0 else x)

    _x = df_4_model.drop(["企业名称","发布日期","Label"],1)
    _z = pd.read_csv("/home/siyuan/bond_risk/_z.csv").drop(["Unnamed: 0","发布日期","Label"],1)
    #_z.index = _z["企业名称"]
    _z = _z.drop("企业名称", axis=1)
    _x.columns = list(_z.columns)
    # !! filter
    #_x = _x[(_x["sub120_60"]>0) & (_x["60"]>0)]
    #_x = _x[(_x["60"]>0)]
    _x = _x[(_x["120"]>0)]
    train_separator = len(_x.index)
    #print(train_separator)
    _pred_data = pd.concat([_x, _z], axis=0)

    _pred_data = set_dummy(_pred_data, False)
    # output predict label
    bst = xgb.Booster()
    bst.load_model("/home/siyuan/data/xgb.model")
    #pdb.set_trace()

    #_lz = pd.read_csv("/home/siyuan/bond_risk/_z.csv")["Label"]
    result_ = predict(bst, _pred_data, _pred_data.iloc[1])
    dict_ = dict(zip(list(_pred_data.index), result_))
    dict_res = dict(zip(list(_pred_data.index)[:train_separator], result_[:train_separator]))
    #dict_res = dict(zip(list(_pred_data.index), result_))

    #print(collections.Counter(list(result_)))
    #print(collections.Counter(list(result_)[:train_separator]))
    cnt = 0
    #pdb.set_trace()
    #print(dict_res)
    for i in dict_res.keys():
        #sql_ = "INSERT INTO resultTable VALUES('', '%s', CURTIME(), '%s');"%(i,str(format(dict_res[i],'.9e')))
        sql_ = "INSERT INTO resultTable VALUES('', '%s', CURTIME(), '%s');"%(i,str(format(dict_res[i],'.9e')))
        #print(sql_)
        sql_res_ = mysql_bond_risk_.execute(sql_)
        #print(sql_res_)
        cnt+=1
    pdb.set_trace()
    mysql_bond_risk_.connect.commit()
    #import read_result_sql
    #read_result_sql.load_resultTable()

if __name__ == "__main__":
    pass
    """
    for i in range(0, 30)[::-1]:
        main(i)
        import read_result_sql
        read_result_sql.load_resultTable()
        break
    """

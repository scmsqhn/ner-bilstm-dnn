#coding=utf8
# !/usr/bin/env python
import pymongo
import pymysql
import os,redis
import pandas as pd
import codecs,chardet,re
import traceback
import pdb

#DEBUG = False
DEBUG = True



cq={}
cq['ip']="113.204.229.74"
cq['db']="taiyuan"
#cq['tb']="gz_gongan_alarm_1617"
cq['tb']="original_data"
cq['login']="root"
cq['pw']="123456"
cq['port']=15004

sd={} # shandong storm
sd['ip']="192.168.172.130"
sd['db']="shandong"
sd['tb']="shandongtest"
sd['login']="root"
sd['pw']="oec!Server"
sd['port']=3306

tt={} # test
tt['ip']="192.168.172.130"
tt['db']="test"
tt['tb']="239110"
tt['login']="root"
tt['pw']="oec!Server"
tt['port']=3306

gx={} # guangxi
gx['ip']="117.107.241.65"
gx['db']="datacenter_baidumap"
gx['tb']="merchant_info_guangxi"
gx['login']="user_baidumap"
gx['pw']="lqh4kwHnfm5&iinnzGdl"
gx['port']=3306

sdm={} # shandong storm
sdm['ip']="117.107.241.65"
sdm['db']="datacenter_baidumap"
sdm['tb']="merchant_info_shandong"
sdm['login']="user_baidumap"
sdm['pw']="lqh4kwHnfm5&iinnzGdl"
sdm['port']=3306

rs={}#risk
rs['ip'] = "117.107.241.69"
rs['port'] = 3310
rs['db'] = "zhaiquanyujing"
rs['mid_tb'] = "middleTable"
rs['result_tb'] = "resultTable"
rs['tag_tb'] = "tag_info_t"
rs['login'] = "production"
rs['pw']="V2aBPgBwb8EuPkSe"

db_dict = {}
db_dict['sd']=sd
db_dict['rs']=rs
db_dict['sdm']=sdm
db_dict['gx']=gx
db_dict['tt']=tt
db_dict['cq']=cq

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

class SqlHelper(ISqlHelper):
    def __init__(self, Config):
        self.connect = pymysql.connect(
              host=Config['ip'],
              port=Config['port'],
              user=Config['login'],
              passwd=Config['pw'],
              db=Config['db'],
              charset='utf8',
        )
        self.cursor = self.connect.cursor()
        self.config = Config

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
class MongoConn(ISqlHelper):
    remote_db_ip = "127.0.0.1"
    #remote_db_ip = "10.6.5.32"
    def __init__(self, cmd):
        if cmd == "remote":
            self.conn = self.get_remote_conn(MongoConn.remote_db_ip)
            self.db = self.getdb(self.conn)
        else:
            raise Exception("Mongo db interface init error !")

    def get_remote_conn(self,ip_str):
        conn=pymongo.MongoClient(ip_str,27017)
        return conn

    def getlocalconn(self):
        conn=pymongo.MongoClient('localhost',27017)
        return conn

    def getdb(self, _conn=""):
        _conn = self.conn
        self.db = _conn.myDB
        return self.db

    #def getdb(self):
    #    self.db = self.conn.myDB
    #    return self.db

    def get_db_with_name(self, _conn, _name):
        db = _conn[_name]
        return db

    def getcoll(self, _db):
        my_collection=_db.myCollection
        return my_collection

    def get_coll_with_name(self, _name):
        my_collection=self.db[_name]
        return my_collection

    def test(self, my_collection):
        tom={'name':'Tom','age':18,'sex':'男','hobbies':['吃饭','睡觉','打豆豆']}
        alice={'name':'Alice','age':19,'sex':'女','hobbies':['读书','跑步','弹吉他']}
        tom_id=my_collection.insert(tom)
        alice_id=my_collection.insert(alice)
        print(tom_id)
        print(alice_id)

    def query(self, my_collection):
        cursor=my_collection.find()
        print(cursor.count())   # 获取文档个数
        for item in cursor:
            print(item)
        my_collection.update({'name':'Tom'},{'$set':{'hobbies':['向Alice学习读书','跟Alice一起跑步','向Alice学习弹吉他']}})
        for item in my_collection.find():
                print(item)

    def delete_coll(self, my_collection):
        my_collection.remove()
        return 0

    def async_from_mysql_remote(self, collectionName, mysql_conn):
        coll = self.get_coll_with_name(collectionName)
        #coll.create_index(['compnames','label','cnt','date','date_input'], ordered=False,unique=True,dropDups=True)
        print(collectionName)
        mysql_conn.cursor.execute("show full columns from %s;"%collectionName)
        cols = [i[0] for i in mysql_conn.cursor]
        mysql_conn.cursor.execute("select * from %s;"%collectionName)
        cnt = 0
        for i in mysql_conn.cursor:
            i = [str(j) for j in i]
            cnt+=1
            if DEBUG:
                print(list(i))
            d = dict(zip(cols,list(i)))
            coll.insert(d)
            coll.find_and_modify(query=d, update={"$set": d}, upsert=True, full_response= True)
            if cnt%10000==1:
                print('coll insert ', str(d))

class redis_helper(ISqlHelper):
  def __init__(self):
    super(redis_helper, self).__init__()
    self.r = redis.Redis(host='10.6.4.67', port=6379)
    self.mass_set_cmd = "mass_set.txt"
    print("\n> connect to redis")
    self.print_redis_status()
    pass

  def print_redis_status(self):
    print('\n> dbsize:%s' % self.r.dbsize())
    print('\n> ping %s' % self.r.ping())

  def load_txt(self,filepath):
    print("\n> load_txt, ",filepath)
    with open(filepath, 'r') as f:
      return f.read()

  def load_txt_rb_lst(self, filepath):
    outlines =  []
    f = open(filepath, 'rb')
    _cont = f.read()
    lines = _cont.split(b"\r\n")
    _coding = chardet.detect(lines[:10])['encoding']
    for line in lines:
        try:
          outlines.append(line.decode(_coding))
        except:
          if DEBUG:
            print(line, chardet.detect(line))
            traceback.print_exc()
          pass
    f.close()
    return outlines

  def cut(self, line):
    words = jieba.cut(line, HMM=True)
    return words

  def redis_cmd(self, _lst):
    #assert len(_lst) ==3
    #print(_lst)
    #ex = "*3\r\n$3\r\nSET\r\n$5\r\nmykey\r\n$7\r\nmyvalue\r\n"
    cmd = "*%s\r\n$%s\r\n%s\r\n$%s\r\n%s\r\n$%s\r\n%s\r\n\r\n" %     (len(_lst), len(_lst[0]), _lst[0], len(_lst[1]), _lst[1], len(_lst[2]), _lst[2])
    #print(cmd)
    return cmd

  def mass_set(self, keys, values, mass_cmd_file):
    assert len(keys) == len(values)
    with open(mass_cmd_file, 'w+') as f:
      for cnt in range(len(keys)):
        f.write(r.redis_cmd(['SET', str(keys[cnt]), str(values[cnt])]))
    return 0

  def run_mass_set(self, set_txt = 'mass_set.txt'):
    return os.popen('sudo cat %s | redis-cli --pipe' % os.path.join(CURDIR, set_txt))

  def sav_df_2_redis(self,redisConn, key, df):
    try:
      redisConn.set(key, df.to_msgpack(compress='zlib'))
      return 0
    except:
      if DEBUG:
        traceback.print_exc()
      return -1

  def read_df_from_redis(self, redisConn, key):
    try:
      return pd.read_msgpack(redisConn.get(key))
    except:
      if DEBUG:
        traceback.print_exc()
      return -1

  def save_txt_2_redis(filepath):
    cont = r.load_txt(filepath)
    cont = re.sub(r" ",r"",cont)
    r.mass_set([i for i in range(len(cont))], cont, 'mass_set.txt')
    r.run_mass_set(set_txt = 'mass_set.txt')

  def csv_2_df_2_redis(self, filepath):
    df = pd.read_csv(filepath)
    key = t.split(r".")[0]
    self.sav_df_2_redis(self.r, key, df)
    return key

  def redis_2_df_2_csv(self, key):
    df = self.read_df_from_redis(self.r, key)
    df.to_csv(key + r".csv")
    return df

  def cut_file(self, filepath):
    pass

def redis_main():
  pass
  r = redis_helper()
  l = []
  ch = Data_Helper()
  files = r.get_py('/home/ubuntu/extcode', l, 'TXT')
  f = open("ner_sample_base_v10.txt", "a+")
  for _file in files:
      cont = r.load_txt(_file)
      cont = ch.pass_char_num(cont)
      cont = ch.classifier_char(cont)
      lines = cont.split("\n")
      for line in lines:
        words = list(jieba.cut(line, HMM=True))
        [f.write(str(word+" ")) for word in words]
        f.write("\r\n")
  print("\n> handle file", _file)
  f.close()

if __name__ == "__main__":
    pass
    mg = MongoConn("remote")
    _d = []
    _d.append(db_dict['rs'])
    #_d.append(db_dict['cq'])
    #_d.append(db_dict['gx'])
    #_d.append(db_dict['sd'])
    #_d.append(db_dict['sdm'])
    #_d.append(db_dict['tt'])
    for i in _d:
        print("database is ", i)
        try:
            mysql_helper_instance = SqlHelper(i)
            mysql_helper_instance.cursor.execute("show tables")
            tbs = []
            [tbs.append(i[0]) for i in mysql_helper_instance.cursor]
            print("tbs is ", tbs)
            if DEBUG:
                print(tbs)

            #tbs = ['gz_gongan_case','gz_gongan_alarm_1617','original_data']
            for tb in list(tbs):
                if DEBUG:
                    print(tb)
                try:
                  mg.async_from_mysql_remote(tb, mysql_helper_instance)
                except:
                    traceback.print_exc()

        except:
            if DEBUG:
                traceback.print_exc()
            pass


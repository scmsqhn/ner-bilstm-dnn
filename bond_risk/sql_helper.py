# sql handler for all mysql of my 
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
import pymysql

import pymysql  

class Configstormshandong(object):
    ip = "192.168.172.130"
    db = "shandong"
    tb = "shandongtest"
    login = "root"
    pw = "oec/!Server"

class Configtest(object):
    ip = "192.168.172.130"
    db = "test"
    tb = "239110"
    login = "root"
    pw = "oec/!Server"


class Configliuzhou(object):
    ip = "192.168.172.130"
    db = "dataaccess"
    tb = "liuzhou_b_receive"
    login = "root"
    pw = "oec/!Server"

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

class Config_map_shandong(object):

    ip = "117.107.241.65"
    db = "datacenter_baidumap"
    tb = "merchant_info_shandong"
    login = "user_baidumap"
    pw = "lqh4kwHnfm5&iinnzGdl"

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
        self.connect = pymysql.Connect(  
              host=Config.ip,
              port=3306,  
              user=Config.login,
              passwd=Config.pw,  #'lqh4kwHnfm5&iinnzGdl',  #"onfig.pw,
              db=Config.db,
              charset='utf8'  
        )
        self.cursor = self.connect.cursor()
        self.table = Config.tb
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
        conditions的格式是个字典。类似self.params
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
        print(sql)
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def select(self, cols, tb):
        sql = "select %s from %s;" % (cols, tb)
        print(sql)
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def execute(self, _sql):
        print(_sql)
        self.cursor.execute(_sql)
        self.connect.commit()
        return self.cursor.fetchall()

    def close(self):
       self.cursor.close()
       self.connect.close()
    
if __name__ == "__main__":
    print("> this is sql test")
    shandong_ = SqlHelper(Config_map_shandong)
    guangxi_ = SqlHelper(Config_guangxi)
    storm_shandong_= SqlHelper(Configstormshandong)
    storm_test_ = SqlHelper(Configtest)
    liuzhou_ = SqlHelper(Configliuzhou)




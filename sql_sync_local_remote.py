#coding=utf-8

#import MySQLdb
import pymysql as MySQLdb
import pickle as pk
import traceback

def write2file(DB_data,save_filename):
    with open(save_filename, 'wb') as f:
        pk.dump(DB_data, f)

      # 创建数据库连接
conn2db = MySQLdb.connect(
        host='117.107.241.69',
        port = 3310,
        user='production',
        passwd = "V2aBPgBwb8EuPkSe",
        db = "zhaiquanyujing",
        charset = 'utf8',
        )

cur = conn2db.cursor() # 操作游标
DB_data = cur.execute("select * from middleTable;") # SQL语句 ，查询需要到处内容
DB_datas = cur.fetchmany(DB_data)
write2file(DB_datas,'./data/risk_data_sync')

cur.close()
conn2db.commit()
try:
    conn2db.close() # 关闭连接
    print("closed connection...")
except Exception:
    traceback.print_exc()

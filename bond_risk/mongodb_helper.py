# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pymongo
import pymysql

class MongoConn(object):

    def __init__(self):
        pass
        self.db = ""

    def getconn(self):
        conn=pymongo.MongoClient('localhost',27017)
        return conn

    def getdb(self, _conn):
        self.db = _conn.myDB
        return self.db

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

    def async_from_mysql_remote(self, collectionName):
        coll = self.get_coll_with_name(collectionName)
        pass


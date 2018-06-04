import pickle
import traceback
import logging
from mongodb_helper import MongoConn

def logger_init():
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")
    return logger

mylogger = logger_init()

def load_mongodb_conn(name="bondrisk"):
    mylogger.info("load_mongodb_conn()")
    mongoConn = MongoConn()
    conn = mongoConn.getconn()
    mongoConn.getdb(conn)
    coll =mongoConn.get_coll_with_name(name)
    return coll

def struc_data_2_dict(new_man, keys=['id', 'compname', 'date','label','cnt', 'date_input']):
    return dict(zip(keys,new_man))

def load_data_from_pickle_2_mongodb(_collection):
    mylogger.info("pload_data_from_pickle_2_mongodb")
    try:
        with open('./data/risk_data_sync', 'rb') as base_record_file:
            records=pickle.load(base_record_file)
            print(records)
        for i in records:
            dict_new_man = struc_data_2_dict(i)
            print(dict_new_man)
            _collection.insert(dict_new_man)
        return 0
    except:
        traceback.print_exc()
        return -1

def judge_result(res, name=""):
    if res == 0:
        mylogger.info('%s complete'%str(name))
    if res == -1:
        mylogger.info('%s error'%str(name))

def pullpush():
    mylogger.info("pullpush")
    # pull the data from pickle and into the mongodb
    coll = load_mongodb_conn()
    result = load_data_from_pickle_2_mongodb(coll)
    judge_result(result, "load_data_from_pick_2_mongodb")

def findall():
    cnt=10
    coll = load_mongodb_conn()
    for item in coll.find():
        print(item)
        if cnt<0:
            cnt-=1
            break


def main(name):
    _dic = {}
    _dic["pullpush"] = pullpush
    _dic["findall"] = findall 

    for i in name:
        print(i)
        _dic[i]()
    pass

if __name__ == "__main__":
    job_lst =  []
    #job_lst.append("pullpush")
    job_lst.append("findall")
    main(job_lst)

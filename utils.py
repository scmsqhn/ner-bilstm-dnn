import logging, sys, argparse
import datetime,re
from  dutil.substrfind import del_sub_str
from dutil.utility import get_config

def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#公安盗窃系统识别
def get_entity(tag_seq, char_seq):
    LOC = pro_address(comb_entity(get_LOC_entity(tag_seq, char_seq)),char_seq)
    return LOC

def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    b_loc = 0  # 开始位置
    e_loc = 0  # 结束位置
    try:
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-LOC':
                if 'loc' in locals().keys():
                    e_loc = i
                    LOC.append(loc + "|" + str(b_loc) + "," + str(e_loc))
                    del loc
                loc = char
                if i+1 == length:
                    e_loc = i
                    LOC.append(loc + "|" + str(b_loc) + "," + str(e_loc))
                b_loc = i
            if tag == 'I-LOC':
                if not 'loc' in locals().keys():
                    loc = char
                else:
                    loc += char
                if i + 1 == length:
                    e_loc = i
                    LOC.append(loc + "|" + str(b_loc) + "," + str(e_loc))
                    # print(DATE)
            if tag not in ['I-LOC', 'B-LOC']:
                if 'loc' in locals().keys():
                    e_loc = i
                    LOC.append(loc + "|" + str(b_loc) + "," + str(e_loc))
                    # print(DATE)
                    del loc
                continue
    except Exception as ex:
        print(ex.message)
    return LOC

#根据位置处理是否为连续的信息，合并数据
def comb_entity(entity):
    ENTITY = []
    ent = ""
    if entity:
        if len(entity) ==1:
            ENTITY.append(entity[0].split("|")[0])
        for i in range(1,len(entity)):
            pre_item = entity[i-1]
            post_item = entity[i]
            pre_en = pre_item.split("|")[0]
            pre_num = pre_item.split("|")[1]
            post_en = post_item.split("|")[0]
            post_num = post_item.split("|")[1]
            if ent == "":
                ent = pre_en
            if post_num.split(",")[0] == pre_num.split(",")[1]:
                ent = ent + post_en
            else:
                ENTITY.append(ent)
                ent = post_en
            if i == len(entity)-1 and ent:
                    ENTITY.append(ent)
    return ENTITY

#地址处理:
def pro_address(entity,char_seq):
    # sentence = "".join(char_seq)
    # sentences = [sent for sent in sentence.split("，") if re.findall("\d+", sent)]
    conf = get_config()
    add_extr_words = conf.get("address_words","add_extr_words").split(",")
    stop_words = conf.get("address_words", "stop_words").split(",")
    stop_words = [item.strip() for item in stop_words]
    not_startwith_word = conf.get("address_words", "not_startwith_word").split(",")
    not_startwith_word = [item.strip() for item in not_startwith_word]
    not_end_with_word = conf.get("address_words", "not_end_with_word").split(",")
    not_end_with_word = [item.strip() for item in not_end_with_word]
    not_include_words = conf.get("address_words", "not_include_words").split(",")
    not_include_words = [item.strip() for item in not_include_words]
    startwith_word = conf.get("address_words", "startwith_word").split(",")
    startwith_word = [item.strip() for item in startwith_word]
    end_with_words = conf.get("address_words", "end_with_words").split(",")
    end_with_words = [item.strip() for item in end_with_words]
    split_words = conf.get("address_words", "split_words").split(",")
    split_words = [item.strip() for item in split_words]
    # p_car_num = conf.get("address_words", "p_car_num")
    # p_id = conf.get("address_words", "p_id")
    # p_wlan = conf.get("address_words", "p_wlan")
    # p_date = conf.get("address_words", "p_date")
    # p_phone_num = conf.get("address_words", "p_phone_num")
    ENTITY = []
    for ent in entity:
        ent = ent.strip()
        if not ent.isdigit() and len(ent) > 2:
            if [word for word in add_extr_words if word in ent]:
                ENTITY.append(ent)
            elif len(ent)>6:
                ENTITY.append(ent)
    ENTITY = list(set(ENTITY))

    mid_ent  = ENTITY.copy()
    for item in ENTITY:
        if [word for word in not_include_words if word in item and not \
              [word for word in ["路", "道", "区", "市", "村", "乡", "镇", "县"] if word in item]
            ]:
            mid_ent.remove(item)
        elif [word for word in split_words if len(item.split(word))>1]:
            word = [word for word in split_words if len(item.split(word))>1][0]
            _item = item.split(word)[0]
            if len(item)>=5:
                    mid_ent.append(_item)
            mid_ent.remove(item)
        elif  [word for word in end_with_words if item.endswith(word) and len(item)>5]:
            mid_ent.remove(item)
            size = len([word for word in end_with_words][0])
            mid_ent.append(item[:-size])
        elif  [word for word in end_with_words if item.endswith(word) and len(item)<5]:
            mid_ent.remove(item)
        elif  [word for word in startwith_word if item.startswith(word) and len(item)>5]:
            mid_ent.remove(item)
            mid_ent.append(item[1:])
        elif [word for word in startwith_word if item.startswith(word) and len(item) <=5]:
            mid_ent.remove(item)
        elif re.findall(r"\d+）",item):
            mid_ent.remove(item)
            mid_ent.append(item.split("）")[1])
        elif re.findall(r"\d+\)",item):
            mid_ent.remove(item)
            mid_ent.append(item.split(")")[1])
        elif re.findall(r"\)",item):
            mid_ent.remove(item)
            mid_ent.append(item.split(")")[0])
        elif re.findall(r"）", item):
            mid_ent.remove(item)
            mid_ent.append(item.split("）")[0])
        elif item.endswith("（"):
            mid_ent.remove(item)
            mid_ent.append(item[:-1])
        elif item.endswith("手机"):
            mid_ent.remove(item)
            if [word for word in ["路","道","区","市","村","乡","镇","县"] if word in item]:
                mid_ent.append(item[:-2])
        elif item.startswith("从"):
            if len(item)<7:
                mid_ent.remove(item)
        elif '女士' in item or "老奶奶" in item or "先生" in item:
            mid_ent.remove(item)
        elif item[0] in [1,2,3,4,5,6,7,8,9,0,"（"] and len(item) <7:
            mid_ent.remove(item)
        elif "   " in item:
            mid_ent.remove(item)
            mid_ent.append(item.split("   ")[1])
        elif  (item in stop_words)  or [word for word in not_startwith_word if item.startswith(word) and "人民日报" not in item \
                and "和平里" not in item] or  [word for word in not_end_with_word if item.endswith(word)]:
                mid_ent.remove(item)
    ENTITY = list(set(mid_ent))
    ENTITY = del_sub_str(ENTITY)
    ENTITY.sort(key=lambda x:len(x),reverse=True)
    ENTITY = [item for item in ENTITY if len(item)>2  if not re.findall("^[-./0-9a-zA-Z]+$",item)]
    final_entity = []
    for ent in ENTITY:
        if [word for word in add_extr_words if word in ent]:
            final_entity.append(ent)
    ENTITY = final_entity
    if len(ENTITY) > 1:
        ENTITY = ENTITY[:2]
    return ENTITY
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger

if __name__ == '__main__':

    entity = ["建外SOHO14号楼西区E"]
    # print(pro_address(entity, []))
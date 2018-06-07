#encoding=utf8
# to generate the fomular data 
import data_helper
from data_helper import *
import re
import os
import sys 
import jieba 


tags = ['z','b','i','e','s','p','h','n','u','v','x','d','t','f','Q','q','k','c','r','R']
# begin in end b i e
# single s
# phone p h n
# weixin u v x 
# identifier d t f
import test_request

def load_data_from_api():
    pass
    test_request.post()

def replaceNC(line):
    line = re.sub("[2-9]","3",line)
    line = re.sub("[A-P_a-p]","c",line)
    line = re.sub("[r-z_R-Z]","c",line)
    line = re.sub("(qq|QQ)","QQ",line)
    return line

def load_all_txt():
    cont = ""
    g = open("/home/siyuan/data/alldata.txt","a+")
    f = open("/home/siyuan/data/beijing110_cp.txt", "r") 
    cont = f.read()
    cont = data_helper.full2half(cont)
    g.write(cont)
    f.close()
    print("mark")

    f = open("/home/siyuan/data/guang_xi_.txt", "r") 
    cont = f.read()
    cont = data_helper.full2half(cont)
    g.write(cont)
    f.close()
    print("mark")

    f = open("/home/siyuan/data/liu_zhou_.txt", "r") 
    cont = f.read()
    cont = data_helper.full2half(cont)
    g.write(cont)
    f.close()
    print("mark")

    #f = open("/home/siyuan/data/shandong.txt", "r") 
    #cont = f.read()
    #cont = data_helper.full2half(cont)
    #g.write(cont)
    #f.close()
    g.close()
    print("mark")

def isneed(line):
    if len(re.findall("[a-zA-Z]+",line)) > 0:
        if len(re.findall("[\u4e00-\u9fa5]+",line)) > 0:
            return True
    if len(re.findall("\d{5,}",line))>0:
        if len(re.findall("[\u4e00-\u9fa5]+",line)) > 0:
            return True
    return False        

import sys
sys.path.append("/home/siyuan/extcode_bak/extcode")
import extcode
from extcode import digital_info_extract as digiext

def filter_mark_txt():
    f = open("/home/siyuan/data/alldata.txt", "r")
    cnt = 0
    while(1):
        cnt+=1
        if cnt%1000==1:
            print(cnt)
        cont= f.readline()
        if cont == "":
            break
        print(cont)
        if not isneed(cont):
            continue
        print(cont)
        result_ = digiext.extract_digital([cont])[0]
        print(type(result_))
        print(result_)
        words = jieba.cut(cont, HMM=True)
        g = open("cut_char.txt", "a+")
        for word in words:
            mark_word(word, g, result_)
        print(cont)
    return result_

def mark_word(word, f, result_):
    _l = len(word)
    if _l==1:
        if word == "\n":
            f.write("\n ")
        elif word == "\r":
            f.write("\n ")
        elif word in "[\u4e00-\u9fa5]":
            f.write("%s/s "%word)
        elif word in "[a-zA-Z0-9]":
            f.write("%s/s "%word)
    elif _l==2:
        if word in "".join(result_['wx']):
            f.write("%s/u "%word[0])
            f.write("%s/x "%word[1])
        elif word in result_['phoneNum']:
            f.write("%s/p "%word[0])
            f.write("%s/n "%word[1])
        elif word in result_['identifier']:
            f.write("%s/d "%word[0])
            f.write("%s/f "%word[1])
        elif word in result_['qq']:
            f.write("%s/Q "%word[0])
            f.write("%s/q "%word[1])
        elif word in result_['creditCard']:
            f.write("%s/c "%word[0])
            f.write("%s/r "%word[1])
        else:
            f.write("%s/b "%word[0])
            f.write("%s/e "%word[1])
    else:
        if word in "".join(result_['wx']):
            word = replaceNC(word)
            ct = 0
            f.write("%s/u "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/v "%word[ct])
                    continue
                break
            f.write("%s/x "%word[ct])
        elif word in result_['phoneNum']:
            word = replaceNC(word)
            ct = 0
            f.write("%s/p "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/h "%word[ct])
                    continue
                break
            f.write("%s/n "%word[ct])
        elif word in result_['identifier']:
            word = replaceNC(word)
            ct = 0
            f.write("%s/d "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/t "%word[ct])
                    continue
                break
            f.write("%s/f "%word[ct])
        elif word in result_['creditCard']:
            word = replaceNC(word)
            ct = 0
            f.write("%s/c "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/r "%word[ct])
                    continue
                break
            f.write("%s/R "%word[ct])
        elif word in result_['qq']:
            word = replaceNC(word)
            ct = 0
            f.write("%s/Q "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/k "%word[ct])
                    continue
                break
            f.write("%s/q "%word[ct])
        else:
            word = replaceNC(word)
            ct = 0
            f.write("%s/b "%word[ct])
            while(1):
                ct+=1
                if _l>ct+1:
                    f.write("%s/i "%word[ct])
                    continue
                break
            f.write("%s/e "%word[ct])



def filter_marker():
    with open("cut_char.txt","r") as f:
        with open("cut_char_without_marker.txt","a+") as g:
            cnt = 0
            while(1):
                cnt+=1
                line = f.readline()
                line = re.sub("[2-9]/(.) ",r"3/\1 ",line)
                line = re.sub("[A-Pa-p]/(.) ",r"c/\1 ",line)
                line = re.sub("[R-Zr-z]/(.) ",r"c/\1 ",line)
                line = re.sub("[^\u4e00-\u9fa50-9a-zA-Z_/-]/(.)","",line)
                #line = re.sub("[^\u4e00-\u9fa54e0-9a-zA-Z]/. ","",line)
                #_line = re.sub("^ *", "", line)
                g.write(line)
                if cnt%10000==1:
                    print(cnt)
                if line == "":
                    break

# ================================================================================
#
# block 1 start make data and save into file which the data like thie "我/b"
#
# ================================================================================
# group 3 txt into 1

if True:
  #load_all_txt()
  # transform into w/s 
  #result_ = filter_mark_txt()
  # cancel the marker ",.;"
  filter_marker()


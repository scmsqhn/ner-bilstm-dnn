#coding=utf-8
import re
import pandas as pd
import traceback
import codecs

global cnt
cnt= 0 
len_1 = 0
len_0 = 0
global len_2
len_2 = 0
global len_3
len_3 = 0
global len_4
len_4 = 0

def pt(i,len_0, len_1, len_2, len_3, len_4):
  global cnt
  cnt+=1
  try:
    #print("type i ",i, type(i))
    print(tpd[i], type(tpd[i]))
    if str(tpd[i]) == '0':
        len_0+=1
        print(len_0)
    elif str(tpd[i]) == '1':
        len_1+=1
        print(len_1)
    elif str(tpd[i]) == '2':
        len_2+=1
        print(len_2)
    elif str(tpd[i]) == '3':
        len_3+=1
        print(len_3)
    else:
        print(tpd[i])
        while(1):
            pass
    print(i)
    print("\n")
    return len_0, len_1, len_2, len_3, len_4
  except:
    len_4+=1
    #except KeyError:
    #traceback.print_exc()
    #continue
    pass
    return len_0, len_1, len_2, len_3, len_4

g = pd.read_csv("pred_result.csv", encoding="utf-8")
tp = g.iloc[:,[0,-1]]

tpd = {}
tpd = dict(zip(tp.iloc[:,0], tp.iloc[:,1]))
for i in tpd.keys():
    i = i#.encode('utf-8')
#print(tpd)
f = codecs.open("risk_name_lst.txt", "r", encoding='utf-8')
cont= f.read()
#cont = re.sub("[^\u4e00-\u9fa5]","",cont)
lines = cont.split("\n")

for i in lines:
  #print(cnt, "======")
  len_0, len_1, len_2, len_3, len_4 = pt(i,len_0,len_1,len_2,len_3,len_4)


"""""
print("分类len_0",len_0)
print("分类len_1",len_1)
print("分类len_2",len_2)
print("分类len_3",len_3)
print("无数据",len_4)
print("违约企业总数:",cnt)
"""

keylst = ['cls1','cls2','cls3','cls4','cls5_no_data','totval_fault_cnt']
value2dlst = [[len_0, len_1, len_2, len_3, len_4, cnt]]
from beautifultable import BeautifulTable
def tabout(keylst, value2dlst):
    table = BeautifulTable()
    table.column_headers = keylst
    for i in value2dlst:
        table.append_row(i)
    print(table)

tabout(keylst, value2dlst)

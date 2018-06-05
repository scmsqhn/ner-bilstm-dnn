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

def pt(i,len_0,len_1):
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
    else:
        print(tpd[i])
        while(1):
            pass
    print(i)
    print("\n")
    return len_0, len_1
  except:
    global len_2
    len_2+=1
    #except KeyError:
    #traceback.print_exc()
    #continue
    pass
    return len_0, len_1


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
  len_0, len_1 = pt(i,len_0,len_1)

print("len_0",len_0)
print("len_1",len_1)
print("len_2",len_2)
print("cnt",cnt)

print("=="*10)
print(tp[tp['pred']==-1])
print("=="*10)
print(tp[tp['pred']==0])
for i in tp[tp['pred']==0].values:
    print(i)
print("=="*10)
print(tp[tp['pred']==2])
print("=="*10)
print(tp[tp['pred']==3])
print("=="*10)

#print("对已经发生违企业的聚类情况")
#print("37 79 98 214")
#print("预测正确　遗漏　无数据　总违约企业")



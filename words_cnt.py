# to connt the words  from ner_samples.txt

import jieba
import re
import os
import sys

_l = []

with open("/home/siyuan/data/ner_sample.txt","r") as f:
    cont = f.read()
    lines = cont.split('\n')
    for line in range(0,int(0.1*len(lines))):
        _l.append(line)

_words_cnt = {}

def cnt_words(_words_cnt, lines):
  for line in lines:
    words = jieba.cut(line, HMM=True)
    for word in words:
      if word in _words_cnt.keys():
        _words_cnt[word]+=1
      else:
        _words_cnt = 1

cnt_words(_words_cnt, _l)

import collections

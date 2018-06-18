lb = open('/home/distdev/iba/dmp/gongan/shandong_crim_classify/data/lb.txt')
pred = open('/home/distdev/iba/dmp/gongan/shandong_crim_classify/data/pred.txt')
dataeval = open('/home/distdev/iba/dmp/gongan/shandong_crim_classify/data/eval.txt.bak')

res = []
dct = {}

lines = dataeval.readlines()
for line in lines:
    items = line.split("\t")
    res.append(dct[items[0]])

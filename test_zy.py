import data_generator
import numpy as np

def zy_mat(y_str):
  A = {}
  B = {}
  status_lst = data_generator.tags[1:]
  for i in status_lst:
    for j in status_lst:
      A["%s%s"%(i,j)] = 1e-9 
      B[i] = 1e-9
#  print(A)
#  print(B)

  zy = dict()
  label = y_str
  for t in range(len(label) - 1):
      key = label[t] + label[t+1]
      print(key)
      A[key] += 1.0
      B[label[t]] += 1.0

  zy = {}
  zy_keys = list(set(A.keys()))
  for key in zy_keys:
     zy[key] = A[key] / B[key[1]]

  keys = sorted(zy.keys())
  print('the transition probability: ')
  #for key in keys:
  #    print(key, zy[key])
  zy = {i:np.log(zy[i]+1) for i in list(zy.keys())}
  print(zy)
  return zy


zy_mat("bibibieeeennnuvvvvvnnnppphhhhxxx")

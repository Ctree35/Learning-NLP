import numpy as np
from numpy import random

alpha_list = 'abcdefghijklmnopqrstuvwxyz'

def single_pos_data_gen(m):
  rand_ary = [alpha_list[i] for i in random.randint(0, 26, m//2 if m % 2 == 0 else m//2+1)]
  for i in reversed(range(m//2)):
    rand_ary.append(rand_ary[i])
  return rand_ary


def check(lst):
  for i in range(len(lst)):
    if not lst[i] == lst[len(lst)-i-1]:
      return False
  return True

def single_neg_data_gen(m):
  rand_ary = [alpha_list[i] for i in random.randint(0, 26, m)]
  if not check(rand_ary):
    return rand_ary
  else:
    return single_neg_data_gen(m)

def data_to_tup(raw_data):
  def pt_xform(x):
    num = ord(x) - ord('a')
    ans = []
    if num > 0:
      for i in range(num):
        ans.append(0)
    ans.append(1)
    while len(ans) < 26:
      ans.append(0)
    return ans
  return [pt_xform(x) for x in raw_data]

def gen_data_batch(batchsize, examplesize, pos_neg = None):
  dataz = []
  labelz = []
  for i in range(0, batchsize):
    label_i = random.rand() > 0.5
    if pos_neg == True:
      label_i = True
    if pos_neg == False:
      label_i = False
    data_i = single_pos_data_gen(examplesize) if label_i else single_neg_data_gen(examplesize)
    dataz.append(data_to_tup(data_i))
    labelz.append([1., 0.] if label_i else [0., 1.])
  return np.array(dataz, np.float32), np.array(labelz, np.float32)
import os
import numpy as np
import time

def get_gpu_util():
  strs = os.popen("nvidia-smi").read().split("\n")

  txt = strs[9]
  txt = txt.split('|')[3]

  txt = txt.strip().split(' ')[0].split('%')[0]
  return int(txt)


utils = []
utils_1m = []
utils_5m = []
for i in range(10000):
  util = get_gpu_util()
  utils.append(util)
  utils_1m.append(util)
  utils_5m.append(util)
  if len(utils_1m) > 60 * 2:
    utils_1m = utils_1m[1:]
  if len(utils_1m) > 60 * 2 * 5:
    utils_1m = utils_1m[1:]
  #print('util {}%'.format(util))
  time.sleep(0.5)
  print('{}: cur {}%, avg util {:.1f}%, 1min {:.1f}%, 5min {:.1f}%'.format(i, util, np.average(utils), np.average(utils_1m),np.average(utils_5m)))

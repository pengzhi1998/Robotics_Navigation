import os
from utils import *
import numpy as np

my_open = open('./adaptation.txt', "r")
lines = my_open.readlines()
whole_data = []
data_onemodel = []
data_onetrajectory = []

t = 0
data_perrun = []
data_whole = []
for line in lines:
    if (t + 2) % 3 == 0:
        data_perrun.append(np.round(float(line.strip("\n")), 2))
    if (t+2)%30 == 0:
        data_whole.append(data_perrun)
        data_perrun = []
    t += 1

for i in range(len(data_whole)):
    print(np.round(np.mean(data_whole[i]), 2), np.round(np.std(data_whole[i]), 2))


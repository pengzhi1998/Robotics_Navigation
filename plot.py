import re
import os
from utils import *
import numpy as np

my_open = open(os.path.join(assets_dir(), 'learned_models/test_rewards_randomization.txt'), "r")
lines = my_open.readlines()
whole_data = []
data_piece = []
t = 0

for line in lines:
    if line.split():
        if 'test' in line and t!=0:
            whole_data.append(data_piece)
            data_piece = []
        elif t!=0:
            line = line.strip("\n").split()
            data_piece.append(line)
    t += 1
whole_data.append(data_piece)
print(np.shape(whole_data))

_plotdata = []

t = 1
for data in whole_data:
    succ_num = 0.
    steps = []
    for i in range(len(data)):
        if i != 0:
            if data[i][0] == 'success':
                succ_num += 1.
                steps.append(float(data[i][1])/2)
    if len(steps) == 0:
        steps.append(0)
    successful_ratio = succ_num/100.0
    rewards = [float(i[2]) for i in data]
    print(t, np.round(np.mean(rewards[1:]), 2), np.round(np.std(rewards[1:]), 2), np.round(np.mean(steps), 2),
        np.round(np.std(steps), 2), successful_ratio)
    _plotdata.append([
        t, np.round(np.mean(rewards[1:]), 2), np.round(np.std(rewards[1:]), 2), np.round(np.mean(steps), 2),
        np.round(np.std(steps), 2), successful_ratio * 100
    ])
    t += 1

plot_data = []
for i in range(len(_plotdata)):
    if i%3 == 0:
        plot_data.append(_plotdata[i])

for i in range(len(_plotdata)):
    if i%3 == 1:
        plot_data.append(_plotdata[i])

for i in range(len(_plotdata)):
    if i%3 == 2:
        plot_data.append(_plotdata[i])

for i in range(len(plot_data)):
    print(i+1, ":", plot_data[i])
import os
from utils import *
import numpy as np

my_open = open(os.path.join(assets_dir(), 'learned_models/test_efficiency.txt'), "r")
lines = my_open.readlines()
whole_data = []
data_onemodel = []
data_onetrajectory = []
t = 0

for line in lines:
    if line.split():
        if 'with' in line and t!=0:
            whole_data.append(data_onemodel)
            data_onemodel = []
        elif t!=0:
            line = line.strip("\n").split()
            data_onetrajectory.append(line)
    else:
        data_onemodel.append(data_onetrajectory)
        data_onetrajectory = []
    t += 1
data_onemodel.append(data_onetrajectory)
whole_data.append(data_onemodel)
# print(np.shape(whole_data), whole_data)

data = []
data_success = 0
for data_onemodel in whole_data:
    for wp in range(5):
        data_model_wp = []
        for data_onetrajectory in data_onemodel:
            if wp == 0:
                if data_onetrajectory[wp][1] == "Success":
                    data_piece = float(data_onetrajectory[wp][0])/2.
                    data_model_wp.append(data_piece)
            else:
                if data_onetrajectory[wp][1] == "Success":
                    data_piece = (float(data_onetrajectory[wp][0]) - float(data_onetrajectory[wp-1][0]))/2.
                    data_model_wp.append(data_piece)
            # print(data_model_wp)
            if wp == 4 and data_onetrajectory[wp][1] == "Success":
                data_success += 1.
        print(np.round(np.mean(data_model_wp), 2), np.round(np.std(data_model_wp), 2))
        data.append(data_model_wp)
    print(data_success/100.)
    data_success = 0




# _plotdata = []
#
# t = 1
# for data in whole_data:
#     succ_num = 0.
#     steps = []
#     for i in range(len(data)):
#         if data[i][0] == 'success':
#             succ_num += 1.
#             steps.append(float(data[i][1]))
#     if len(steps) == 0:
#         steps.append(0)
#     successful_ratio = succ_num/10.0
#     rewards = [float(i[2]) for i in data]
#     print(t, np.round(np.mean(rewards), 2), np.round(np.std(rewards), 2), np.round(np.mean(steps), 2),
#         np.round(np.std(steps), 2), successful_ratio)
#     _plotdata.append([
#         t, np.round(np.mean(rewards), 2), np.round(np.std(rewards), 2), np.round(np.mean(steps), 2),
#         np.round(np.std(steps), 2), successful_ratio * 100
#     ])
#     t += 1
#
# plot_data = []
# for i in range(len(_plotdata)):
#     if i%3 == 0:
#         plot_data.append(_plotdata[i])
#
# for i in range(len(_plotdata)):
#     if i%3 == 1:
#         plot_data.append(_plotdata[i])
#
# for i in range(len(_plotdata)):
#     if i%3 == 2:
#         plot_data.append(_plotdata[i])
#
# for i in range(len(plot_data)):
#     print(i+1, ":", plot_data[i])

# withoutechosounder = [540, 548, 543, 585, 501, 594] # 60%
# withechosounder = [463, 465, 466, 453, 460, 457, 467, 466, 467, 464]
# bug2 = [734, 729, 729, 729, 729, 729, 729, 729, 729, 729]
#
# mean_withoutechosounder, std_withoutechosounder = np.mean(withoutechosounder), np.std(withoutechosounder)
# mean_withechosounder, std_withechosounder = np.mean(withechosounder), np.std(withechosounder)
# mean_bug2, std_bug2 = np.mean(bug2), np.std(bug2)
# print(mean_withoutechosounder, std_withoutechosounder, mean_withechosounder, std_withechosounder, mean_bug2, std_bug2)
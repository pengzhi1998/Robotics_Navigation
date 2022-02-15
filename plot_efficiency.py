# refer from: https://blog.csdn.net/weixin_43790779/article/details/110143115
import os
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

my_open = open(os.path.join(assets_dir(), 'learned_models/test_pos_plot_efficiency.txt'), "r")
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
            data_onetrajectory.append([float(line[0]), float(line[1])])
    else:
        data_onemodel.append(data_onetrajectory)
        data_onetrajectory = []
    t += 1
data_onemodel.append(data_onetrajectory)
whole_data.append(data_onemodel)

path_withechosounder = whole_data[0]
path_bug2 = whole_data[1]
path_withoutechosounder = whole_data[2]
colors = ['dodgerblue', 'g', 'm', 'r', 'c']

t = 0
for wp in path_withechosounder:
    for i in range(len(wp) - 1):
        start = (wp[i][0], wp[i + 1][0])
        end = (wp[i][1], wp[i + 1][1])
        plt.plot(start, end, linewidth=2., color=colors[t])
    t += 1

''' goals '''
plt.plot(5, -5, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(4, -6, 1, color='k', fontsize=12)
plt.plot(-10, 0, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, -1, 2, color='k', fontsize=12)
plt.plot(-10, 10, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, 9, 3, color='k', fontsize=12)
plt.plot(-20, 20, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-21, 19, 4, color='k', fontsize=12)
plt.plot(-10, 50, color='k', marker='o', markerfacecolor='yellow', markersize=80)
plt.text(-11, 49, 5, color='k', fontsize=20)

''' initial position '''
circle1 = plt.Circle((14, -15), 1, color='g')
circle1.set_zorder(1000)
plt.gca().add_patch(circle1)

''' obstacles '''
plt.plot(11.2, -10.3, color='k', marker='o', markersize=8)
plt.plot(-3.01, -12.88, color='k', marker='o', markersize=8)
plt.plot(-12.96, -9.97, color='k', marker='o', markersize=8)
plt.plot(1.65, -4.75, color='k', marker='o', markersize=8)
plt.plot(-9.58, -2.21, color='k', marker='o', markersize=8)
plt.plot(-19.93, -3.11, color='k', marker='o', markersize=8)
plt.plot(8.65, 3.82, color='k', marker='o', markersize=8)
plt.plot(-2.53, 4.18, color='k', marker='o', markersize=8)
plt.plot(-20.13, 12.2, color='k', marker='o', markersize=8)
plt.plot(-14.07, 13.6, color='k', marker='o', markersize=8)
plt.plot(0.43, 14.04, color='k', marker='o', markersize=8)
plt.plot(7.73, 10.07, color='k', marker='o', markersize=8)
plt.plot(14.19, 13.17, color='k', marker='o', markersize=8)

d_line = mlines.Line2D([], [], color='dodgerblue',
                          markersize=15, label='Path 1')
g_line = mlines.Line2D([], [], color='g',
                          markersize=15, label='Path 2')
m_line = mlines.Line2D([], [], color='m',
                          markersize=15, label='Path 3')
r_line = mlines.Line2D([], [], color='r',
                          markersize=15, label='Path 4')
c_line = mlines.Line2D([], [], color='c',
                          markersize=15, label='Path 5')
plt.legend(handles=[d_line, g_line, m_line, r_line, c_line])

plt.axis('scaled')
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(axis='x')
plt.grid(axis='y')
plt.savefig("./assets/learned_models/withechosounder.pdf")
plt.show()

t = 0
for wp in path_withoutechosounder:
    for i in range(len(wp) - 1):
        start = (wp[i][0], wp[i + 1][0])
        end = (wp[i][1], wp[i + 1][1])
        plt.plot(start, end, linewidth=2., color=colors[t])
    t += 1
plt.plot(5, -5, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(4, -6, 1, color='k', fontsize=12)
plt.plot(-10, 0, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, -1, 2, color='k', fontsize=12)
plt.plot(-10, 10, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, 9, 3, color='k', fontsize=12)
plt.plot(-20, 20, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-21, 19, 4, color='k', fontsize=12)
plt.plot(-10, 50, color='k', marker='o', markerfacecolor='yellow', markersize=80)
plt.text(-11, 49, 5, color='k', fontsize=20)
circle1 = plt.Circle((14, -15), 1, color='g')
circle1.set_zorder(1000)
plt.gca().add_patch(circle1)
plt.plot(11.2, -10.3, color='k', marker='o', markersize=8)
plt.plot(-3.01, -12.88, color='k', marker='o', markersize=8)
plt.plot(-12.96, -9.97, color='k', marker='o', markersize=8)
plt.plot(1.65, -4.75, color='k', marker='o', markersize=8)
plt.plot(-9.58, -2.21, color='k', marker='o', markersize=8)
plt.plot(-19.93, -3.11, color='k', marker='o', markersize=8)
plt.plot(8.65, 3.82, color='k', marker='o', markersize=8)
plt.plot(-2.53, 4.18, color='k', marker='o', markersize=8)
plt.plot(-20.13, 12.2, color='k', marker='o', markersize=8)
plt.plot(-14.07, 13.6, color='k', marker='o', markersize=8)
plt.plot(0.43, 14.04, color='k', marker='o', markersize=8)
plt.plot(7.73, 10.07, color='k', marker='o', markersize=8)
plt.plot(14.19, 13.17, color='k', marker='o', markersize=8)

plt.axis('scaled')
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(axis='x')
plt.grid(axis='y')
plt.savefig("./assets/learned_models/withoutechosounder.pdf")
plt.show()

t = 0
for wp in path_bug2:
    for i in range(len(wp) - 1):
        start = (wp[i][0], wp[i + 1][0])
        end = (wp[i][1], wp[i + 1][1])
        plt.plot(start, end, linewidth=2., color=colors[t])
    t += 1
plt.plot(5, -5, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(4, -6, 1, color='k', fontsize=12)
plt.plot(-10, 0, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, -1, 2, color='k', fontsize=12)
plt.plot(-10, 10, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-11, 9, 3, color='k', fontsize=12)
plt.plot(-20, 20, color='k', marker='o', markerfacecolor='yellow', markersize=15)
plt.text(-21, 19, 4, color='k', fontsize=12)
plt.plot(-10, 50, color='k', marker='o', markerfacecolor='yellow', markersize=80)
plt.text(-11, 49, 5, color='k', fontsize=20)
circle1 = plt.Circle((14, -15), 1, color='g')
circle1.set_zorder(1000)
plt.gca().add_patch(circle1)
plt.plot(11.2, -10.3, color='k', marker='o', markersize=8)
plt.plot(-3.01, -12.88, color='k', marker='o', markersize=8)
plt.plot(-12.96, -9.97, color='k', marker='o', markersize=8)
plt.plot(1.65, -4.75, color='k', marker='o', markersize=8)
plt.plot(-9.58, -2.21, color='k', marker='o', markersize=8)
plt.plot(-19.93, -3.11, color='k', marker='o', markersize=8)
plt.plot(8.65, 3.82, color='k', marker='o', markersize=8)
plt.plot(-2.53, 4.18, color='k', marker='o', markersize=8)
plt.plot(-20.13, 12.2, color='k', marker='o', markersize=8)
plt.plot(-14.07, 13.6, color='k', marker='o', markersize=8)
plt.plot(0.43, 14.04, color='k', marker='o', markersize=8)
plt.plot(7.73, 10.07, color='k', marker='o', markersize=8)
plt.plot(14.19, 13.17, color='k', marker='o', markersize=8)

plt.axis('scaled')
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(axis='x')
plt.grid(axis='y')
plt.savefig("./assets/learned_models/withbug2.pdf")
plt.show()





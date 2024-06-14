# # import numpy as np
# # import matplotlib.pyplot as plt
# # x = range(1,7,1)
# # y = [93.7,93.4,93.6,93.4,93.9,10.6]
# # plt.plot(x,y,marker='o')
# # plt.xticks(x, ('1e-5','1e-4','1e-3','1e-2','2e-2','1e-1'),rotation=60)
# # # plt.plot(np.arange(len(x)),x,label="train acc")
# # plt.show()
#
# import numpy as np
# import turtle
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# x = range(1,5,1)
# # y = [93.7,93.4,93.6,93.4,93.9]
# # y=[93.3,93.8,93.4,93.1,93.8,93.9]
# y=[2.3774,0.6088,0.1482,0.5187]
# # y=[12.6303,1.4088,0.1148,1.7540]
# # y=[0.1272,0.0409,0.0106,0.0320]
# # y=[0.9268,0.9918,0.9993,0.9898]
# # t = turtle.Turtle()
# # t.fillcolor('orange')
#
# # for x, y in zip(x_data, y_data):
# # plt.axvline(x=x, color='b', linestyle='-')
# # plt.axhline(y=y, color='g', linestyle='-')
# # plt.axvline(x=x, color='b', linestyle='-')
# # plt.axhline(y=y.all(), color='g', linestyle='-')
# fig, ax = plt.subplots()
# plt.figure(figsize=(11, 8.5))
# ax.set_title("Amplitude+Phase", fontsize=24)
# plt.bar(x,y,color=['darkblue'])
# # plt.scatter(x,y,marker='|',s=1000,color='red',linewidths=2)
# # plt.scatter(x,y,marker='s',s=80,color='red',edgecolors='crimson',linewidths=1.5)
#
# # plt.plot(x,y,color='cornflowerblue')
#
# # plt.plot(x,y,'|k')
# plt.yticks(y,fontsize=22)
# plt.xticks(x, ('1e-5','1e-4','1e-3','1e-2'),fontsize=22)
# tick_loc = [x + 0.5 for x in range(4)]
# ax.set_xticks(tick_loc)  # 设置x轴坐标位置
# # ax.set_xticklabels(list('1e-5,1e-4,1e-3,1e-2'))
# # plt.xticks(x, ('0.1','0.2','0.3','1','3','10'),fontsize=22)
# # plt.plot(np.arange(len(x)),x,label="train acc")
# # plt.grid(which='major', axis='x', linewidth=0.4, linestyle='-.', color='0.75',dashes=(10, 10))
# # plt.grid(which='major', axis='y', linewidth=0.4, linestyle='-.', color='0.75',dashes=(10, 10))
# # plt.show()
# # plt.xlabel(r'$\theta$ ($\times$ $10^{-3}$)',fontsize=24)
# plt.xlabel(r'$\Theta$',fontsize=24)
# plt.ylabel('平均绝对误差（MAE）',fontsize=24)
# # plt.ylabel('均方误差（MSE)',fontsize=24)
# # plt.ylabel('平均绝对百分比误差（MAPE）',fontsize=24)
# # plt.ylabel('决定系数（R2）',fontsize=24)
# # plt.savefig('F:yuliengxu/pre/lambda.eps')
# plt.show()
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
# f1_1 = [46.5,58.3,45.5,0.412,0.462]
# f1_1 = [92.0,82.4,74.1,65.9,65.5,55.1]
f1_1 = [19,75.0,85.4,82.64,68.0,72.9] #学习率
# f1_1 = [75.69,78.47,80.56,82.64,77.78]
# f1_1=[2.3774,0.6088,0.1482,0.5187]
# f1_1=[12.6303,1.4088,0.1148,1.7540]
# f1_1=[0.1272,0.0409,0.0106,0.0320]
# f1_1=[0.9268,0.9918,0.9993,0.9898]
# x = ['0','0.001','0.005','0.1','0.2']
x = ['1e-1','1e-2','1e-3','5e-3','1e-4','5e-4']
# x = ['1','2','3','4','5','6']
x_len = np.arange(len(x))
total_width, n = 0.9, 3
width = 0.45
xticks = x_len - (total_width - width) / 2#92a6be
plt.figure(figsize=(9.51869, 7), dpi=167)

ax = plt.axes()
plt.grid(axis="y", c='#d2c9eb', linestyle='--', zorder=0)
plt.bar(xticks, f1_1, width=0.9 * width, color="cornflowerblue",align='edge', edgecolor='black', linewidth=2,
        zorder=5)
x_num=np.arange(len(x))
plt.xlim(min(x_num)-1,max(x_num)+1)
# plt.bar(xticks + width, f1_2, width=0.9 * width, label="Official", color="#c48d60", edgecolor='black', linewidth=2,
#         zorder=10)
# plt.text(xticks[0], f1_1[0] + 0.3, f1_1[0], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
# plt.text(xticks[1], f1_1[1] + 0.3, f1_1[1], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
# plt.text(xticks[2], f1_1[2] + 0.3, f1_1[2], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)
# plt.text(xticks[3], f1_1[3] + 0.3, f1_1[3], ha='center', fontproperties='Times New Roman', fontsize=35, zorder=10)

# plt.text(xticks[0] + width, f1_2[0] + 0.3, f1_2[0], ha='center', fontproperties='Times New Roman', fontsize=35,
#          zorder=10)
# plt.text(xticks[1] + width, f1_2[1] + 0.3, f1_2[1], ha='center', fontproperties='Times New Roman', fontsize=35,
#          zorder=10)
# plt.text(xticks[2] + width, f1_2[2] + 0.3, f1_2[2], ha='center', fontproperties='Times New Roman', fontsize=35,
#          zorder=10)

# plt.legend(prop={'family': 'Times New Roman', 'size': 35}, ncol=2)
x_len = [0,1,2,3,4,5]
# x_len = np.array(x_len)
plt.xticks([index  for index in range(len(x))], x, fontproperties='Times New Roman', fontsize=15)
# plt.yticks(fontproperties='Times New Roman', fontsize=40)
# plt.ylim(ymin=75)
plt.tick_params(labelsize=15)
# plt.yticks( f1_1, fontproperties='Times New Roman', fontsize=22)
# plt.yticks(fontproperties='Times New Roman', fontsize=40)
plt.xlabel('学习率',fontsize=15)
# plt.xlabel('lr',fontsize=20)
plt.ylabel('识别精度（%）',fontsize=15)
# plt.ylabel('所有类别精度OS（%）',fontsize=20)
# plt.ylabel('平均绝对百分比误差（MAPE）',fontsize=20)
# plt.ylabel('决定系数（R2）',fontsize=20)
plt.tick_params(labelsize=15)
ax.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
ax.spines['left'].set_color('black')

plt.show()
plt.savefig('./9.png')
print("hello")
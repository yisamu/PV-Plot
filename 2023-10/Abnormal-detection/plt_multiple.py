
#plt coeffcient,HR,p-value
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



data = pd.read_csv('cox_result.csv')

columns=data['Column']
coefficients = data['Coefficient']
hazard_ratios = data['Hazard Ratio']
p_values = data['P-Value']

# # 设置直方图宽度
# bar_width = 0.3  # 设置直方图宽度为0.3
# # current_directory = os.getcwd()
#
# # 绘制Coefficient的直方图
# plt.figure(figsize=(7, 5))
# plt.bar(columns, coefficients, color='b', alpha=0.7, width=bar_width)
# plt.xlabel('Column', fontsize=14)
# plt.ylabel('Coefficient', fontsize=14)
# plt.title('Coefficient Values', fontsize=16)
# plt.xticks(rotation=45, fontsize=13)  # 设置刻度尺字体大小
# plt.yticks(fontsize=13)  # 设置刻度尺字体大小
# plt.tight_layout()
# # Specify the file name and save the plot in the current directory
# file_name = "Coefficient Values.pdf"
# plt.savefig(file_name)
# plt.show()
#
#
# # 绘制Hazard Ratio的直方图
# plt.figure(figsize=(7, 5))
# plt.bar(columns, hazard_ratios, color='g', alpha=0.7, width=bar_width)
# plt.xlabel('Column', fontsize=14)
# plt.ylabel('Hazard Ratio', fontsize=14)
# plt.title('Hazard Ratio Values', fontsize=16)
# plt.xticks(rotation=45, fontsize=13)  # 设置刻度尺字体大小
# plt.yticks(fontsize=13)  # 设置刻度尺字体大小
# plt.tight_layout()
# # Specify the file name and save the plot in the current directory
# file_name = "Hazard Ratio Values.pdf"
# plt.savefig(file_name)
# plt.show()
#
# # 绘制P-Value的直方图
# # plt.figure(figsize=(7, 5))
# # plt.bar(columns, p_values, color='r', alpha=0.7, width=bar_width)
# # plt.xlabel('Column', fontsize=14)
# # plt.ylabel('P-Value', fontsize=14)
# # plt.title('P-Value', fontsize=16)
# # plt.xticks(rotation=45, fontsize=13)  # 设置刻度尺字体大小
# # plt.yticks(fontsize=13)  # 设置刻度尺字体大小
# # plt.tight_layout()
# # file_name = "P-Value.pdf"
# # plt.savefig( file_name)
# # plt.show()
#
# plt.bar(columns, p_values, color='r', alpha=0.7, width=bar_width)
# plt.xlabel('Column', fontsize=14)
# plt.ylabel('P-Value', fontsize=14)
# plt.title('P-Value', fontsize=16)
# plt.xticks(rotation=45, fontsize=13)  # 设置刻度尺字体大小
# plt.yticks(fontsize=13)  # 设置刻度尺字体大小
# plt.yscale('log')  # 将y轴设置为对数坐标
# plt.tight_layout()
# file_name = "Log_P-Value.pdf"
# plt.savefig(file_name)
# plt.show()

import matplotlib.pyplot as plt

# 设置直方图宽度
bar_width = 0.3  # 设置直方图宽度为0.3

# 创建一个3行1列的子图
fig, axs = plt.subplots(3, 1, figsize=(6, 10))

# 绘制Coefficient的直方图
axs[0].bar(columns, coefficients, color='b', alpha=0.7, width=bar_width)
axs[0].set_ylabel('Coefficient', fontsize=14)
axs[0].set_title('Coefficient Values', fontsize=16)
axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 清空横坐标标签
axs[0].tick_params(axis='y', labelsize=13)

# 绘制Hazard Ratio的直方图
axs[1].bar(columns, hazard_ratios, color='g', alpha=0.7, width=bar_width)
axs[1].set_ylabel('Hazard Ratio', fontsize=14)
axs[1].set_title('Hazard Ratio Values', fontsize=16)
axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # 清空横坐标标签
axs[1].tick_params(axis='y', labelsize=13)

# 绘制P-Value的直方图（取对数）
axs[2].bar(columns, p_values, color='r', alpha=0.7, width=bar_width)
axs[2].set_xlabel('Column', fontsize=14)  # 只设置最后一个子图的横坐标标签
axs[2].set_ylabel('Log P-Value', fontsize=14)
axs[2].set_title('Log P-Value', fontsize=16)
axs[2].set_yscale('log')  # 将y轴设置为对数坐标
axs[2].tick_params(axis='x', rotation=45, labelsize=13)
axs[2].tick_params(axis='y', labelsize=13)

plt.tight_layout()  # 调整子图的布局
file_name = "Multiple_Plots.pdf"
plt.savefig(file_name)
plt.show()

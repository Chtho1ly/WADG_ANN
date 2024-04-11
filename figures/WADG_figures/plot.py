'''
Author: Chen Shi
Date: 2024-03-21 14:28:44
Description: Plot class. Draw figures.
'''

import matplotlib.pyplot as plt
# 平滑曲线图
from scipy.interpolate import make_interp_spline
import numpy as np
import csv


class Plot:
    def __init__(self, query_num:int, wadg_parmeters:str, top_k:int,
                 nsg_csv_file_path, wadg_csv_file_path):
        self.nsg_pre = []
        self.wadg_pre = []

        self.nsg_qps = []
        self.wadg_qps = []
        
        self.query_num = query_num
        self.top_k = top_k
        self.wadg_parmeters = wadg_parmeters

        # read
        self.read_csv(nsg_csv_file_path, 'nsg')
        self.read_csv(wadg_csv_file_path, 'wadg')

    # read csv file
    def read_csv(self, csv_file_path, nsg_or_wadg):
        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            # header is no need
            next(reader)
            for row in reader:
                if (nsg_or_wadg == 'nsg'):
                    self.nsg_pre.append(float(row[0]))
                    self.nsg_qps.append(self.query_num / float(row[1]))
                elif (nsg_or_wadg == 'wadg'):
                    self.wadg_pre.append(float(row[0]))
                    self.wadg_qps.append(self.query_num / float(row[1]))
        
        # print and check
        if (nsg_or_wadg == 'nsg'):
            print("nsg_pre:", self.nsg_pre)
            print("nsg_qps:", self.nsg_qps)
        elif (nsg_or_wadg == 'wadg'):
            print("wadg_pre:", self.wadg_pre)
            print("wadg_qps:", self.wadg_qps)


    # 对比平滑曲线图
    def comparison_line_chart(self):
        x1 = self.nsg_pre
        x2 = self.wadg_pre

        y1 = self.nsg_qps
        y2 = self.wadg_qps

        x1_array = np.array(x1)
        x2_array = np.array(x2)

        y1_array = np.array(y1)
        y2_array = np.array(y2)

        x1_smooth = np.linspace(x1_array.min(), x1_array.max(), 300)
        x2_smooth = np.linspace(x2_array.min(), x2_array.max(), 300)

        y1_smooth = make_interp_spline(x1_array, y1_array)(x1_smooth)
        y2_smooth = make_interp_spline(x2_array, y2_array)(x2_smooth)

        # 子绘图 1
        # subplot(rows, cols, index)
        # 绘图一行两个
        # plt.subplot(1, 2, 1) 
        # 设置字体
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        # x轴标题
        plt.xlabel('Precision@100')
        plt.xlim(0.96, 1)
        # y轴标题
        plt.ylabel('Queries per Second')
        # 绘制平滑曲线图，添加数据点，设置点的大小
        plt.plot(x1_smooth, y1_smooth)  
        plt.plot(x2_smooth, y2_smooth)
        # 曲线上打点
        plt.scatter(x1_array, y1_array, marker = 's')
        plt.scatter(x2_array, y2_array, marker = 's')
        # 设置曲线名称
        plt.legend(['NSG', 'WADG'])
        plt.grid(linestyle = '--', alpha = 0.5)
        # plt.title('SIFT, query_num=' + str(self.query_num) + ', wadg_args=' + self.wadg_parmeters +
        #           ', Top-K=' + str(self.top_k))
        plt.title(f'SIFT, query_num={self.query_num}, wadg_args={self.wadg_parmeters}, Top-K={self.top_k}')

        plt.show()


if __name__ == "__main__":
    """
    query_num:
    wadg_parameters: hot_point_num - window_size - cluster_num
    """
    p = Plot(10000, "2K-2K-K", 100, './nsg.csv', './wadg.csv')
    p.comparison_line_chart()
    # p.comparision_histogram()

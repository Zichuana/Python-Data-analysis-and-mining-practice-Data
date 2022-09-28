import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 引入KMeans
import matplotlib.pyplot as plt

datafile = './data/discretization_data.xls'  # 参数初始化
data = pd.read_excel(datafile)  # 读取数据
data = data['肝气郁结证型系数'].copy()
k = 4

d1 = pd.cut(data, k, labels=range(k))  # 等宽离散化，各个类比依次命名为0,1,2,3

# 等频率离散化
w = [1.0 * i / k for i in range(k + 1)]
w = data.describe(percentiles=w)[4:4 + k + 1]  # 使用describe函数自动计算分位数
w[0] = w[0] * (1 - 1e-10)
d2 = pd.cut(data, w, labels=range(k))


model = KMeans(n_clusters=k)  # 建立模型，n_jobs是并行数，一般等于CPU数较好
model.fit(np.array(data).reshape((len(data), 1)))  # 训练模型
c = pd.DataFrame(model.cluster_centers_).sort_values(0)  # 输出聚类中心，并且排序（默认是随机序的）
w = c.rolling(2).mean()  # 相邻两项求中点，作为边界点
w = w.dropna()
w = [0] + list(w[0]) + [data.max()]  # 把首末边界点加上
d3 = pd.cut(data, w, labels=range(k))


def cluster_plot(d, k, cot):  # 自定义作图函数来显示聚类结果
    plt.figure(figsize=(8, 3))
    for j in range(0, k):
        plt.plot(data[d == j], [j for i in d[d == j]], 'o')

    plt.ylim(-0.5, k - 0.5)
    plt.savefig('./data/'+str(cot)+'.png', dpi=300)
    return plt


cluster_plot(d1, k, 1).show()
cluster_plot(d2, k, 2).show()
cluster_plot(d3, k, 3).show()






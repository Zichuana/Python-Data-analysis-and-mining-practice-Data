import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import Lasso
import sys
import numpy as np
import pandas as pd
from GM11 import GM11  # 引入自编的灰色预测函数
from sklearn.svm import LinearSVR

from matplotlib import font_manager
print(matplotlib.get_data_path())
print(matplotlib.get_cachedir())
# exit()
# matplotlib.rcParams['font.sans-serif'] = 'WenQuanYi Micro Hei'
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# font_path = '/share/zh_CN/msyh.ttf'  # ttf的路径 最好是具体路径
# font_manager.fontManager.addfont(font_path)


inputfile = './data_chapter6/data.csv'  # 输入的数据文件
data = pd.read_csv(inputfile)  # 读取数据

# 描述性统计分析
description = [data.min(), data.max(), data.mean(), data.std()]  # 依次计算最小值、最大值、均值、标准差
# print(description)
description = pd.DataFrame(description, index=['Min', 'Max', 'Mean', 'STD']).T  # 将结果存入数据框
# print(description)
print('描述性统计结果：\n', np.round(description, 2))  # 保留两位小数

# 相关性分析
corr = data.corr(method='pearson')  # 计算相关系数矩阵
print('相关系数矩阵为：\n', np.round(corr, 2))  # 保留两位小数

# 绘制热力图
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.subplots(figsize=(10, 10))  # 设置画面大小
sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Blues")
plt.title('相关性热力图')
plt.show()
plt.close()

lasso = Lasso(1000)  # 调用Lasso()函数，设置λ的值为1000
lasso.fit(data.iloc[:, 0:13], data['y'])
print('相关系数为：', np.round(lasso.coef_, 5))  # 输出结果，保留五位小数

print('相关系数非零个数为：', np.sum(lasso.coef_ != 0))  # 计算相关系数非零的个数

mask = lasso.coef_ != 0  # 返回一个相关系数是否为零的布尔数组
print('相关系数是否为零：', mask)
# print(len(mask))
# print(data_chapter6)
data = data.iloc[:, 0:13]
new_reg_data = data.iloc[:, mask]  # 返回相关系数非零的数据
new_reg_data.to_csv('./data_chapter6/new_reg_data.csv')  # 存储数据
print('输出数据的维度为：', new_reg_data.shape)  # 查看输出数据的维度

# 构建灰色预测模型
inputfile1 = './data_chapter6/new_reg_data.csv'  # 输入的数据文件
inputfile2 = './data_chapter6/data.csv'  # 输入的数据文件
new_reg_data = pd.read_csv(inputfile1)  # 读取经过特征选择后的数据
data = pd.read_csv(inputfile2)  # 读取总的数据
new_reg_data.index = range(1994, 2014)
new_reg_data.loc[2014] = None
new_reg_data.loc[2015] = None
l = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']
for i in l:
  f = GM11(new_reg_data.loc[range(1994, 2014), i].values)[0]
  new_reg_data.loc[2014, i] = f(len(new_reg_data)-1)  # 2014年预测结果
  new_reg_data.loc[2015, i] = f(len(new_reg_data))  # 2015年预测结果
  new_reg_data[i] = new_reg_data[i].round(2)  # 保留两位小数
outputfile = './data_chapter6/new_reg_data_GM11.xls'  # 灰色预测后保存的路径
y = list(data['y'].values)  # 提取财政收入列，合并至新数据框中
y.extend([np.nan, np.nan])
new_reg_data['y'] = y
new_reg_data.to_excel(outputfile)  # 结果输出
print('预测结果为：\n', new_reg_data.loc[2014:2015, :])  # 预测结果展示

# 构建支持向量回归预测模型
inputfile = './data_chapter6/new_reg_data_GM11.xls'  # 灰色预测保存的路径
data = pd.read_excel(inputfile)  # 读取数据
print(data)
feature = ['x1', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x13']  # 属性所在列
data_train = data.loc[range(0, 20)]  # 取2014年前的数据建模
print(data_train)
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std  # 数据标准化
x_train = data_train[feature].values  # 属性数据
y_train = data_train['y'].values  # 标签数据

linearsvr = LinearSVR()  # 调用LinearSVR()函数
linearsvr.fit(x_train, y_train)
x = ((data[feature] - data_mean[feature])/data_std[feature]).values  # 预测，并还原结果。
data['y_pred'] = linearsvr.predict(x) * data_std['y'] + data_mean['y']
outputfile = './data_chapter6/new_reg_data_GM11_revenue.xls'  # SVR预测后保存的结果
data.to_excel(outputfile)

print('真实值与预测值分别为：\n', data[['y', 'y_pred']])

fig = data[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])  # 画出预测结果图
plt.show()

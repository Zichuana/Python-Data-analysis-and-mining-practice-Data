import pandas as pd  # 导入数据分析库Pandas
from scipy.interpolate import lagrange  # 导入拉格朗日插值函数
from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型
import joblib
from sklearn.metrics import roc_curve  # 导入ROC曲线函数
from cm_plot import *  # 导入自行编写的混淆矩阵可视化函数
import matplotlib.pyplot as plt
from random import shuffle  # 导入随机函数shuffle，用来打算数据
from keras.models import Sequential  # 导入神经网络初始化函数
from keras.layers.core import Dense, Activation  # 导入神经网络层函数、激活函数

inputfile = './data_chapter11/missing_data.xls'  # 输入数据路径,需要使用Excel格式；
outputfile = './data_chapter11/missing_data_processed.xlsx'  # 输出数据路径,需要使用Excel格式

data = pd.read_excel(inputfile, header=None)  # 读入数据
print(data)

ll = len(data)
print(ll)


# 自定义列向量插值函数
# s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    if n - k < 0:
        y = s[list(range(5 - k, 5)) + list(range(6, 6 + k))]
    elif n + k + 1 > ll - 1:
        y = s[list(range(14 - k, 14)) + list(range(14 + 1, 14 + 1 + k))]
    else:
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
    y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果


# 逐个元素判断是否需要插值
for i in data.columns:
    print("i:", i)
    for j in range(len(data)):
        if (data[i].isnull())[j]:  # 如果为空即插值。
            print(data[i])
            data[i][j] = ployinterp_column(data[i], j)

data.to_excel(outputfile, header=None, index=False)  # 输出结果

# 构建并测试CART决策树模型

datafile = './data_chapter11/model.xls'  # 数据名
data = pd.read_excel(datafile)  # 读取数据，数据的前三列是特征，第四列是标签
data = data.values  # 将表格转换为矩阵
shuffle(data)  # 随机打乱数据

p = 0.8  # 设置训练数据比例
train = data[:int(len(data) * p), :]  # 前80%为训练集
test = data[int(len(data) * p):, :]  # 后20%为测试集

# 构建CART决策树模型

treefile = './data_chapter11/tree.pkl'  # 模型输出名字
tree = DecisionTreeClassifier()  # 建立决策树模型
tree.fit(train[:, :3], train[:, 3])  # 训练

# 保存模型
joblib.dump(tree, treefile)

cm_plot(train[:, 3], tree.predict(train[:, :3])).show()  # 显示混淆矩阵可视化结果
# 注意到Scikit-Learn使用predict方法直接给出预测结果。

fpr, tpr, thresholds = roc_curve(test[:, 3], tree.predict_proba(test[:, :3])[:, 1], pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')  # 作出ROC曲线
plt.xlabel('False Positive Rate')  # 坐标轴标签
plt.ylabel('True Positive Rate')  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()  # 显示作图结果

datafile = './data_chapter11/model.xls'
data = pd.read_excel(datafile)
data = data.values
shuffle(data)

p = 0.8  # 设置训练数据比例
train = data[:int(len(data) * p), :]
test = data[int(len(data) * p):, :]

# 构建LM神经网络模型
netfile = './data_chapter11/net.model'  # 构建的神经网络模型存储路径

net = Sequential()  # 建立神经网络
net.add(Dense(input_dim=3, units=10))  # 添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Activation('relu'))  # 隐藏层使用relu激活函数
net.add(Dense(units=1))  # 添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
net.compile(loss='binary_crossentropy', optimizer='adam')  # 编译模型，使用adam方法求解

net.fit(train[:, :3], train[:, 3], epochs=1000, batch_size=1)  # 训练模型，循环1000次
net.save_weights(netfile)  # 保存模型

predict_result = (net.predict(train[:, :3]) > 0.5).astype("int32")  # 预测结果变形
'''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''
print(predict_result)
print(train[:, 3])
cm_plot(train[:, 3], predict_result).show()  # 显示混淆矩阵可视化结果

predict_result = (net.predict(test[:, :3]) > 0.5).astype("int32")
fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')  # 作出ROC曲线
plt.xlabel('False Positive Rate')  # 坐标轴标签
plt.ylabel('True Positive Rate')  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()  # 显示作图结果

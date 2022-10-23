import pandas as pd
from numpy.random import shuffle  # 引入随机函数
from sklearn import svm
from sklearn import metrics

inputfile = './data_chapter18/moment.csv'  # 数据文件
outputfile1 = './data_chapter18/cm_train.xls'  # 训练样本混淆矩阵保存路径
outputfile2 = './data_chapter18/cm_test.xls'  # 测试样本混淆矩阵保存路径
data = pd.read_csv(inputfile, encoding='gbk')  # 读取数据，指定编码为gbk
data = data.values
shuffle(data)  # 随机打乱数据
data_train = data[:int(0.8 * len(data)), :]  # 选取前80%为训练数据
data_test = data[int(0.8 * len(data)):, :]  # 选取前20%为测试数据
# 构造特征和标签
x_train = data_train[:, 2:] * 30
y_train = data_train[:, 0].astype(int)
x_test = data_test[:, 2:] * 30
y_test = data_test[:, 0].astype(int)
# 导入模型相关的函数，建立并且训练模型
model = svm.SVC()
model.fit(x_train, y_train)
# 导入输出相关的库，生成混淆矩阵
cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))
print(cm_train)
print(cm_test)
pd.DataFrame(cm_train, index=range(1, 6), columns=range(1, 6)).to_excel(outputfile1)
pd.DataFrame(cm_test, index=range(1, 6), columns=range(1, 6)).to_excel(outputfile2)

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# 用水事件划分
threshold = pd.Timedelta('4 min')  # 阈值为分钟
inputfile = './data_chapter13/water_heater.xls'  # 输入数据路径,需要使用Excel格式
outputfile = './data_chapter13/dividsequence.xlsx'  # 输出数据路径,需要使用Excel格式

data = pd.read_excel(inputfile)
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format='%Y%m%d%H%M%S')
data = data[data[u'水流量'] > 0]  # 只要流量大于0的记录
d = data[u'发生时间'].diff() > threshold  # 相邻时间作差分，比较是否大于阈值
data[u'事件编号'] = d.cumsum() + 1  # 通过累积求和的方式为事件编号

data.to_excel(outputfile)

# 阈值寻优
inputfile = './data_chapter13/water_heater.xls'  # 输入数据路径,需要使用Excel格式
n = 4  # 使用以后四个点的平均斜率

threshold = pd.Timedelta(minutes=5)  # 专家阈值
data = pd.read_excel(inputfile)
data[u'发生时间'] = pd.to_datetime(data[u'发生时间'], format='%Y%m%d%H%M%S')
data = data[data[u'水流量'] > 0]  # 只要流量大于0的记录


def event_num(ts):
    d = data[u'发生时间'].diff() > ts  # 相邻时间作差分，比较是否大于阈值
    return d.sum() + 1  # 这样直接返回事件数


dt = [pd.Timedelta(minutes=i) for i in np.arange(1, 9, 0.25)]
h = pd.DataFrame(dt, columns=[u'阈值'])  # 定义阈值列
h[u'事件数'] = h[u'阈值'].apply(event_num)  # 计算每个阈值对应的事件数
h[u'斜率'] = h[u'事件数'].diff() / 0.25  # 计算每两个相邻点对应的斜率
# h[u'斜率指标'] = pd.rolling_mean(h[u'斜率'].abs(), n)  # 采用后n个的斜率绝对值平均作为斜率指标
h[u'斜率指标'] = h[u'斜率'].abs().rolling(n).mean()
ts = h[u'阈值'][h[u'斜率指标'].idxmin() - n]
# 注：用idxmin返回最小值的Index，由于rolling_mean()自动计算的是前n个斜率的绝对值平均
# 所以结果要进行平移（-n）

if ts > threshold:
    ts = pd.Timedelta(minutes=4)

print(ts)

inputfile1 = './data_chapter13/train_neural_network_data.xls'  # 训练数据
inputfile2 = './data_chapter13/test_neural_network_data.xls'  # 测试数据
testoutputfile = './data_chapter13/test_output_data.xlsx'  # 测试数据模型输出文件
data_train = pd.read_excel(inputfile1)  # 读入训练数据(由日志标记事件是否为洗浴)
data_test = pd.read_excel(inputfile2)  # 读入测试数据(由日志标记事件是否为洗浴)
y_train = data_train.iloc[:, 4].values  # 训练样本标签列
x_train = data_train.iloc[:, 5:17].values  # 训练样本特征
y_test = data_test.iloc[:, 4].values  # 测试样本标签列
x_test = data_test.iloc[:, 5:17].values  # 测试样本特征

model = Sequential()  # 建立模型
model.add(Dense(input_dim=11, units=17))  # 添加输入层、隐藏层的连接
model.add(Activation('relu'))  # 以Relu函数为激活函数
model.add(Dense(input_dim=17, units=10))  # 添加隐藏层、隐藏层的连接
model.add(Activation('relu'))  # 以Relu函数为激活函数
model.add(Dense(input_dim=10, units=1))  # 添加隐藏层、输出层的连接
model.add(Activation('sigmoid'))  # 以sigmoid函数为激活函数
# 编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)  # 训练模型
model.save_weights('./data_chapter13/net.model')  # 保存模型参数

r = pd.DataFrame(model.predict_step(x_test), columns=[u'预测结果'])

pd.concat([data_test.iloc[:, :5], r], axis=1).to_excel(testoutputfile)
model.predict(x_test)

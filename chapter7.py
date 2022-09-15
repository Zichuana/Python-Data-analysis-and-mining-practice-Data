import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans  # 导入kmeans算法


# 对数据进行基本的探索
# 返回缺失值个数以及最大最小值
datafile = './data_chapter7/air_data.csv'  # 航空原始数据,第一行为属性标签
resultfile = './data_chapter7/explore.csv'  # 数据探索结果表

# 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）
data = pd.read_csv(datafile, encoding='utf-8')
# 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）
explore = data.describe(percentiles=[], include='all').T  # T是转置，转置后更方便查阅
explore['null'] = len(data)-explore['count']  # describe()函数自动计算非空值数，需要手动计算空值数
explore = explore[['null', 'max', 'min']]
explore.columns = ['空值数', '最大值', '最小值']  # 表头重命名
'''
这里只选取部分探索结果。
describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、
freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）
'''

explore.to_csv(resultfile)  # 导出结果


# 2. 对数据的分布分析
datafile = './data_chapter7/air_data.csv'  # 航空原始数据,第一行为属性标签

# 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）
data = pd.read_csv(datafile, encoding='utf-8')

# 客户信息类别
# 提取会员入会年份
ffp = data['FFP_DATE'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d'))
ffp_year = ffp.map(lambda x: x.year)
# 绘制各年份会员入会人数直方图
fig = plt.figure(figsize=(8, 5))  # 设置画布大小
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
plt.hist(ffp_year, bins='auto', color='#0504aa')
plt.xlabel('年份')
plt.ylabel('入会人数')
plt.title('各年份会员入会人数')
plt.show()
plt.close()

# 提取会员不同性别人数
male = pd.value_counts(data['GENDER'])['男']
female = pd.value_counts(data['GENDER'])['女']
# 绘制会员性别比例饼图
fig = plt.figure(figsize=(7, 4))  # 设置画布大小
plt.pie([male, female], labels=['男', '女'], colors=['lightskyblue', 'lightcoral'], autopct='%1.1f%%')
plt.title('会员性别比例')
plt.show()
plt.close()

# 提取不同级别会员的人数
lv_four = pd.value_counts(data['FFP_TIER'])[4]
lv_five = pd.value_counts(data['FFP_TIER'])[5]
lv_six = pd.value_counts(data['FFP_TIER'])[6]
# 绘制会员各级别人数条形图
fig = plt.figure(figsize=(8, 5))  # 设置画布大小
plt.bar(x=range(3), height=[lv_four, lv_five, lv_six], width=0.4, alpha=0.8, color='skyblue')
plt.xticks([index for index in range(3)], ['4', '5', '6'])
plt.xlabel('会员等级')
plt.ylabel('会员人数')
plt.title('会员各级别人数')
plt.show()
plt.close()

# 提取会员年龄
age = data['AGE'].dropna()
age = age.astype('int64')
# 绘制会员年龄分布箱型图
fig = plt.figure(figsize=(5, 10))
plt.boxplot(age,
            patch_artist=True,
            labels=['会员年龄'],  # 设置x轴标题
            boxprops={'facecolor': 'lightblue'})  # 设置填充颜色
plt.title('会员年龄分布箱线图')
# 显示y坐标轴的底线
plt.grid(axis='y')
plt.show()
plt.close()


# 处理缺失值与异常值
datafile = './data_chapter7/air_data.csv'  # 航空原始数据路径
cleanedfile = './data_chapter7/data_cleaned.csv'  # 数据清洗后保存的文件路径

# 读取数据
airline_data = pd.read_csv(datafile, encoding='utf-8')
print('原始数据的形状为：', airline_data.shape)

# 去除票价为空的记录
airline_notnull = airline_data.loc[airline_data['SUM_YR_1'].notnull() &
                                   airline_data['SUM_YR_2'].notnull(), :]
print('删除缺失记录后数据的形状为：', airline_notnull.shape)

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录。
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM'] > 0) & (airline_notnull['avg_discount'] != 0)
index4 = airline_notnull['AGE'] > 100  # 去除年龄大于100的记录
airline = airline_notnull[(index1 | index2) & index3 & ~index4]
print('数据清洗后数据的形状为：', airline.shape)

airline.to_csv(cleanedfile)  # 保存清洗后的数据


# 属性选择、构造与数据标准化
# 读取数据清洗后的数据
cleanedfile = './data_chapter7/data_cleaned.csv'  # 数据清洗后保存的文件路径
airline = pd.read_csv(cleanedfile, encoding='utf-8')

# 选取需求属性
airline_selection = airline[['FFP_DATE', 'LOAD_TIME', 'LAST_TO_END',
                             'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]
print('筛选的属性前5行为：\n', airline_selection.head())

# 构造属性L
L = pd.to_datetime(airline_selection['LOAD_TIME']) - pd.to_datetime(airline_selection['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int')/30

# 合并属性
airline_features = pd.concat([L, airline_selection.iloc[:, 2:]], axis=1)
airline_features.columns = ['L', 'R', 'F', 'M', 'C']
print(airline_features)
print('构建的LRFMC属性前5行为：\n', airline_features.head())

data = StandardScaler().fit_transform(airline_features)
np.savez('./data_chapter7/airline_scale.npz', data)
print(data)
print('标准化后LRFMC五个属性为：\n', data[:5, :])


# 读取标准化后的数据
airline_scale = np.load('./data_chapter7/airline_scale.npz')['arr_0']
print()
k = 5  # 确定聚类中心数

# 构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters=k, random_state=123)
fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练

# 查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_  # 聚类中心
print('各类聚类中心为：\n', kmeans_cc)
kmeans_labels = kmeans_model.labels_  # 样本的类别标签
print('各样本的类别标签为：\n', kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同类别样本的数目
print('最终每个类别的数目为：\n', r1)

# 输出聚类分群的结果
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, columns=['ZL', 'ZR', 'ZF', 'ZM', 'ZC'])   # 将聚类中心放在数据框中
print(cluster_center)
cluster_center.index = pd.DataFrame(kmeans_model.labels_).drop_duplicates().iloc[:, 0]  # 将样本类别作为数据框索引
print(cluster_center.index)

# 客户分群雷达图
labels = ['ZL', 'ZR', 'ZF', 'ZM', 'ZC']
legen = ['客户群' + str(i + 1) for i in cluster_center.index]  # 客户群命名，作为雷达图的图例

lstype = ['-', '--', (0, (3, 5, 1, 5, 1, 5)), ':', '-.']
kinds = list(cluster_center.iloc[:, 0])
print("kinds:\n", kinds)
# 由于雷达图要保证数据闭合，因此再添加L列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[['ZL']]], axis=1)
print("cluster_center\n", cluster_center)
centers = np.array(cluster_center.iloc[:, 0:])
print("centers:\n", centers)

# 分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
print("angle:\n", angle)
angle = np.concatenate((angle, [angle[0]]))
print("angle:\n", angle)
labels = np.concatenate((labels, [labels[0]]))
print(labels)

# 绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)  # 以极坐标的形式绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 画线

color = ['b', 'g', 'r', 'c', 'y']  # 指定颜色
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], color=color[i], linewidth=2, label=u'客户群'+str(i))
# 添加属性标签
ax.set_thetagrids(angle * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()
plt.close()



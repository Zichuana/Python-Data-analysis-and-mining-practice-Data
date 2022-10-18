import pandas as pd
import pymysql as pm
import re
from sqlalchemy import create_engine
import os
from random import sample

# 第一种连接方式


engine = create_engine('mysql+pymysql://root:Zqy20020227_@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)
print(engine)
print(sql)
# 第二种连接方式


con = pm.connect(host='127.0.0.1', user='root', passwd='Zqy20020227_', port=3306, db='test', charset='utf8')
data = pd.read_sql('select * from all_gzdata', con=con)
con.close()  # 关闭连接

# 保存读取的数据
data.to_csv('./data_chapter15/all_gzdata.csv', index=False, encoding='utf-8')

engine = create_engine('mysql+pymysql://root:Zqy20020227_@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)


# 分析网页类型
counts = [i['fullURLId'].value_counts() for i in sql]  # 逐块统计
counts = counts.copy()
counts = pd.concat(counts).groupby(level=0).sum()  # 合并统计结果，把相同的统计项合并（即按index分组并求和）
counts = counts.reset_index()  # 重新设置index，将原来的index作为counts的一列。
counts.columns = ['index', 'num']  # 重新设置列名，主要是第二列，默认为0
counts['type'] = counts['index'].str.extract('(\d{3})')  # 提取前三个数字作为类别id
counts_ = counts[['type', 'num']].groupby('type').sum()  # 按类别合并
counts_.sort_values(by='num', ascending=False, inplace=True)  # 降序排列
counts_['ratio'] = counts_.iloc[:, 0] / counts_.iloc[:, 0].sum()
print(counts_)


# 代码11-3

# 因为只有107001一类，但是可以继续细分成三类：知识内容页、知识列表页、知识首页
def count107(i):  # 自定义统计函数
    j = i[['fullURL']][i['fullURLId'].str.contains('107')].copy()  # 找出类别包含107的网址
    j['type'] = None  # 添加空列
    j['type'][j['fullURL'].str.contains('info/.+?/')] = '知识首页'
    j['type'][j['fullURL'].str.contains('info/.+?/.+?')] = '知识列表页'
    j['type'][j['fullURL'].str.contains('/\d+?_*\d+?\.html')] = '知识内容页'
    return j['type'].value_counts()


# 注意：获取一次sql对象就需要重新访问一下数据库(!!!)
# engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)

counts2 = [count107(i) for i in sql]  # 逐块统计
counts2 = pd.concat(counts2).groupby(level=0).sum()  # 合并统计结果
print(counts2)
# 计算各个部分的占比
res107 = pd.DataFrame(counts2)
# res107.reset_index(inplace=True)
res107.index.name = '107类型'
res107.rename(columns={'type': 'num'}, inplace=True)
res107['比例'] = res107['num'] / res107['num'].sum()
res107.reset_index(inplace=True)
print(res107)


# 代码11-4

def countquestion(i):  # 自定义统计函数
    j = i[['fullURLId']][i['fullURL'].str.contains('\?')].copy()  # 找出类别包含107的网址
    return j


# engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)

counts3 = [countquestion(i)['fullURLId'].value_counts() for i in sql]
counts3 = pd.concat(counts3).groupby(level=0).sum()
print(counts3)

# 求各个类型的占比并保存数据
df1 = pd.DataFrame(counts3)
df1['perc'] = df1['fullURLId'] / df1['fullURLId'].sum() * 100
df1.sort_values(by='fullURLId', ascending=False, inplace=True)
print(df1.round(4))


# 代码11-5

def page199(i):  # 自定义统计函数
    j = i[['fullURL', 'pageTitle']][(i['fullURLId'].str.contains('199')) &
                                    (i['fullURL'].str.contains('\?'))]
    j['pageTitle'].fillna('空', inplace=True)
    j['type'] = '其他'  # 添加空列
    j['type'][j['pageTitle'].str.contains('法律快车-律师助手')] = '法律快车-律师助手'
    j['type'][j['pageTitle'].str.contains('咨询发布成功')] = '咨询发布成功'
    j['type'][j['pageTitle'].str.contains('免费发布法律咨询')] = '免费发布法律咨询'
    j['type'][j['pageTitle'].str.contains('法律快搜')] = '快搜'
    j['type'][j['pageTitle'].str.contains('法律快车法律经验')] = '法律快车法律经验'
    j['type'][j['pageTitle'].str.contains('法律快车法律咨询')] = '法律快车法律咨询'
    j['type'][(j['pageTitle'].str.contains('_法律快车')) |
              (j['pageTitle'].str.contains('-法律快车'))] = '法律快车'
    j['type'][j['pageTitle'].str.contains('空')] = '空'

    return j


# 注意：获取一次sql对象就需要重新访问一下数据库
# engine = create_engine('mysql+pymysql://root:123456@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)  # 分块读取数据库信息
# sql = pd.read_sql_query('select * from all_gzdata limit 10000', con=engine)

counts4 = [page199(i) for i in sql]  # 逐块统计
counts4 = pd.concat(counts4)
d1 = counts4['type'].value_counts()
print(d1)
d2 = counts4[counts4['type'] == '其他']
print(d2)
# 求各个部分的占比并保存数据
df1_ = pd.DataFrame(d1)
df1_['perc'] = df1_['type'] / df1_['type'].sum() * 100
df1_.sort_values(by='type', ascending=False, inplace=True)
print(df1_)


# 代码11-6

def xiaguang(i):  # 自定义统计函数
    j = i.loc[(i['fullURL'].str.contains('\.html')) == False,
              ['fullURL', 'fullURLId', 'pageTitle']]
    return j


# 注意获取一次sql对象就需要重新访问一下数据库
engine = create_engine('mysql+pymysql://root:Zqy20020227_@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)  # 分块读取数据库信息

counts5 = [xiaguang(i) for i in sql]
counts5 = pd.concat(counts5)

xg1 = counts5['fullURLId'].value_counts()
print(xg1)
# 求各个部分的占比
xg_ = pd.DataFrame(xg1)
xg_.reset_index(inplace=True)
xg_.columns = ['index', 'num']
xg_['perc'] = xg_['num'] / xg_['num'].sum() * 100
xg_.sort_values(by='num', ascending=False, inplace=True)

xg_['type'] = xg_['index'].str.extract('(\d{3})')  # 提取前三个数字作为类别id

xgs_ = xg_[['type', 'num']].groupby('type').sum()  # 按类别合并
xgs_.sort_values(by='num', ascending=False, inplace=True)  # 降序排列
xgs_['percentage'] = xgs_['num'] / xgs_['num'].sum() * 100

print(xgs_.round(4))

# 代码11-7
#
# 分析网页点击次数
# 统计点击次数
engine = create_engine('mysql+pymysql://root:Zqy20020227_@127.0.0.1:3306/test?charset=utf8')
sql = pd.read_sql('all_gzdata', engine, chunksize=10000)  # 分块读取数据库信息

counts1 = [i['realIP'].value_counts() for i in sql]  # 分块统计各个IP的出现次数
counts1 = pd.concat(counts1).groupby(level=0).sum()  # 合并统计结果，level=0表示按照index分组
print(counts1)

counts1_ = pd.DataFrame(counts1)
counts1_
counts1['realIP'] = counts1.index.tolist()

counts1_[1] = 1  # 添加1列全为1
hit_count = counts1_.groupby('realIP').sum()  # 统计各个“不同点击次数”分别出现的次数
# 也可以使用counts1_['realIP'].value_counts()功能
hit_count.columns = ['用户数']
hit_count.index.name = '点击次数'

# 统计1~7次、7次以上的用户人数
hit_count.sort_index(inplace=True)
hit_count_7 = hit_count.iloc[:7, :]
time = hit_count.iloc[7:, 0].sum()  # 统计点击次数7次以上的用户数
hit_count_7 = hit_count_7.append([{'用户数': time}], ignore_index=True)
hit_count_7.index = ['1', '2', '3', '4', '5', '6', '7', '7次以上']
hit_count_7['用户比例'] = hit_count_7['用户数'] / hit_count_7['用户数'].sum()
print(hit_count_7)

# 代码11-8

# 分析浏览一次的用户行为

engine = create_engine('mysql+pymysql://root:Zqy20020227_@127.0.0.1:3306/test?charset=utf8')
all_gzdata = pd.read_sql_table('all_gzdata', con=engine)  # 读取all_gzdata数据

# 对realIP进行统计
# 提取浏览1次网页的数据
real_count = pd.DataFrame(all_gzdata.groupby("realIP")["realIP"].count())
real_count.columns = ["count"]
real_count["real"] = real_count.index.tolist()
user_one = real_count[(real_count["count"] == 1)]  # 提取只登录一次的用户
print(user_one)
print(real_count)

# 通过realIP与原始数据合并
real_one = pd.merge(user_one, all_gzdata, left_on="realIP", right_on="realIP")

# 统计浏览一次的网页类型
URL_count = pd.DataFrame(real_one.groupby("fullURLId")["fullURLId"].count())
URL_count.columns = ["count"]
URL_count.sort_values(by='count', ascending=False, inplace=True)  # 降序排列
# 统计排名前4和其他的网页类型
URL_count_4 = URL_count.iloc[:4, :]
time = hit_count.iloc[4:, 0].sum()  # 统计其他的
URLindex = URL_count_4.index.values
URL_count_4 = URL_count_4.append([{'count': time}], ignore_index=True)
URL_count_4.index = [URLindex[0], URLindex[1], URLindex[2], URLindex[3],
                     '其他']
URL_count_4['比例'] = URL_count_4['count'] / URL_count_4['count'].sum()
print(URL_count_4)

# 代码11-9

# 在浏览1次的前提下, 得到的网页被浏览的总次数
fullURL_count = pd.DataFrame(real_one.groupby("fullURL")["fullURL"].count())
fullURL_count.columns = ["count"]
fullURL_count["fullURL"] = fullURL_count.index.tolist()
fullURL_count.sort_values(by='count', ascending=False, inplace=True)  # 降序排列


# 读取数据
con = pm.connect(host='127.0.0.1', user='root', passwd='Zqy20020227_', port=3306, db='test', charset='utf8')
data = pd.read_sql('select * from all_gzdata', con=con)
con.close()  # 关闭连接

# 取出107类型数据
index107 = [re.search('107', str(i)) != None for i in data.loc[:, 'fullURLId']]
data_107 = data.loc[index107, :]

# 在107类型中筛选出婚姻类数据
index = [re.search('hunyin', str(i)) != None for i in data_107.loc[:, 'fullURL']]
data_hunyin = data_107.loc[index, :]

# 提取所需字段(realIP、fullURL)
info = data_hunyin.loc[:, ['realIP', 'fullURL']]

# 去除网址中“？”及其后面内容
da = [re.sub('\?.*', '', str(i)) for i in info.loc[:, 'fullURL']]
info.loc[:, 'fullURL'] = da  # 将info中‘fullURL’那列换成da
# 去除无html网址
index = [re.search('\.html', str(i)) != None for i in info.loc[:, 'fullURL']]
index.count(True)  # True 或者 1 ， False 或者 0
info1 = info.loc[index, :]

# 代码11-11

# 找出翻页和非翻页网址
index = [re.search('/\d+_\d+\.html', i) != None for i in info1.loc[:, 'fullURL']]
index1 = [i == False for i in index]
info1_1 = info1.loc[index, :]  # 带翻页网址
info1_2 = info1.loc[index1, :]  # 无翻页网址
# 将翻页网址还原
da = [re.sub('_\d+\.html', '.html', str(i)) for i in info1_1.loc[:, 'fullURL']]
info1_1.loc[:, 'fullURL'] = da
# 翻页与非翻页网址合并
frames = [info1_1, info1_2]
info2 = pd.concat(frames)
# 或者
info2 = pd.concat([info1_1, info1_2], axis=0)  # 默认为0，即行合并
# 去重（realIP和fullURL两列相同）
info3 = info2.drop_duplicates()
# 将IP转换成字符型数据
info3.iloc[:, 0] = [str(index) for index in info3.iloc[:, 0]]
info3.iloc[:, 1] = [str(index) for index in info3.iloc[:, 1]]
len(info3)

# 代码11-12

# 筛选满足一定浏览次数的IP
IP_count = info3['realIP'].value_counts()
# 找出IP集合
IP = list(IP_count.index)
count = list(IP_count.values)
# 统计每个IP的浏览次数，并存放进IP_count数据框中,第一列为IP，第二列为浏览次数
IP_count = pd.DataFrame({'IP': IP, 'count': count})
# 3.3筛选出浏览网址在n次以上的IP集合
n = 2
index = IP_count.loc[:, 'count'] > n
IP_index = IP_count.loc[index, 'IP']

# 代码11-13

# 划分IP集合为训练集和测试集
index_tr = sample(range(0, len(IP_index)), int(len(IP_index) * 0.8))  # 或者np.random.sample
index_te = [i for i in range(0, len(IP_index)) if i not in index_tr]
IP_tr = IP_index[index_tr]
IP_te = IP_index[index_te]
# 将对应数据集划分为训练集和测试集
index_tr = [i in list(IP_tr) for i in info3.loc[:, 'realIP']]
index_te = [i in list(IP_te) for i in info3.loc[:, 'realIP']]
data_tr = info3.loc[index_tr, :]
data_te = info3.loc[index_te, :]
print(len(data_tr))
IP_tr = data_tr.iloc[:, 0]  # 训练集IP
url_tr = data_tr.iloc[:, 1]  # 训练集网址
IP_tr = list(set(IP_tr))  # 去重处理
url_tr = list(set(url_tr))  # 去重处理
print(len(url_tr))

# 利用训练集数据构建模型

UI_matrix_tr = pd.DataFrame(0, index=IP_tr, columns=url_tr)
# 求用户－物品矩阵
for i in data_tr.index:
    UI_matrix_tr.loc[data_tr.loc[i, 'realIP'], data_tr.loc[i, 'fullURL']] = 1
print(sum(UI_matrix_tr.sum(axis=1)))

# 求物品相似度矩阵（因计算量较大，需要耗费的时间较久）
Item_matrix_tr = pd.DataFrame(0, index=url_tr, columns=url_tr)
for i in Item_matrix_tr.index:
    for j in Item_matrix_tr.index:
        a = sum(UI_matrix_tr.loc[:, [i, j]].sum(axis=1) == 2)
        b = sum(UI_matrix_tr.loc[:, [i, j]].sum(axis=1) != 0)
        Item_matrix_tr.loc[i, j] = a / b

# 将物品相似度矩阵对角线处理为零
for i in Item_matrix_tr.index:
    Item_matrix_tr.loc[i, i] = 0

# 利用测试集数据对模型评价
IP_te = data_te.iloc[:, 0]
url_te = data_te.iloc[:, 1]
IP_te = list(set(IP_te))
url_te = list(set(url_te))
print(IP_te, url_te)
# 测试集数据用户物品矩阵
UI_matrix_te = pd.DataFrame(0, index=IP_te, columns=url_te)
for i in data_te.index:
    UI_matrix_te.loc[data_te.loc[i, 'realIP'], data_te.loc[i, 'fullURL']] = 1

# 对测试集IP进行推荐
Res = pd.DataFrame('NaN', index=data_te.index,
                   columns=['IP', '已浏览网址', '推荐网址', 'T/F'])
Res.loc[:, 'IP'] = list(data_te.iloc[:, 0])
Res.loc[:, '已浏览网址'] = list(data_te.iloc[:, 1])

# 开始推荐
for i in Res.index:
    if Res.loc[i, '已浏览网址'] in list(Item_matrix_tr.index):
        Res.loc[i, '推荐网址'] = Item_matrix_tr.loc[Res.loc[i, '已浏览网址'],
                                 :].argmax()
        if Res.loc[i, '推荐网址'] in url_te:
            Res.loc[i, 'T/F'] = UI_matrix_te.loc[Res.loc[i, 'IP'],
                                                 Res.loc[i, '推荐网址']] == 1
        else:
            Res.loc[i, 'T/F'] = False

# 保存推荐结果
Res.to_csv('./data_chapter15/Res.csv', index=False, encoding='utf8')

Res = pd.read_csv('./data_chapter15/Res.csv', keep_default_na=False, encoding='utf8')

# 计算推荐准确率
Pre = round(sum(Res.loc[:, 'T/F'] == 'True') / (len(Res.index) - sum(Res.loc[:, 'T/F'] == 'NaN')), 3)

print(Pre)

# 计算推荐召回率
Rec = round(sum(Res.loc[:, 'T/F'] == 'True') / (sum(Res.loc[:, 'T/F'] == 'True') + sum(Res.loc[:, 'T/F'] == 'NaN')), 3)

print(Rec)

# 计算F1指标
F1 = round(2 * Pre * Rec / (Pre + Rec), 3)
print(F1)

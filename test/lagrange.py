import pandas as pd  # 导入数据分析库Pandas
from scipy.interpolate import lagrange  # 导入拉格朗日插值函数

inputfile = './data/catering_sale.xls'  # 销量数据路径
outputfile = './data/sales.xls'  # 输出数据路径

data = pd.read_excel(inputfile)  # 读入数据
print(data)
print(data['销量'].values)
print(data.loc[:, '销量'])
for index, i in enumerate(data.loc[:, '销量']):
    if i < 400 or i > 5000:
        print(index)
        data.loc[index:index, '销量'] = None
print(data['销量'].values)


# data.loc[:, '销量'][(data['销量'] < 400) | (data['销量'] > 5000)] = None  # 过滤异常值，将其变为空值


# 自定义列向量插值函数
# s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
def ployinterp_column(s, n, k=5):
    if n < 5:
        y = s[list(range(0, n)) + list(range(n + 1, n + 1 + k))]
        y = y[y.notnull()]
    elif n + 1 + k > 201:
        y = s[list(range(n - k, n)) + list(range(n+1, 201))]  # 取数
        y = y[y.notnull()]  # 剔除空值
    else :
        y = s[list(range(n - k, n)) + list(range(n+1, n+1+k))]  # 取数
        y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果


# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:  # 如果为空即插值。
            data[i][j] = ployinterp_column(data[i], j)
print(data.loc[:, '销量'].values)
data.to_excel(outputfile)  # 输出结果，写入文件

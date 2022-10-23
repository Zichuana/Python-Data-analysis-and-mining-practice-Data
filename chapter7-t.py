import pandas as pd
import matplotlib.pyplot as plt
from mplfonts import use_font
use_font('Noto Sans CJK SC')

datafile = './data_chapter7/air_data.csv'  # 航空原始数据,第一行为属性标签

# 读取原始数据，指定UTF-8编码（需要用文本编辑器将数据装换为UTF-8编码）
data = pd.read_csv(datafile, encoding='utf-8')
# 乘机信息类别
lte = data['LAST_TO_END']
fc = data['FLIGHT_COUNT']
sks = data['SEG_KM_SUM']

# 绘制最后乘机至结束时长箱线图
fig = plt.figure(figsize=(5, 8))
plt.boxplot(lte,
            patch_artist=True,
            labels=['时长'],  # 设置x轴标题
            boxprops={'facecolor': 'lightblue'})  # 设置填充颜色
plt.title('会员最后乘机至结束时长分布箱线图')
# 显示y坐标轴的底线
plt.grid(axis='y')
plt.show()
plt.close

# 绘制客户飞行次数箱线图
fig = plt.figure(figsize=(5, 8))
plt.boxplot(fc,
            patch_artist=True,
            labels=['飞行次数'],  # 设置x轴标题
            boxprops={'facecolor': 'lightblue'})  # 设置填充颜色
plt.title('会员飞行次数分布箱线图')
# 显示y坐标轴的底线
plt.grid(axis='y')
plt.show()
plt.close

# 绘制客户总飞行公里数箱线图
fig = plt.figure(figsize=(5, 10))
plt.boxplot(sks,
            patch_artist=True,
            labels=['总飞行公里数'],  # 设置x轴标题
            boxprops={'facecolor': 'lightblue'})  # 设置填充颜色
plt.title('客户总飞行公里数箱线图')
# 显示y坐标轴的底线
plt.grid(axis='y')
plt.show()
plt.close

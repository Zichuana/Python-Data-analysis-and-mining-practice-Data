import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("./data/dish_sale.xls")
plt.figure(figsize=(8, 4))
plt.plot(data['月份'], data['A部门'], color='green', label='A部门', marker='o')
plt.plot(data['月份'], data['B部门'], color='red', label='B部门', marker='s')
plt.plot(data['月份'], data['C部门'],  color='skyblue', label='C部门', marker='x')
plt.plot(data['月份'], data['D部门'], color='orange', label='D部门', marker='p')
plt.plot(data['月份'], data['E部门'],  color='violet', label='E部门', marker='*')
plt.legend()  # 显示图例
plt.ylabel('销售额（万元）')

plt.savefig("./data/ABC三个部门的销售金额随时间的变化趋势图", dpi=300)
plt.show()
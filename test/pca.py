import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
y = data.target
x = data.data
print(x)
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
print(pca)
reduced_x = pca.fit_transform(x)  # 对样本进行降维
print(reduced_x)
# reduced_x = np.dot(reduced_x, pca.components_) + pca.mean_  # 还原数据

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
# print(reduced_x)
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

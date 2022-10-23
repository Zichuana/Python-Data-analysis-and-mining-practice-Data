import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV  # Lasso回归聚类分析的方法对数据进行离散化，并对相关代码进行分析，绘出结果。


# 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
data = [
    [0.067732, 3.176513], [0.427810, 3.816464], [0.995731, 4.550095], [
        0.738336, 4.256571], [0.981083, 4.560815],
    [0.526171, 3.929515], [0.378887, 3.526170], [0.033859, 3.156393], [
        0.132791, 3.110301], [0.138306, 3.149813],
    [0.247809, 3.476346], [0.648270, 4.119688], [0.731209, 4.282233], [
        0.236833, 3.486582], [0.969788, 4.655492],
    [0.607492, 3.965162], [0.358622, 3.514900], [0.147846, 3.125947], [
        0.637820, 4.094115], [0.230372, 3.476039],
    [0.070237, 3.210610], [0.067154, 3.190612], [0.925577, 4.631504], [
        0.717733, 4.295890], [0.015371, 3.085028],
    [0.335070, 3.448080], [0.040486, 3.167440], [
        0.212575, 3.364266], [0.617218, 3.993482], [0.541196, 3.891471]
]
# ========岭回归========
# 生成X和y矩阵
dataMat = np.array(data)
print(dataMat)
X = dataMat[:, 0:1]   # 变量x
y = dataMat[:, 1]  # 变量y

model = Ridge(alpha=0.5)
model = RidgeCV(alphas=[0.1, 1.0, 10.0])  # 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model.fit(X, y)   # 线性回归建模
print('岭回归系数矩阵:\n', model.coef_)
print('线性回归模型:\n', model)
# print('交叉验证最佳alpha值',model.alpha_)  # 只有在使用RidgeCV算法时才有效
# 使用模型预测
predicted_ridge = model.predict(X)
# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='*')
plt.plot(X, predicted_ridge, c='r')
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")
# 显示图形
plt.show()
# 生成X和y矩阵

# ========Lasso回归========
dataMat = np.array(data)
X = dataMat[:, 0:1]   # 变量x
y = dataMat[:, 1]  # 变量y

model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
# model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
# model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(X, y)   # 线性回归建模
print('lasso回归系数矩阵:\n', model.coef_)
print('线性回归模型:\n', model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
# 使用模型预测
predicted_lasso = model.predict(X)
# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='*')
plt.plot(X, predicted_lasso, c='r')
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")
# 显示图形
plt.show()


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X, y, marker='*')
plt.plot(X, predicted_ridge, c='r')
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")
plt.title('Ridge')
plt.subplot(1, 2, 2)
plt.scatter(X, y, marker='*')
plt.plot(X, predicted_lasso, c='r')
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")
plt.title('lasso')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X, y, marker='*')
plt.plot(X, predicted_ridge, color='green', marker='*', label='Ridge')
plt.plot(X, predicted_lasso, color='red', marker='*', label='lasso')
# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")
plt.show()

model_lasso_2 = Lasso(alpha=1)
model_lasso_2.fit(X, y)
predicted_lasso_2 = model_lasso_2.predict(X)

model_lasso_3 = Lasso(alpha=0.1)
model_lasso_3.fit(X, y)
predicted_lasso_3 = model_lasso_3.predict(X)

model_lasso_4 = Lasso(alpha=0.001)
model_lasso_4.fit(X, y)
predicted_lasso_4 = model_lasso_4.predict(X)

model_lasso_5 = Lasso(alpha=0.00000000001)
model_lasso_5.fit(X, y)
predicted_lasso_5 = model_lasso_5.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, marker='*', color='violet')
plt.plot(X, predicted_lasso_2, color='green', label='1')
plt.plot(X, predicted_lasso_3, color='red', label='0.1')
plt.plot(X, predicted_lasso, color='skyblue', label='0.01')
plt.plot(X, predicted_lasso_4, color='orange', label='0.001')
plt.plot(X, predicted_lasso_5, color='pink', label='0.00000000001')
# 绘制x轴和y轴坐标
plt.legend()  # 显示图例
plt.xlabel("x")
plt.ylabel("y")
plt.show()
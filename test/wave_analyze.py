from scipy.io import loadmat  # mat是Python专用格式，需要用loadmat读取它
import pywt  # 导入PyWavelets

# 小波变换特征提取
# 利用小波分析进行特征分析
# 参数初始化
inputfile = './data/leleccum.mat'  # 提取自Matlab的信号文件

mat = loadmat(inputfile)
signal = mat['leleccum'][0]
print(signal)

coeffs = pywt.wavedec(signal, 'bior3.7', level=5)

print(coeffs)
# 返回结果为level+1个数字，第一个数组为逼近系数数组，后面的依次是细节系数数组

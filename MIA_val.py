import numpy as np
import matplotlib.pyplot as plt

# 创建一个二维概率向量
probabilities = np.array([[0.1, 0.2, 0.3],
                          [0.25, 0.15, 0.3],
                          [0.05, 0.35, 0.2]])

# 绘制热图
plt.imshow(probabilities, cmap='Reds', interpolation='nearest')

# 添加数值标签
for i in range(probabilities.shape[0]):
    for j in range(probabilities.shape[1]):
        plt.text(j, i, f'{probabilities[i, j]:.2f}', ha='center', va='center')

# 设置刻度标签和标题
plt.xticks(np.arange(probabilities.shape[1]), np.arange(1, probabilities.shape[1] + 1))
plt.yticks(np.arange(probabilities.shape[0]), np.arange(1, probabilities.shape[0] + 1))
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Probability Visualization')

# 添加颜色条
plt.colorbar()

# 显示图形
plt.savefig("test.png")
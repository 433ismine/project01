import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载手部关键点数据
hand_data = np.load('hand_data.npy')

# 定义关键点连接
connections = [
    # 定义拇指、食指、中指、无名指、小指和手腕的连接
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (0, 21)
]

# 准备绘制
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制关键点
for keypoints in hand_data:
    x = [point[0] for point in keypoints]
    y = [point[1] for point in keypoints]
    z = [point[2] for point in keypoints]
    
    ax.scatter(x, y, z, c='r', marker='o')

    # 绘制连线
    for start, end in connections:
        if start < len(keypoints) and end < len(keypoints):
            ax.plot([keypoints[start][0], keypoints[end][0]],
                    [keypoints[start][1], keypoints[end][1]],
                    [keypoints[start][2], keypoints[end][2]], c='b')


ax.set_title('3D Hand Keypoints')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 显示图形
plt.show()
import numpy as np
import cv2
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

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

# 设置视频参数
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10
frame_size = (640, 480)
video_writer = cv2.VideoWriter('hand_tracking.mp4', fourcc, fps, frame_size)

# 绘制关键点和连线，并保存为视频
for keypoints in hand_data:
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        print(f"跳过第 {keypoints} 帧，因为它不符合预期的维度 (N, 3)")
        continue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    x = [point[0] for point in keypoints]
    y = [point[1] for point in keypoints]
    z = [point[2] for point in keypoints]
    
    ax.scatter(x, y, z, c='r', marker='o')
    
    for start, end in connections:
        if start < len(keypoints) and end < len(keypoints):
            ax.plot([x[start], x[end]], [y[start], y[end]], [z[start], z[end]], c='b')
    
    # 将matplotlib图转换为OpenCV图像
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    image = np.asarray(buffer)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    # 写入视频帧
    video_writer.write(image)
    
    # 关闭matplotlib图
    plt.close(fig)

# 释放资源
video_writer.release()
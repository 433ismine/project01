# # -*- coding:utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# def read_skeleton(file):
#     skeleton_sequence = {
#         'numFrame': 0,
#         'frameInfo': []
#     }
#     with open(file, 'r') as f:
#         for line in f:
#             line = line.strip()  # 去除行尾的空白字符
#             if line and not line.startswith('#'):  # 跳过空行和注释行
#                 values = line.split(',')  # 按逗号分隔
#                 frame, body, joint, x, y, z = map(float, values)  # 转换为浮点数
#                 if frame > skeleton_sequence['numFrame']:
#                     skeleton_sequence['numFrame'] = int(frame)  # 更新帧数
#                     # 为新的帧创建一个新的帧信息字典
#                     skeleton_sequence['frameInfo'].append({
#                         'frameNum': frame,
#                         'numBody': 1,
#                         'bodyInfo': [{
#                             'bodyID': body,
#                             'numJoint': 1,
#                             'jointInfo': [{
#                                 'x': x, 'y': y, 'z': z
#                             }]
#                         }]
#                     })
#                 else:
#                     # 找到正确的帧并添加关节数据
#                     for frame_info in skeleton_sequence['frameInfo']:
#                         if frame_info['frameNum'] == frame:
#                             for body_info in frame_info['bodyInfo']:
#                                 if body_info['bodyID'] == body:
#                                     body_info['jointInfo'].append({
#                                         'x': x, 'y': y, 'z': z
#                                     })
#                                     body_info['numJoint'] = len(body_info['jointInfo'])
#                                     break
#                             break
#     return skeleton_sequence
#
# ## 读取关节的x，y，z三个坐标
# def read_xyz(file, max_body=2, num_joint=21):
#     seq_info = read_skeleton(file)
#     data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
#     for n, f in enumerate(seq_info['frameInfo']):
#         frame_num = int(f['frameNum'])  # 获取帧编号
#         for m in range(max_body):
#             for j, v in enumerate(f['bodyInfo'][0]['jointInfo']):
#                 if j < num_joint:
#                     data[:, frame_num - 1, j, m] = [v['x'], v['y'], v['z']]
#     return data
#
# def Print2D(num_frame, point, arms, rightHand, leftHand, legs, body,waist):
#
#     # 求坐标最大值
#     xmax = np.max(point[0, :, :, :])
#     xmin = np.min(point[0, :, :, :])
#     ymax = np.max(point[1, :, :, :])
#     ymin = np.min(point[1, :, :, :])
#     zmax = np.max(point[2, :, :, :])
#     zmin = np.min(point[2, :, :, :])
#
#     n = 0     # 从第n帧开始展示
#     m = num_frame   # 到第m帧结束，n<m<row
#     plt.figure()
#     plt.ion()
#     for i in range(n, m):
#         plt.cla() # # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响
#
#         # 画出两个body所有关节
#         plt.scatter(point[0, i, :, :], point[1, i, :, :], c='red', s=40.0) # c: 颜色;  s: 大小
#
#         # 连接第一个body的关节，形成骨骼
#         plt.plot(point[0, i, arms, 0], point[1, i, arms, 0], c='green', lw=2.0)
#         plt.plot(point[0, i, rightHand, 0], point[1, i, rightHand, 0], c='green', lw=2.0) # c: 颜色;  lw: 线条宽度
#         plt.plot(point[0, i, leftHand, 0], point[1, i, leftHand, 0], c='green', lw=2.0)
#         plt.plot(point[0, i, legs, 0], point[1, i, legs, 0], c='green', lw=2.0)
#         plt.plot(point[0, i, body, 0], point[1, i, body, 0], c='green', lw=2.0)
#         plt.plot(point[0, i, waist, 0], waist[1, i, body, 0], c='green', lw=2.0)
#
#
#         # 连接第二个body的关节，形成骨骼
#         plt.plot(point[0, i, arms, 1], point[1, i, arms, 1], c='green', lw=2.0)
#         plt.plot(point[0, i, rightHand, 1], point[1, i, rightHand, 1], c='green', lw=2.0)
#         plt.plot(point[0, i, leftHand, 1], point[1, i, leftHand, 1], c='green', lw=2.0)
#         plt.plot(point[0, i, legs, 1], point[1, i, legs, 1], c='green', lw=2.0)
#         plt.plot(point[0, i, body, 1], point[1, i, body, 1], c='green', lw=2.0)
#         plt.plot(point[0, i, waist, 1], waist[1, i, body, 1], c='green', lw=2.0)
#
#         plt.text(xmax, ymax+0.2, 'frame: {}/{}'.format(i, num_frame-1)) # 文字说明
#         plt.xlim(xmin-0.5, xmax+0.5) # x坐标范围
#         plt.ylim(ymin-0.3, ymax+0.3) # y坐标范围
#         plt.pause(0.001) # 停顿延时
#
#     plt.ioff()
#     plt.show()
#
#
# def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body,waist):
#     id_color_mapping = {
#         0: 'red',  # 例如，关节 ID 0 对应红色
#         5: 'orange',  # 关节 ID 1 对应蓝色
#         9: 'orange',  # 关节 ID 2 对应绿色
#         13: 'orange',  # 关节 ID 3 对应橙色
#         17: 'orange',  # 其他关节可以继续添加
#
#     }
#
#     # 求坐标最大值
#     xmax = np.max(point[0, :, :, :])
#     xmin = np.min(point[0, :, :, :])
#     ymax = np.max(point[1, :, :, :])
#     ymin = np.min(point[1, :, :, :])
#     zmax = np.max(point[2, :, :, :])
#     zmin = np.min(point[2, :, :, :])
#
#     n = 0     # 从第n帧开始展示
#     m = num_frame   # 到第m帧结束，n<m<row
#     plt.figure()
#     plt.ion()
#     for i in range(n, m):
#         plt.cla() # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响
#
#         plot3D = plt.subplot(projection = '3d')
#         plot3D.view_init(120, 90) # 改变视角
#
#         Expan_Multiple = 1.5 # 坐标扩大倍数，绘图较美观
#
#         # 画出两个body所有关节
#         for j in range(point.shape[2]):  # 遍历所有关节
#             color = id_color_mapping.get(j, 'orange')  # 默认颜色为黑色
#             plot3D.scatter(point[0, i, :, :]*Expan_Multiple, point[1, i, :, :]*Expan_Multiple, point[2, i, :, :], c=color, s=40.0) # c: 颜色;  s: 大小
#
#         # 连接第一个body的关节，形成骨骼
#         plot3D.plot(point[0, i, arms, 0]*Expan_Multiple, point[1, i, arms, 0]*Expan_Multiple, point[2, i, arms, 0], c='green', lw=2.0)
#         plot3D.plot(point[0, i, rightHand, 0]*Expan_Multiple, point[1, i, rightHand, 0]*Expan_Multiple, point[2, i, rightHand, 0], c='green', lw=2.0) # c: 颜色;  lw: 线条宽度
#         plot3D.plot(point[0, i, leftHand, 0]*Expan_Multiple, point[1, i, leftHand, 0]*Expan_Multiple, point[2, i, leftHand, 0], c='green', lw=2.0)
#         plot3D.plot(point[0, i, legs, 0]*Expan_Multiple, point[1, i, legs, 0]*Expan_Multiple, point[2, i, legs, 0], c='green', lw=2.0)
#         plot3D.plot(point[0, i, body, 0]*Expan_Multiple, point[1, i, body, 0]*Expan_Multiple, point[2, i, body, 0], c='green', lw=2.0)
#         plot3D.plot(point[0, i, waist, 0]*Expan_Multiple, point[1, i, waist, 0]*Expan_Multiple, point[2, i, waist, 0], c='green', lw=2.0)
#
#         # 连接第二个body的关节，形成骨骼
#         plot3D.plot(point[0, i, arms, 1]*Expan_Multiple, point[1, i, arms, 1]*Expan_Multiple, point[2, i, arms, 1], c='green', lw=2.0)
#         plot3D.plot(point[0, i, rightHand, 1]*Expan_Multiple, point[1, i, rightHand, 1]*Expan_Multiple, point[2, i, rightHand, 1], c='green', lw=2.0)
#         plot3D.plot(point[0, i, leftHand, 1]*Expan_Multiple, point[1, i, leftHand, 1]*Expan_Multiple, point[2, i, leftHand, 1], c='green', lw=2.0)
#         plot3D.plot(point[0, i, legs, 1]*Expan_Multiple, point[1, i, legs, 1]*Expan_Multiple, point[2, i, legs, 1], c='green', lw=2.0)
#         plot3D.plot(point[0, i, body, 1]*Expan_Multiple, point[1, i, body, 1]*Expan_Multiple, point[2, i, body, 1], c='green', lw=2.0)
#         plot3D.plot(point[0, i, waist, 1]*Expan_Multiple, point[1, i, waist, 1]*Expan_Multiple, point[2, i, waist, 1], c='green', lw=2.0)
#
#
#         plot3D.text(xmax-0.3, ymax+1.1, zmax+0.3, 'frame: {}/{}'.format(i, num_frame-1)) # 文字说明
#         plot3D.set_xlim3d(xmin-0.5, xmax+0.5) # x坐标范围
#         plot3D.set_ylim3d(ymin-0.3, ymax+0.3) # y坐标范围
#         plot3D.set_zlim3d(zmin-0.3, zmax+0.3) # z坐标范围
#         plt.axis('off')
#         plt.pause(0.001) # 停顿延时
#
#
#
#     plt.ioff()
#     plt.show()
#
# ## main函数
# def main():
#     data_path = '../data/graph/front.skeleton'  # 新的.skeleton文件路径
#     point = read_xyz(data_path)   # 读取 x,y,z三个坐标
#     print('Read Data Done!')  # 数据读取完毕
#
#     num_frame = point.shape[1]  # 帧数
#     print(point.shape)  # 坐标数(3) × 帧数 × 关节数(21) × max_body(2)
#
#     # 相邻关节标号
#     arms = [0,1,2,3,4]
#     rightHand = [0,5,6,7,8]
#     leftHand = [0,9,10,11,12]
#     legs = [0,13,14,15,16]
#     body = [0,17,18,19,20]
#     waist = [0,20]
#     Print3D(num_frame, point, arms, rightHand, leftHand, legs, body,waist)
#    #Print2D(num_frame, point, arms, rightHand, leftHand, legs, body,waist)
#
# main()

# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_skeleton(file):
    skeleton_sequence = {
        'numFrame': 0,
        'frameInfo': []
    }
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()  # 去除行尾的空白字符
            if line and not line.startswith('#'):  # 跳过空行和注释行
                values = line.split(',')  # 按逗号分隔
                frame, body, joint, x, y, z = map(float, values)  # 转换为浮点数
                if frame > skeleton_sequence['numFrame']:
                    skeleton_sequence['numFrame'] = int(frame)  # 更新帧数
                    # 为新的帧创建一个新的帧信息字典
                    skeleton_sequence['frameInfo'].append({
                        'frameNum': frame,
                        'numBody': 1,
                        'bodyInfo': [{
                            'bodyID': body,
                            'numJoint': 1,
                            'jointInfo': [{
                                'x': x, 'y': y, 'z': z
                            }]
                        }]
                    })
                else:
                    # 找到正确的帧并添加关节数据
                    for frame_info in skeleton_sequence['frameInfo']:
                        if frame_info['frameNum'] == frame:
                            for body_info in frame_info['bodyInfo']:
                                if body_info['bodyID'] == body:
                                    body_info['jointInfo'].append({
                                        'x': x, 'y': y, 'z': z
                                    })
                                    body_info['numJoint'] = len(body_info['jointInfo'])
                                    break
                            break
    return skeleton_sequence

## 读取关节的x，y，z三个坐标
def read_xyz(file, max_body=2, num_joint=21):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        frame_num = int(f['frameNum'])  # 获取帧编号
        for m in range(max_body):
            for j, v in enumerate(f['bodyInfo'][0]['jointInfo']):
                if j < num_joint:
                    data[:, frame_num - 1, j, m] = [v['x'], v['y'], v['z']]
    return data

def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body, waist):
    # 定义关节 ID 对应的颜色
    id_color_mapping = {
        0: 'red',
        5: 'red',
        9: 'red',
        13: 'red',
        17: 'red',

    }

    # 求坐标最大值
    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    n = 0     # 从第n帧开始展示
    m = num_frame   # 到第m帧结束，n<m<row
    plt.figure()
    plt.ion()
    plt.axis('off')
    for i in range(n, m):
        plt.cla() # Clear axis, 即清除当前图形中的当前活动轴, 其他轴不受影响

        plot3D = plt.subplot(projection='3d')
        plot3D.view_init(120, 90) # 改变视角

        Expan_Multiple = 1.2 # 坐标扩大倍数，绘图较美观

        # 画出所有关节，使用 ID 对应的颜色
        for j in range(point.shape[2]):  # 遍历所有关节
            color = id_color_mapping.get(j, 'red')  # 默认颜色为黑色
            plot3D.scatter(point[0, i, j, :]*Expan_Multiple,
                           point[1, i, j, :]*Expan_Multiple,
                           point[2, i, j, :],
                           c=color, s=40.0)  # 根据 ID 获取颜色

        # 连接第一个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 0]*Expan_Multiple, point[1, i, arms, 0]*Expan_Multiple, point[2, i, arms, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0]*Expan_Multiple, point[1, i, rightHand, 0]*Expan_Multiple, point[2, i, rightHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 0]*Expan_Multiple, point[1, i, leftHand, 0]*Expan_Multiple, point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0]*Expan_Multiple, point[1, i, legs, 0]*Expan_Multiple, point[2, i, legs, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0]*Expan_Multiple, point[1, i, body, 0]*Expan_Multiple, point[2, i, body, 0], c='green', lw=2.0)
        # plot3D.plot(point[0, i, waist, 0]*Expan_Multiple, point[1, i, waist, 0]*Expan_Multiple, point[2, i, waist, 0], c='green', lw=2.0)

        # 连接第二个body的关节，形成骨骼
        plot3D.plot(point[0, i, arms, 1]*Expan_Multiple, point[1, i, arms, 1]*Expan_Multiple, point[2, i, arms, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 1]*Expan_Multiple, point[1, i, rightHand, 1]*Expan_Multiple, point[2, i, rightHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 1]*Expan_Multiple, point[1, i, leftHand, 1]*Expan_Multiple, point[2, i, leftHand, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 1]*Expan_Multiple, point[1, i, legs, 1]*Expan_Multiple, point[2, i, legs, 1], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 1]*Expan_Multiple, point[1, i, body, 1]*Expan_Multiple, point[2, i, body, 1], c='green', lw=2.0)
        # plot3D.plot(point[0, i, waist, 1]*Expan_Multiple, point[1, i, waist, 1]*Expan_Multiple, point[2, i, waist, 1], c='green', lw=2.0)

        plot3D.text(xmax-0.3, ymax+1.1, zmax+0.3, 'frame: {}/{}'.format(i, num_frame-1)) # 文字说明
        plot3D.set_xlim3d(xmin-0.5, xmax+0.5) # x坐标范围
        plot3D.set_ylim3d(ymin-0.3, ymax+0.3) # y坐标范围
        plot3D.set_zlim3d(zmin-0.3, zmax+0.3) # z坐标范围
        # plt.axis('off')
        plt.pause(0.001) # 停顿延时

    plt.ioff()
    plt.show()

## main函数
def main():
    data_path = './data/1/hand_dataA011.skeleton'
    point = read_xyz(data_path)
    print('Read Data Done!')

    num_frame = point.shape[1]
    print(point.shape)

    # 相邻关节标号
    arms = [0,1,2,3,4]
    rightHand = [0,5,6,7,8]
    leftHand = [0,9,10,11,12]
    legs = [0,13,14,15,16]
    body = [0,17,18,19,20]
    waist = [0,20]

    Print3D(num_frame, point, arms, rightHand, leftHand, legs, body, waist)

main()
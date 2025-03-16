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
            line = line.strip()
            if line and not line.startswith('#'):
                values = line.split(',')
                frame, body, joint, x, y, z = map(float, values)
                if frame > skeleton_sequence['numFrame']:
                    skeleton_sequence['numFrame'] = int(frame)
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

def read_xyz(file, max_body=2, num_joint=21):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        frame_num = int(f['frameNum'])
        for m in range(max_body):
            for j, v in enumerate(f['bodyInfo'][0]['jointInfo']):
                if j < num_joint:
                    data[:, frame_num - 1, j, m] = [v['x'], v['y'], v['z']]
    return data

def Print3D(num_frame, point, arms, rightHand, leftHand, legs, body, waist, track_joints=[0], track_length=30):
    id_color_mapping = {
        0: 'red',
        5: 'red',
        9: 'red',
        13: 'red',
        17: 'red',
    }

    trajectory = {
        j: {
            'x': [],
            'y': [],
            'z': []
        } for j in track_joints
    }

    xmax = np.max(point[0, :, :, :])
    xmin = np.min(point[0, :, :, :])
    ymax = np.max(point[1, :, :, :])
    ymin = np.min(point[1, :, :, :])
    zmax = np.max(point[2, :, :, :])
    zmin = np.min(point[2, :, :, :])

    plt.figure()
    plt.ion()
    for i in range(num_frame):
        plt.cla()
        plot3D = plt.subplot(projection='3d')
        plot3D.view_init(120, 90)
        Expan_Multiple = 1.2

        # 绘制当前帧关节点
        for j in range(point.shape[2]):
            color = id_color_mapping.get(j, 'red')
            plot3D.scatter(
                point[0, i, j, :] * Expan_Multiple,
                point[1, i, j, :] * Expan_Multiple,
                point[2, i, j, :],
                c=color, s=40.0, alpha=1.0
            )

        # 更新并绘制轨迹
        for joint_id in track_joints:
            x = point[0, i, joint_id, 0] * Expan_Multiple
            y = point[1, i, joint_id, 0] * Expan_Multiple
            z = point[2, i, joint_id, 0]

            trajectory[joint_id]['x'].append(x)
            trajectory[joint_id]['y'].append(y)
            trajectory[joint_id]['z'].append(z)

            if len(trajectory[joint_id]['x']) > track_length:
                trajectory[joint_id]['x'].pop(0)
                trajectory[joint_id]['y'].pop(0)
                trajectory[joint_id]['z'].pop(0)

            n_points = len(trajectory[joint_id]['x'])
            if n_points == 0:
                continue

            # 分离RGB和Alpha
            colors = plt.cm.viridis(np.linspace(0, 1, n_points))[:, :3]  # 仅取RGB
            alphas = np.linspace(0.3, 1.0, n_points)

            plot3D.scatter(
                trajectory[joint_id]['x'],
                trajectory[joint_id]['y'],
                trajectory[joint_id]['z'],
                c=colors,
                alpha=alphas,
                s=20,
                marker='o',
                edgecolors='none'
            )

        # 绘制骨骼连接
        plot3D.plot(point[0, i, arms, 0] * Expan_Multiple, point[1, i, arms, 0] * Expan_Multiple, point[2, i, arms, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, rightHand, 0] * Expan_Multiple, point[1, i, rightHand, 0] * Expan_Multiple, point[2, i, rightHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, leftHand, 0] * Expan_Multiple, point[1, i, leftHand, 0] * Expan_Multiple, point[2, i, leftHand, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, legs, 0] * Expan_Multiple, point[1, i, legs, 0] * Expan_Multiple, point[2, i, legs, 0], c='green', lw=2.0)
        plot3D.plot(point[0, i, body, 0] * Expan_Multiple, point[1, i, body, 0] * Expan_Multiple, point[2, i, body, 0], c='green', lw=2.0)

        plot3D.text(xmax - 0.3, ymax + 1.1, zmax + 0.3, f'Frame: {i}/{num_frame-1}\nTracked Joints: {track_joints}')
        plot3D.set_xlim3d(xmin - 0.5, xmax + 0.5)
        plot3D.set_ylim3d(ymin - 0.3, ymax + 0.3)
        plot3D.set_zlim3d(zmin - 0.3, zmax + 0.3)

    plt.ioff()
    plt.show()
def main():
    data_path = '../data/graph/4.skeleton'
    point = read_xyz(data_path)
    print('Read Data Done!')
    num_frame = point.shape[1]

    arms = [0,1,2,3,4]
    rightHand = [0,5,6,7,8]
    leftHand = [0,9,10,11,12]
    legs = [0,13,14,15,16]
    body = [0,17,18,19,20]
    waist = [0,20]

    Print3D(
        num_frame, point, arms, rightHand, leftHand,
        legs, body, waist,
        track_joints=[0, 5, 9],
        track_length=50
    )

main()
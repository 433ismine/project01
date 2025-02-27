import os
import numpy as np


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
                frame_index = int(frame)

                # 确保 frameInfo 列表有足够的空间
                while len(skeleton_sequence['frameInfo']) <= frame_index:
                    skeleton_sequence['frameInfo'].append({
                        'frameNum': frame_index,
                        'bodyInfo': []
                    })

                # 更新帧数
                skeleton_sequence['numFrame'] = max(skeleton_sequence['numFrame'], frame_index)

                # 检查是否已有该身体信息
                body_found = False
                for body_info in skeleton_sequence['frameInfo'][frame_index]['bodyInfo']:
                    if body_info['bodyID'] == body:
                        body_info['jointInfo'].append({'x': x, 'y': y, 'z': z})
                        body_info['numJoint'] += 1
                        body_found = True
                        break

                if not body_found:
                    skeleton_sequence['frameInfo'][frame_index]['bodyInfo'].append({
                        'bodyID': body,
                        'numJoint': 1,
                        'jointInfo': [{'x': x, 'y': y, 'z': z}]
                    })

    return skeleton_sequence


# def extract_features(skeleton_data):
#     """提取特征，例如关节位置"""
#     features = []
#     for frame_info in skeleton_data['frameInfo']:
#         for body_info in frame_info['bodyInfo']:
#             # 提取关节位置，确保转换为 NumPy 数组
#             joint_positions = np.array([list(joint.values()) for joint in body_info['jointInfo']])
#
#             if joint_positions.size > 0:
#                 mean_position = np.mean(joint_positions, axis=0)  # 计算平均位置
#                 features.append(mean_position)
#             else:
#                 print(
#                     f"Warning: No joint positions found for body ID {body_info['bodyID']} in frame {frame_info['frameNum']}.")
#
#     return np.array(features)
#
#
# # 指定包含 .skeleton 文件的文件夹路径
# folder_path = '../../data/1'
# skeleton_files = [f for f in os.listdir(folder_path) if f.endswith('.skeleton')]
#
# all_features = []
#
# for file in skeleton_files:
#     file_path = os.path.join(folder_path, file)
#     skeleton_data = read_skeleton(file_path)
#     features = extract_features(skeleton_data)
#     all_features.append(features)
#
# # 计算所有样本的特征中心
# all_features = np.concatenate(all_features, axis=0)
# feature_center = np.mean(all_features, axis=0)
#
# # 保存特征中心
# np.save('../../data/111/0225+352006.npy', feature_center)

def extract_features(skeleton_data):
    features = []
    for frame_info in skeleton_data['frameInfo']:
        for body_info in frame_info['bodyInfo']:
            joint_positions = np.array([list(joint.values()) for joint in body_info['jointInfo']])
            if joint_positions.size > 0:
                mean_position = np.mean(joint_positions, axis=0)  # 计算平均位置
                features.append(mean_position)
    return np.array(features)

# 指定包含 .skeleton 文件的文件夹路径
folder_path = '../../data/1'
skeleton_files = [f for f in os.listdir(folder_path) if f.endswith('.skeleton')]

all_features = []

for file in skeleton_files:
    file_path = os.path.join(folder_path, file)
    skeleton_data = read_skeleton(file_path)
    features = extract_features(skeleton_data)
    all_features.append(features)

# 计算所有样本的特征中心
all_features = np.concatenate(all_features, axis=0)
feature_center = np.mean(all_features, axis=0)

# 假设每个特征都对应一个特定的关节
# 重新塑形为模型特征格式，假设 C=3, T=1, V=21 (21个关节)
num_samples = all_features.shape[0]
C = 3  # 特征维度
T = 1  # 假设只有一个时间步
V = 21  # 假设有21个关节

# 这里需要确保 features 的数量与 V 一致
if num_samples % V != 0:
    raise ValueError("提取的特征数量必须是关节数的倍数。")

all_features_reshaped = all_features.reshape(-1, V, C)  # 变为 (num_samples // V, V, C)
all_features_reshaped = all_features_reshaped.transpose(0, 2, 1)  # 变为 (num_samples // V, C, V)

# 保存特征中心
np.save('../../data/111/0225+352006.npy', feature_center)
print("done")

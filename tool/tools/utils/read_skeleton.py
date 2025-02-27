import os
import sys
import pickle
import argparse
import numpy as np
from numpy.lib.format import open_memmap
import random



random.seed(42)

max_body = 1
num_joint = 21
max_frame = 100 
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

def gendata(data_path, out_path, ignored_sample_path=None, train_ratio=0.8):
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    
    # for filename in os.listdir(data_path):
    #     if filename in ignored_samples:
    #         continue
    #     action_class = int(filename[filename.find('A') + 1:filename.find('A') + 2])
    #     sample_name.append(filename)
    #     sample_label.append(action_class - 1)
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
           continue
    # 提取动作类别，假设动作类别是两位数
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 3])
        print(f"Filename: {filename}, Action Class: {action_class}")
        sample_name.append(filename)
        sample_label.append(action_class - 1)

    # 打乱样本
    indices = list(range(len(sample_name)))
    random.shuffle(indices)
    sample_name = [sample_name[i] for i in indices]
    sample_label = [sample_label[i] for i in indices]

    # 分割训练集和验证集
    train_size = int(len(sample_name) * train_ratio)
    train_name = sample_name[:train_size]
    train_label = sample_label[:train_size]
    val_name = sample_name[train_size:]
    val_label = sample_label[train_size:]

    # 保存训练集标签和数据
    with open('{}/train_label.pkl'.format(out_path), 'wb') as f:
        pickle.dump((train_name, train_label), f)
    
    fp_train = open_memmap('{}/train_data.npy'.format(out_path), dtype='float32', mode='w+', shape=(len(train_label),3, max_frame, num_joint,2))
    for i, s in enumerate(train_name):
        print_toolbar(i * 1.0 / len(train_label), '({:>5}/{:<5}) Processing train data: '.format(i + 1, len(train_label)))
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint, max_frame=max_frame)
        fp_train[i, :, :, :] = data
    end_toolbar()

    # 保存验证集标签和数据
    with open('{}/val_label.pkl'.format(out_path), 'wb') as f:
        pickle.dump((val_name, val_label), f)
    
    fp_val = open_memmap('{}/val_data.npy'.format(out_path), dtype='float32', mode='w+', shape=(len(val_label),3, max_frame, num_joint,2))
    for i, s in enumerate(val_name):
        print_toolbar(i * 1.0 / len(val_label), '({:>5}/{:<5}) Processing val data: '.format(i + 1, len(val_label)))
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint, max_frame=max_frame)

        fp_val[i, :, :, ] = data
    end_toolbar()


def read_xyz(file, max_body=1, num_joint=21, max_frame=300):
    seq_info = read_skeleton(file)  # Assuming you have a read_skeleton function
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))

    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]

    # Handle frame length
    if seq_info['numFrame'] < max_frame:
        # Pad along the second dimension (frame dimension)
        data = np.pad(data, ((0, 0), (0, max_frame - seq_info['numFrame']), (0, 0), (0, 0)), 'constant')
    else:
        data = data[:, :max_frame, :, :]

    return data


# def read_skeleton(file):
#     with open(file, 'r') as f:
#         lines = f.readlines()
#         numFrame = int(lines[0].strip())
#         frameInfo = []
#         for t in range(numFrame):
#             frame_info = {}
#             numBody = int(lines[1 + 3 * t].strip())
#             bodyInfo = []
#             for m in range(numBody):
#                 body_info_key = [
#                     'bodyID'
#                 ]
#                 body_info = {
#                     k: float(v)
#                     for k, v in zip(body_info_key, lines[2 + 3 * t].split())
#                 }
#                 numJoint = int(lines[3 + 3 * t].split()[0])
#                 jointInfo = []
#                 for v in range(numJoint):
#                     joint_info_key = [
#                         'x', 'y', 'z'
#                     ]
#                     joint_info = {
#                         k: float(v)
#                         for k, v in zip(joint_info_key, lines[4 + 3 * t + v].split())
#                     }
#                     jointInfo.append(joint_info)
#                 bodyInfo.append(body_info)
#             frame_info['bodyInfo'] = bodyInfo
#             frameInfo.append(frame_info)
#     return frameInfo
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


def add_noise(joint):
    noise = np.random.normal(0, 0.01, 3)  # 添加小噪声
    joint['x'] += noise[0]
    joint['y'] += noise[1]
    joint['z'] += noise[2]


def translate(joint, tx, ty, tz):
    joint['x'] += tx
    joint['y'] += ty
    joint['z'] += tz


def rotate(joint, angle):
    rad = np.radians(angle)
    cos = np.cos(rad)
    sin = np.sin(rad)
    x, y = joint['x'], joint['y']
    joint['x'] = cos * x - sin * y
    joint['y'] = sin * x + cos * y


def data_augmentation(skeleton_sequence):
    for frame_info in skeleton_sequence['frameInfo']:
        for body_info in frame_info['bodyInfo']:
            for joint in body_info['jointInfo']:
                # 添加噪声
                add_noise(joint)

                # 随机平移
                tx, ty, tz = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
                translate(joint, tx, ty, tz)

                # 随机旋转
                if random.random() < 0.5:  # 50%的概率旋转
                    rotate(joint, random.uniform(-10, 10))

    return skeleton_sequence
# 第一行： 表示整个数据集中包含的帧数（frame）。
# 后续每三行表示一帧的数据：
# 第一行： 表示该帧中包含的身体数量（body）。
# 第二行： 表示每个身体的 ID。   
# 第三行： 表示该身体包含的关节数量。
# 接下来的几行： 每行表示一个关节的坐标（x, y, z）。
# 示例数据：

# 10  # 总共10帧
# 3   # 第1帧有3个身体
# 1  # 第一个身体的ID
# 15  # 第一个身体有15个关节
# 0.1 0.2 0.3  # 第一个身体的第一个关节坐标
# ...
# 2   # 第二帧有2个身体
# ...
# 数据类型：
# 帧数、身体数量、关节数量： 整数
# 身体 ID 和关节坐标： 浮点数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Converter.')
    parser.add_argument('--data_path', default='../../../data/3')
    parser.add_argument('--ignored_sample_path', default='none')
    parser.add_argument('--out_folder', default='../../../data/data_test')
    arg = parser.parse_args()
    arg.ignored_sample_path = None
    gendata(arg.data_path, arg.out_folder, arg.ignored_sample_path)


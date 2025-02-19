#单个文件
# import numpy as np

# def read_skeleton_file(skeleton_file):
#     data = []
#     with open(skeleton_file, 'r') as file:
#         for line in file:
#             frame_id, body_id, joint_id, x, y, z = line.strip().split(',')
#             data.append([int(frame_id), int(body_id), int(joint_id), float(x), float(y), float(z)])
#     return data

# def organize_data(data):
#     num_frames = max(item[0] for item in data) + 1
#     num_bodies = max(item[1] for item in data) + 1
#     num_joints = 21
#     joint_dim = 3  
#     dataset = np.zeros((num_frames, num_bodies, num_joints, joint_dim), dtype=np.float32)
#     for frame_id, body_id, joint_id, x, y, z in data:
#         dataset[frame_id, body_id, joint_id] = [x, y, z]

#     return dataset
# def save_as_npy(data, output_file):
#     np.save(output_file, data)

# def main():
#     skeleton_file = '../hand_data.skeleton'
#     npy_file = '../hand_data_test.npy'

#     data = read_skeleton_file(skeleton_file)
#     dataset = organize_data(data)
#     save_as_npy(dataset, npy_file)
#     print(f"Data saved to {npy_file}")

# if __name__ == "__main__":
#     main()

#多个文件
import numpy as np
import os

def read_and_organize_skeleton_files(skeleton_dir, num_bodies, num_joints):
    joint_dim = 3  # x, y, z
    all_data = []

    # 遍历目录中的所有.skeleton文件
    for filename in os.listdir(skeleton_dir):
        if filename.endswith(".skeleton"):
            file_path = os.path.join(skeleton_dir, filename)
            with open(file_path, 'r') as file:
                data = []
                for line in file:
                    frame_id, body_id, joint_id, x, y, z = line.strip().split(',')
                    data.append([int(frame_id), int(body_id), int(joint_id), float(x), float(y), float(z)])
                all_data.append(data)

    # 找到所有文件中的最大帧数
    max_frames = max(max(item[0] for item in frame_data) + 1 for frame_data in all_data)

    # 初始化数据数组
    dataset = np.zeros((len(all_data), max_frames, num_bodies, num_joints, joint_dim), dtype=np.float32)

    # 填充数据数组
    for i, frame_data in enumerate(all_data):
        for frame_data_point in frame_data:
            frame_id, body_id, joint_id, x, y, z = frame_data_point
            # 确保body_id在正确的范围内
            if body_id < num_bodies:
                dataset[i, frame_id, body_id, joint_id, :] = [x, y, z]

    return dataset
def save_as_npy(data, output_file):
    np.save(output_file, data)

def main():
    skeleton_dir = '../data_skeleton' 
    output_file = 'data_skeleton.npy'  

    num_bodies = 1 
    num_joints = 21  

    dataset = read_and_organize_skeleton_files(skeleton_dir, num_bodies, num_joints)
    save_as_npy(dataset, output_file)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
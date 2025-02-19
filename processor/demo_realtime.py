
#!/usr/bin/env python
#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time
import numpy as np
import torch
import skvideo.io
from .io import IO
import tools
import tools.utils as utils
import cv2
import mediapipe as mp
from pyrealsense2 import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.hands
pose = mp_pose.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 初始化 RealSense 摄像头
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


fig = plt.figure()
plot3D = fig.add_subplot(111, projection='3d')
plt.ion()  

def display_3d_coordinates(multi_poseshow):
    # 定义各个部分的关键点索引
    arms = [0, 1, 2, 3, 4]
    rightHand = [0, 5, 6, 7, 8]
    leftHand = [0, 9, 10, 11, 12]
    legs = [0, 13, 14, 15, 16]
    body = [0, 17, 18, 19, 20]
    waist = [0, 20]  

    plot3D.cla()
    Expan_Multiple = 1

    if len(multi_poseshow) > 0:  # 检查是否有手部数据
        for hand_data in multi_poseshow:
            hand_id = hand_data['id']  # 获取手的 ID
            hand = hand_data['landmarks']  # 获取手的关键点
            
            # 绘制每个关节
            plot3D.scatter(hand[:, 0], hand[:, 1], hand[:, 2], c='red', s=40)

            # 绘制连接线，确保连接线的顺序和逻辑正确
            # 连接手臂
            plot3D.plot(hand[arms, 0] * Expan_Multiple, hand[arms, 1] * Expan_Multiple, hand[arms, 2], c='green', lw=2.0)

            # 连接右手
            plot3D.plot(hand[rightHand, 0] * Expan_Multiple, hand[rightHand, 1] * Expan_Multiple, hand[rightHand, 2], c='green', lw=2.0)

            # 连接左手
            plot3D.plot(hand[leftHand, 0] * Expan_Multiple, hand[leftHand, 1] * Expan_Multiple, hand[leftHand, 2], c='green', lw=2.0)

            # 连接腿部
            plot3D.plot(hand[legs, 0] * Expan_Multiple, hand[legs, 1] * Expan_Multiple, hand[legs, 2], c='green', lw=2.0)

            # 连接身体
            plot3D.plot(hand[body, 0] * Expan_Multiple, hand[body, 1] * Expan_Multiple, hand[body, 2], c='green', lw=2.0)

            # 连接腰部
            if hand.shape[0] > 21:  
                plot3D.plot(hand[waist, 0] * Expan_Multiple, hand[waist, 1] * Expan_Multiple, hand[waist, 2], c='green', lw=2.0)

            # 标注手的 ID
            plot3D.text(hand[0, 0], hand[0, 1], hand[0, 2], f'ID: {hand_id}', color='blue')

    plot3D.set_xlim([-1, 1])
    plot3D.set_ylim([-1, 1])
    plot3D.set_zlim([0, 1])
    plt.draw()
    plt.pause(0.001)

class DemoRealtime(IO):
    def start(self):
        # 加载动作分类标签
        label_name_path = './resource/hand_test/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # 初始化模型和姿态跟踪器
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        # 开始实时识别
        start_time = time.time()
        frame_index = 0
        while True:
            tic = time.time()

            # 获取 RealSense 摄像头的图像
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            orig_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 使用 MediaPipe 进行姿态估计
            results = pose.process(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            multi_poseshow = []
            if results.multi_hand_landmarks:
                multi_pose = []
                for hand_id,hand_landmarks in enumerate(results.multi_hand_landmarks):  # 遍历每个手部
                    handshow = [] 
                    for id,landmark in enumerate(hand_landmarks.landmark):  # 遍历每个手部的 landmark
                    # 将 MediaPipe 的归一化坐标转换为像素坐标
                        x = int(landmark.x * orig_image.shape[1])
                        y = int(landmark.y * orig_image.shape[0])

                    # 确保坐标在深度图的有效范围内
                        x = max(0, min(x, depth_image.shape[1] - 1))
                        y = max(0, min(y, depth_image.shape[0] - 1))

                    # 获取深度信息
                        z = depth_image[y, x] / 1000.0  # 将深度从毫米转换为米
                        multi_pose.append([x, y, z])

                        handshow.append([x / orig_image.shape[1], y / orig_image.shape[0], z])  # 归一化坐标
                    if len(handshow) == 21:  # 假设每只手有 21 个关键点
                        handshow=np.array(handshow)
                        multi_poseshow.append({'id': id, 'landmarks': handshow}) 
                    else:
                        print(f"Warning: Expected 21 landmarks, but got {len(handshow)} for one hand.")

                if len(multi_pose) > 0:
                   multi_pose = np.array(multi_pose)  # Shape: (num_hands, 21, 3)
                   multi_pose = multi_pose.reshape(1, -1, 3)  # Convert to shape (1, num_joints, 3)
            else:
                multi_pose = np.zeros((0, 21, 3))  # 如果未检测到姿态，返回空数组
                multi_poseshow = np.zeros((0, 21, 3))

# 转换为适合 3D 显示的格式
            if multi_poseshow:
                multi_poseshow = np.array(multi_poseshow)  # 将包含相同形状元素的列表转换为数组

    # 调用 3D 显示函数
                display_3d_coordinates(multi_poseshow)

            # 姿态数据归一化
            if len(multi_pose) > 0:
                multi_pose[:, :, 0] = multi_pose[:, :, 0] / orig_image.shape[1]
                multi_pose[:, :, 1] = multi_pose[:, :, 1] / orig_image.shape[0]
                multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
                multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
                multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0
            
            # 更新姿态跟踪器
            frame_index = int((time.time() - start_time) * self.arg.fps)
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()

            # 如果有有效的姿态序列，进行模型预测
            if data_numpy is not None:
                data = torch.from_numpy(data_numpy)
                data = data.unsqueeze(0)
                data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

                # 模型预测
                voting_label_name, video_label_name, output, intensity = self.predict(data)

                # 可视化
                app_fps = 1 / (time.time() - tic)
                image = self.render(data_numpy, voting_label_name, video_label_name, intensity, orig_image, app_fps)
                cv2.imshow("ST-GCN", image)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict(self, data):
        # 模型前向传播
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()
        print("Raw output:", output)
        print("Output shape:", output.shape)

        # 获取分类结果
        voting_label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print("voting_label:", voting_label)
        print("Number of labels:", len(self.label_name))
        voting_label_name = self.label_name[voting_label]

        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l] for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = []
        for t in range(num_frame):
            frame_label_name = []
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)

        return voting_label_name, video_label_name, output, intensity

    def render(self, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        # 可视化函数
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps
        )
        image = next(images)
        image = image.astype(np.uint8)
        return image

    
    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network'
        )
        parser.add_argument('--model_input_frame', default=128, type=int)
        parser.add_argument('--model_fps', default=30, type=int)
        parser.add_argument('--fps', default=30, type=int)
        parser.add_argument('--height', default=1080, type=int, help='height of frame in the output video.')
        parser.set_defaults(config='./config/st_gcn/kinetics-skeleton/demo_realtime.yaml')
        parser.set_defaults(print_log=False)
        return parser
    


class naive_pose_tracker():
    """ 简单的姿态跟踪器 """
    def __init__(self, data_frame=128, num_joint=21, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = []

    def update(self, multi_pose, current_frame):
        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:
            if p.shape[0] != self.num_joint or p.shape[1] != 3:
                print(f"Warning: Unexpected pose shape {p.shape}, expected ({self.num_joint}, 3).")
                continue
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)
            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    def cat_pose(self, trace, pose, pad, pad_mode):
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate((trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]
        if last_pose_xy.shape != curr_pose_xy.shape:
            raise ValueError(f"Shape mismatch: last_pose_xy {last_pose_xy.shape}, curr_pose_xy {curr_pose_xy.shape}")
        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
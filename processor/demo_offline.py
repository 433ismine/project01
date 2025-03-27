#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import torch
import cv2
from .io import IO
import mediapipe as mp
import tools.utils as utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

class naive_pose_tracker():
    def __init__(self, data_frame=128, num_joint=21, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = []

    def update(self, multi_pose, current_frame):
        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) == 0:
            return

        if len(multi_pose.shape) == 2: # Single person
            multi_pose = np.expand_dims(multi_pose, axis=0)

        for p in multi_pose:
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

class DemoOffline(IO):
    def start(self):
        # initiate
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()

        if data_numpy is None:
            print("Error: Could not extract skeleton data.")
            return

        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        # model predict
        voting_label_name, video_label_name, output, intensity = self.predict(data)

        # render the video
        images = self.render_video(data_numpy, voting_label_name, video_label_name, intensity, video)

        # visualize
        for image in images:
            image = image.astype(np.uint8)
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def pose_estimation(self):
        # Read color and depth images
        color_image_dir = self.arg.color_dir
        depth_image_dir = self.arg.depth_dir
        color_images = self.load_images(color_image_dir, ext=['.jpg', '.png'])
        depth_images = self.load_images(depth_image_dir, ext=['.png'])

        if not color_images or not depth_images or len(color_images) != len(depth_images):
            print("Error: Color or depth image sequences are missing or have different lengths.")
            return [], None

        video = []
        pose_tracker = naive_pose_tracker(data_frame=len(color_images))

        for frame_index in range(len(color_images)):
            color_path = color_images[frame_index]
            depth_path = depth_images[frame_index]
            orig_image = cv2.imread(color_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            if orig_image is None or depth_image is None:
                print(f"Error reading image: {color_path} or {depth_path}")
                continue

            # Process color image for pose estimation using MediaPipe
            multi_pose = self.process_image(orig_image, depth_image)

            if multi_pose is None or len(multi_pose.shape) != 3:
                print(f"Warning: No pose detected in frame {frame_index + 1}.")
                continue

            # normalization and pose tracking
            self.normalize_pose(multi_pose, orig_image)
            pose_tracker.update(multi_pose, frame_index)
            video.append(orig_image)

            print('Pose estimation ({}/{}).'.format(frame_index + 1, len(color_images)))

        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy

    def load_images(self, folder_path, ext):
        images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.endswith(e) for e in ext)])
        return images

    def process_image(self, orig_image, depth_image):
        # 使用 MediaPipe 进行姿态估计
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        multi_pose = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * orig_image.shape[1])
                    y = int(landmark.y * orig_image.shape[0])

                    x_depth = max(0, min(x, depth_image.shape[1] - 1))
                    y_depth = max(0, min(y, depth_image.shape[0] - 1))

                    z = depth_image[y_depth, x_depth] / 1000.0  # 将深度从毫米转换为米
                    hand_coords.append([landmark.x, landmark.y, z])
                if len(hand_coords) == 21:
                    multi_pose.append(hand_coords)

            if multi_pose:
                return np.array(multi_pose)
        return None

    def normalize_pose(self, multi_pose, orig_image):
        source_H, source_W, _ = orig_image.shape
        multi_pose[:, :, 0] = multi_pose[:, :, 0] / source_W
        multi_pose[:, :, 1] = multi_pose[:, :, 1] / source_H
        multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
        multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
        multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                               for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network with recorded color and depth images')


        parser.add_argument('--color_dir',
                            default='./color_images',
                            help='Path to the directory containing recorded color images')
        parser.add_argument('--depth_dir',
                            default='./depth_images',
                            help='Path to the directory containing recorded depth images')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

if __name__ == "__main__":
    parser = DemoOffline.get_parser()
    args = parser.parse_args()
    demo = DemoOffline(args)
    demo.start()
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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

class DemoRealtime(IO):

    def start(self):
        video_name = self.arg.video.split('/')[-1].split('.')[0]
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # initiate
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        # start recognition
        start_time = time.time()
        frame_index = 0
        while True:
            tic = time.time()

            # get image
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            orig_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # pose estimation using MediaPipe
            results = pose.process(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                multi_pose = []
                for landmark in results.pose_landmarks.landmark:
                    x, y = int(landmark.x * orig_image.shape[1]), int(landmark.y * orig_image.shape[0])
                    z = depth_image[y, x] / 1000.0  # Convert depth from mm to meters
                    multi_pose.append([x, y, z])
                multi_pose = np.array([multi_pose])
            else:
                multi_pose = np.zeros((0, 33, 3))  # No pose detected

            # normalization
            if len(multi_pose) > 0:
                multi_pose[:, :, 0] = multi_pose[:, :, 0] / orig_image.shape[1]
                multi_pose[:, :, 1] = multi_pose[:, :, 1] / orig_image.shape[0]
                multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
                multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
                multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            # pose tracking
            frame_index = int((time.time() - start_time) * self.arg.fps)
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()
            if data_numpy is not None:
                data = torch.from_numpy(data_numpy)
                data = data.unsqueeze(0)
                data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

                # model predict
                voting_label_name, video_label_name, output, intensity = self.predict(data)

                # visualization
                app_fps = 1 / (time.time() - tic)
                image = self.render(data_numpy, voting_label_name, video_label_name, intensity, orig_image, app_fps)
                cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        voting_label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]

        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l] for l in latest_frame_label]

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

    def render(self, data_numpy, voting_label_name, video_label_name, intensity, orig_image, fps=0):
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]], [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps)
        image = next(images)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        parser.add_argument('--model_input_frame', default=128, type=int)
        parser.add_argument('--model_fps', default=30, type=int)
        parser.add_argument('--height', default=1080, type=int, help='height of frame in the output video.')
        parser.set_defaults(config='./config/st_gcn/kinetics-skeleton/demo_realtime.yaml')
        parser.set_defaults(print_log=False)
        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=33, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:
            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
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

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
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
        # trace.shape: (num_frame, num_joint, 3)
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

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(1)) ** 0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
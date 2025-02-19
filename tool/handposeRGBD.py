import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
T = np.array([[R11, R12, R13, t1],
              [R21, R22, R23, t2],
              [R31, R32, R33, t3],
              [0, 0, 0, 1]])  

# 启动管道
pipeline.start(config)

# 获取深度尺度
profile = pipeline.get_active_profile()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# MediaPipe手势设置
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 参数设置
num_frames_to_fuse = 5  # 融合帧数
confidence_threshold = 0.9  # 置信度阈值
filter_size = 5  # 滤波器大小

# 用于存储深度帧的列表
depth_frames = []

# 创建一个窗口显示图像
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

while True:
    # 获取帧
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # 将深度帧添加到列表中
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_frames.append(depth_image)
    if len(depth_frames) > num_frames_to_fuse:
        depth_frames.pop(0)  # 删除最老的一帧

    # 处理彩色图像，获取手部关键点
    color_image = np.asanyarray(color_frame.get_data())
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # 获取像素坐标
                h, w, c = color_image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # 检查关键点是否在图像范围内
                if 0 <= cx < w and 0 <= cy < h:
                    # 融合深度数据，并获取当前关键点的深度值和置信度
                    fused_depth = np.median(depth_frames, axis=0)  # 中值滤波
                    fused_depth = cv2.blur(fused_depth, (filter_size, filter_size))  # 滤波
                    depth = fused_depth[cy, cx] * depth_scale
                    confidence = depth_frame.get_confidence(cx, cy)

                    # 判断置信度是否满足阈值
                    if confidence >= confidence_threshold:
                        # 计算3D坐标 
                        intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
                        depth_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                        # 像素坐标到相机坐标
                        camera_point = np.array([depth_point[0], depth_point[1], depth_point[2], 1]).T

                        # 相机坐标到世界坐标
                        world_point = np.dot(np.linalg.inv(T), camera_point)
                        print(f"Landmark {id}: 3D coordinates: {depth_point}")

                        # 绘制关键点和深度信息
                        cv2.circle(color_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                        cv2.putText(color_image, f"{depth:.2f}m", (cx, cy - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                    else:
                        print(f"Landmark {id}: Low confidence, depth data discarded")

            # 绘制手部骨架
            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 显示图像
    images = np.hstack((color_image, cv2.applyColorMap(cv2.convertScaleAbs(fused_depth, alpha=0.03), cv2.COLORMAP_JET)))
    cv2.imshow('RealSense', images)

    if cv2.waitKey(5) & 0xFF == 27:
        break

pipeline.stop()

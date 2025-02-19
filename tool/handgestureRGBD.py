import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# 定义手势识别器相关的类

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 配置深度和彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./models/gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO)
 
 
 # 启动管道
pipeline.start(config)

# 获取深度尺度
profile = pipeline.get_active_profile()
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# 用于存储深度帧的列表
depth_frames = []
# 创建手势识别器实例
with GestureRecognizer.create_from_options(options) as recognizer:
    # 初始化摄像头
    cap = cv2.VideoCapture(2)
    frame_count = 0  # 初始化帧计数器
    while True:
    # while cap.isOpened():
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())

        imgRGB = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)  # cv2图像初始化
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        recognition_result = recognizer.recognize_for_video(mp_image, frame_count)
        frame_count += 1
        if recognition_result:
            if recognition_result.gestures:
                t = recognition_result.gestures[0][0].category_name
            else:
                t = "none"
            print(t)
        # print(t)
        cv2.putText(color_frame, t, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("HandsImage", color_frame)  # CV2窗体
        cv2.waitKey(1)  # 关闭窗体
 
 
    # cap.release()
    # cv2.destroyAllWindows()

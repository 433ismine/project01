import cv2
import numpy as np
import mediapipe as mp
from pyrealsense2 import pyrealsense2 as rs

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)


recording = False
hand_data_list = []

def save_hand_data(hand_landmarks, hand_data_list):
    for landmark in hand_landmarks.landmark:
        hand_data_list.append([landmark.x, landmark.y, landmark.z])

def main():
    recording = False
    try:
        while True:
            # 获取深度相机帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            # 使用MediaPipe进行手部跟踪
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 绘制手部关键点
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        cv2.circle(color_image, (int(landmark.x * 640), int(landmark.y * 480)), 5, (255, 0, 0), cv2.FILLED)

            # 显示图像
            cv2.imshow('Hand Tracking', color_image)

            # 按键控制
            key = cv2.waitKey(1)
            if key == ord('s'):  # 开始录制
                recording = True
                hand_data_list = []  # 清空当前数据列表
            elif key == ord('e'):  # 结束录制
                recording = False
                np.save('hand_data.npy', hand_data_list)  # 保存数据
            elif key == ord('q'):  # 退出程序
                break

            # 如果正在录制，保存手部数据
            if recording and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    save_hand_data(hand_landmarks, hand_data_list)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
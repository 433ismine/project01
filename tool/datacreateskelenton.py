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
import csv

def save_to_skeleton(file_name, hand_data_list):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'body', 'joint', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        joint_idx=21
        frame_num = 1
        for frame_data in hand_data_list:
            for joint_idx, joint_data in enumerate(frame_data):
                writer.writerow({
                    'frame': frame_num,
                    'body': 1,  
                    'joint': joint_idx +1 ,
                    'x': joint_data[0],
                    'y': joint_data[1],
                    'z': joint_data[2]
                  
                })
            frame_num += 1

def main():
    recording = False
    try:
        while True:
      
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())

            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        cv2.circle(color_image, (int(landmark.x * 640), int(landmark.y * 480)), 5, (255, 0, 0), cv2.FILLED)

     
            cv2.imshow('Hand Tracking', color_image)

       
            key = cv2.waitKey(1)
            if key == ord('s'):  
                recording = True
                hand_data_list = []  
            elif key == ord('e'):  
                recording = False
                save_to_skeleton('../data/graph/4.skeleton', hand_data_list)
           
            elif key == ord('q'):  
                break

           
            if recording and results.multi_hand_landmarks:
                frame_data = []
                for hand_landmarks in results.multi_hand_landmarks:
                    save_hand_data(hand_landmarks, frame_data)
                hand_data_list.append(frame_data)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
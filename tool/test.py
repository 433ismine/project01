#!/usr/bin/env python
# coding:utf-8
# import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from PIL import ImageDraw, ImageFont
import cv2
import mediapipe as mp
import time
from geometry_msgs.msg import Point
# from PIL import ImageFont

if __name__ == '__main__':
    # rospy.init_node("point",anonymous=True)
    # pub = rospy.Publisher('point',Point, queue_size=10)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    # font_path = "truetype/wqy/wqy-microhei.ttc"
    # font = ImageFont.truetype(font_path, 20, encoding="utf-8")

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)


        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    msg=Point()
                    msg.x=id
                    msg.y=cx
                    msg.z=cy
                    # pub.publish(msg)
                    print(id, cx, cy)

                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                    cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

       #帧率计算
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        cv2.imshow('Hand Landmarks', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

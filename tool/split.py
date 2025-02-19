import cv2
import os
 
# 视频文件路径
video_path = 'gesture021.mp4'
 
# 输出图片的文件夹路径
output_folder = 'twirling of needle'
 
# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
# 打开视频文件
cap = cv2.VideoCapture(video_path)
 
frame_count = 0
while True:
    # 逐帧读取视频
    success, frame = cap.read()
    if not success:
        break  # 如果没有更多帧，则退出循环
 
    # 构建输出图片的文件名
    output_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
 
    # 保存帧为图片
    cv2.imwrite(output_filename, frame)
 
    frame_count += 1
 
# 释放视频对象
cap.release()
 
print(f'共保存了 {frame_count} 帧图片。')

import pyrealsense2 as rs
import cv2
import mediapipe as mp
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to check if a point is within the image bounds
def is_point_within_image(point, image_shape):
    h, w, _ = image_shape
    return 0 <= point[0] < w and 0 <= point[1] < h

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Process the color image with MediaPipe Hands
    results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get pixel coordinates
                h, w, _ = color_image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Check if landmark is within image bounds
                if is_point_within_image((cx, cy), color_image.shape):
                    try:
                        # Get depth value in meters
                        depth = depth_frame.get_distance(cx, cy)

                        # Get normalized image coordinates (0-1)
                        normalized_x = lm.x
                        normalized_y = lm.y

                        # Print landmark information (id, x, y, z)
                        print(f"Landmark {id}: ({normalized_x:.2f}, {normalized_y:.2f}, {depth:.2f}m)")

                        # Draw circle and label on the color image (optional)
                        cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        cv2.putText(color_image, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

                    except RuntimeError as e:
                        print(f"Error getting depth for landmark {id}: {e}")
                else:
                    print(f"Landmark {id} is outside image bounds")

            mp_draw.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show both color image with hand landmarks and depth image with colormap
    images = np.hstack((color_image, depth_colormap))
    cv2.imshow('Hand Tracking', images)

    if cv2.waitKey(5) & 0xFF == 27:
        break

pipeline.stop()

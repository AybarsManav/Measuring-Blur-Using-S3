import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_motion_blur_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    accum_image = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32)

        if accum_image is None:
            accum_image = frame
        else:
            accum_image += frame

        frame_count += 1

    cap.release()
    motion_blur = (accum_image / frame_count).astype(np.uint8)

    return motion_blur

video_path = 'video.mp4'  
motion_blur_img = create_motion_blur_from_video(video_path, max_frames=5)

orig = cv2.VideoCapture(video_path)
ret, frame = orig.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

alpha = 0.7  
beta = 1 - alpha 
out = (motion_blur_img.astype(np.float32) * alpha + img.astype(np.float32) * beta).astype(np.uint8)

plt.figure()
plt.imshow(out)
plt.show()
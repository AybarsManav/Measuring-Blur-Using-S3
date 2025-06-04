import cv2
import numpy as np
from scipy.ndimage import sobel
import matplotlib.pyplot as plt

def downsample_and_crop(frame, target_size=(270, 270)):
    # Bicubic downsampling by factor of 4
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)

    # Center crop
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    crop_x, crop_y = target_size[1] // 2, target_size[0] // 2
    cropped = frame[center_y - crop_y:center_y + crop_y, center_x - crop_x:center_x + crop_x]
    return cropped

def avg_gradient_magnitude(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(magnitude)

def compute_flow(prev, next):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow  

def filter_triplet(f1, f2, f3):
    # Gradient magnitude filter
    for f in (f1, f2, f3):
        if avg_gradient_magnitude(f) < 13:
            return False

    # Optical flow between f1→f2 and f2→f3
    flow12 = compute_flow(f1, f2)
    flow23 = compute_flow(f2, f3)

    # Sufficient motion: 10% pixels with ||flow||_inf >= 8
    for flow in [flow12, flow23]:
        motion_mag = np.linalg.norm(flow, ord=np.inf, axis=-1)
        if np.mean(motion_mag >= 8) < 0.10:
            return False

    # Limited motion: max motion < 16
    if np.max(np.linalg.norm(flow12, ord=np.inf, axis=-1)) > 16 or \
       np.max(np.linalg.norm(flow23, ord=np.inf, axis=-1)) > 16:
        return False

    # No abrupt changes (L1 difference after warping)
    h, w = f1.shape[:2]
    flow = flow12
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped_f1 = cv2.remap(f1, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    l1_dist = np.mean(np.abs(f2.astype(np.float32) - warped_f1.astype(np.float32)))
    if l1_dist > 13:
        return False

    backward_flow = compute_flow(f2, f1)
    forward_flow = flow23
    linear_approx_error = np.mean(np.linalg.norm(forward_flow + backward_flow, axis=-1))
    if linear_approx_error > 0.8:
        return False

    return True

def extract_valid_triplets(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(downsample_and_crop(frame))

    cap.release()

    valid_triplets = []
    for i in range(len(frames) - 2):
        f1, f2, f3 = frames[i], frames[i+1], frames[i+2]
        if filter_triplet(f1, f2, f3):
            valid_triplets.append((f1, f2, f3))

    return valid_triplets

def create_motion_blur_from_triplets(triplets):
    motion_blurred_images = []
    for triplet in triplets:
        motion_blur = np.mean(triplet, axis=0).astype(np.uint8)
        motion_blurred_images.append(motion_blur)
    return motion_blurred_images

triplets = extract_valid_triplets('video.mp4')

motion_blurred_images = create_motion_blur_from_triplets(triplets)

plt.figure()
plt.imshow(motion_blurred_images[0])
plt.show()
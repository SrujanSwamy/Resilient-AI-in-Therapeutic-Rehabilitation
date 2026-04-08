import numpy as np
import cv2

# ============================================================
# CHANGE THESE PATHS TO VIEW DIFFERENT DATA FILES
# ============================================================
EXERCISE = 'Ex2'           # Ex1, Ex2, Ex3, Ex4, Ex5, Ex6
SAMPLE = 'PM_003'          # PM_000, PM_001, PM_002, etc.
CAMERA = 'c17' 
CAMERA_NO='17'            # c17 or c18
FPS = 30                   # 30 or 120
# ============================================================

# Build file paths
joints_path = f'dataset/2d_joints/{EXERCISE}/{SAMPLE}-{CAMERA}-{FPS}fps.npy'
video_path = f'dataset/videos/{EXERCISE}/{SAMPLE}-Camera{CAMERA_NO}-{FPS}fps.mp4'

# Load 2D joints data
joints_2d = np.load(joints_path)
print(f"Loaded: {joints_path}")
print(f"Joints shape: {joints_2d.shape}")  # (frames, joints, 2)

# Load video
cap = cv2.VideoCapture(video_path)
if cap.isOpened():
    print(f"Loaded: {video_path}")
else:
    print(f"Video not found: {video_path}")

# Skeleton connections (joint indices)
skeleton = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # spine to head
    (2, 6), (6, 7), (7, 8), (8, 9), (9, 10),  # left arm
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # right arm
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),  # left leg
    (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),  # right leg
]

frame_idx = 0
num_joints_frames = len(joints_2d)

print("Playing video... Press 'q' to quit, SPACE to pause/resume")

paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw skeleton if we have joint data for this frame
        if frame_idx < num_joints_frames:
            joints = joints_2d[frame_idx]
            
            # Draw skeleton lines
            for start, end in skeleton:
                pt1 = (int(joints[start, 0]), int(joints[start, 1]))
                pt2 = (int(joints[end, 0]), int(joints[end, 1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Draw joints
            for j in range(len(joints)):
                pt = (int(joints[j, 0]), int(joints[j, 1]))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        
        # Display frame number
        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frame_idx += 1
    
    cv2.imshow('Exercise Video', frame)
    
    key = cv2.waitKey(33) & 0xFF  # ~30fps playback
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
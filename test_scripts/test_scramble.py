import cv2
import numpy as np
import hashlib
import random

# -------- CONFIG --------
video_path = 'input_videos/test_video.mp4'
output_path = 'output/scrambled_video.mp4'
PASSWORD = "test123123$"
GRID = 120   # very tiny subsections
# ------------------------

def password_to_seed(password):
    hash_val = hashlib.sha256(password.encode()).hexdigest()
    return int(hash_val[:8], 16)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# Read first frame to get size
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

height, width = frame.shape[:2]

# Trim so GRID divides perfectly
height = (height // GRID) * GRID
width = (width // GRID) * GRID

fps = cap.get(cv2.CAP_PROP_FPS)

# ---- Video Writer ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---- Password-based permutation ----
seed = password_to_seed(PASSWORD)
random.seed(seed)

indices = list(range(GRID * GRID))
random.shuffle(indices)
mapping = np.array(indices)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = frame[:height, :width]

    # Split into tiny blocks
    blocks = frame.reshape(
        GRID, height // GRID,
        GRID, width // GRID,
        3
    ).swapaxes(1, 2)

    blocks = blocks.reshape(GRID * GRID, height // GRID, width // GRID, 3)

    # Scramble
    scrambled_blocks = blocks[mapping]

    # Rebuild frame
    scrambled = scrambled_blocks.reshape(
        GRID, GRID,
        height // GRID,
        width // GRID,
        3
    ).swapaxes(1, 2).reshape(height, width, 3)

    # ---- SAVE FRAME ----
    out.write(scrambled)

    # Optional preview
    cv2.imshow("Tiny Password Scrambled Video", scrambled)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Scrambled video saved to: {output_path}")

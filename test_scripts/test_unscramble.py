import cv2
import numpy as np
import hashlib
import random

# -------- CONFIG --------
video_path = 'output/scrambled_video.mp4'
output_path = 'output/unscrambled_video.mp4'
PASSWORD = "test123123$"
GRID = 120
# ------------------------

def password_to_seed(password):
    hash_val = hashlib.sha256(password.encode()).hexdigest()
    return int(hash_val[:8], 16)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# Read first frame
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

height, width = frame.shape[:2]

# Trim again (must match scramble)
height = (height // GRID) * GRID
width = (width // GRID) * GRID

fps = cap.get(cv2.CAP_PROP_FPS)

# ---- Video Writer ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---- Regenerate SAME permutation ----
seed = password_to_seed(PASSWORD)
random.seed(seed)

indices = list(range(GRID * GRID))
random.shuffle(indices)
mapping = np.array(indices)

# 🔑 INVERSE MAPPING
inverse_mapping = np.argsort(mapping)

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

    # 🔓 UNSCRAMBLE
    restored_blocks = blocks[inverse_mapping]

    # Rebuild frame
    restored = restored_blocks.reshape(
        GRID, GRID,
        height // GRID,
        width // GRID,
        3
    ).swapaxes(1, 2).reshape(height, width, 3)

    # Save frame
    out.write(restored)

    # Optional preview
    cv2.imshow("Unscrambled Video", restored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Unscrambled video saved to: {output_path}")

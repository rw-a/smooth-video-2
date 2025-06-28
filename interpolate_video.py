import cv2
import math
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# ========== Import your codebase ==========
sys.path.append('.')
import utils.config as cfg
from utils.model import Model
from utils.padder import InputPadder

# ========== Argument parsing ==========
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VFIMamba_S', type=str)
parser.add_argument('--n', default=4, type=int, help='Number of total output frames between each input pair')
parser.add_argument('--scale', default=0.0, type=float)
parser.add_argument('input_video', type=str, help='Path to input video')
parser.add_argument('output_video', type=str, help='Path to save output video')

args = parser.parse_args()
assert args.model in ['VFIMamba_S', 'VFIMamba'], 'Model not exists!'

# ========== Model setup ==========
TTA = args.model == 'VFIMamba'
if TTA:
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F=32,
        depth=[2, 2, 2, 3, 3]
    )

model = Model(-1)
model.load_model()
model.eval()
model.device()

# ========== Recursive frame interpolation ==========
def _recursive_generator(frame1, frame2, down_scale, num_recursions, index):
    if num_recursions == 0:
        yield frame1, index
    else:
        mid_frame = model.inference(frame1, frame2, True, TTA=TTA, fast_TTA=TTA, scale=args.scale)
        id = 2 ** (num_recursions - 1)
        yield from _recursive_generator(frame1, mid_frame, down_scale, num_recursions - 1, index - id)
        yield from _recursive_generator(mid_frame, frame2, down_scale, num_recursions - 1, index + id)

# ========== Read video ==========
cap = cv2.VideoCapture(args.input_video)
assert cap.isOpened(), f"Failed to open video: {args.input_video}"

fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output_video, fourcc, fps * args.n, (frame_w, frame_h))

print('========================= Start Interpolating =========================')

ret, prev_frame = cap.read()
frame_index = 0

# Create a tqdm progress bar for frame interpolation (n-1 interpolated segments from frame count)
with tqdm(total=total_frames - 1, desc="Interpolating video") as pbar:
    while ret:
        ret, next_frame = cap.read()
        if not ret:
            break

        f0 = (torch.tensor(prev_frame.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
        f2 = (torch.tensor(next_frame.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

        padder = InputPadder(f0.shape, divisor=32)
        f0, f2 = padder.pad(f0, f2)

        num_recursions = int(math.log2(args.n))
        frame_pairs = list(_recursive_generator(f0, f2, 1.0, num_recursions, args.n // 2))
        frame_pairs = sorted(frame_pairs, key=lambda x: x[1])

        for pred, _ in frame_pairs:
            pred = pred[0]
            pred_np = (padder.unpad(pred).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            out.write(pred_np)

        prev_frame = next_frame
        frame_index += 1
        pbar.update(1)

cap.release()
out.release()

print('========================= Done =========================')

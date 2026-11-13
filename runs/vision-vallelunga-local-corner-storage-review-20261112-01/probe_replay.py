
import argparse
from pathlib import Path
import time

from extract_next_corner_flyvis import extract_pair, setup
from train_vision_teacher_line import read

parser=argparse.ArgumentParser()
parser.add_argument("variant", choices=("original", "eager"))
parser.add_argument("--through", type=int, default=26)
args=parser.parse_args()
import torch
import cv2
torch.set_num_threads(1)
cv2.setNumThreads(1)
print(f"torch={torch.__version__} hip={torch.version.hip}", flush=True)
network, projector, indices, pool, step, _ = setup(Path("runs/vision-next-corner-flyvis-cache-20261110"), batch_size=8)
if args.variant == "eager":
    step = network.simulate
root=Path("runs/vision-vallelunga-local-corner-capture-20261112-01")
camera=next(root.rglob("camera.zip")).parent
start=time.monotonic()
def progress(row, features, images):
    print(row["capture_id"], round(time.monotonic()-start,3), flush=True)
extract_pair(camera,set(range(args.through+1)),network,projector,indices,pool,step,batched=True,on_frame=progress)

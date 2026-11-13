"""Predict image-derived boundaries, with optional image heat-map refinement."""

import numpy as np
import torch

from model import (BoundaryNet, CROP, SIZE, STATIONS, SUPPORT_CUT, input_image, predict)

REFINE_RADIUS_PX = 16.
HEAT_CUT = .5


def readout(heat, anchor_px, row_px):
    """Choose an observed heat-map peak near the metric head, with subpixel localization."""
    row = (row_px - CROP[0] + .5)/2. - .5
    if not np.isfinite(row) or not 0. <= row <= SIZE[1]-1:
        return None
    below, above = int(np.floor(row)), int(np.ceil(row))
    scores = heat[below] + (row-below) * (heat[above]-heat[below])
    native_columns = 2.*(np.arange(SIZE[0])+.5)-.5
    delta = abs(native_columns-anchor_px)
    rank = np.where((delta <= REFINE_RADIUS_PX) & (scores >= HEAT_CUT), scores - .015*delta, -np.inf)
    col = int(np.argmax(rank))
    if not np.isfinite(rank[col]):
        return None
    offset = 0.
    if 0 < col < len(scores)-1:
        a,b,c = scores[col-1:col+2]
        denom = a-2.*b+c
        if denom < -1e-6:
            offset = float(np.clip(.5*(a-c)/denom, -.5, .5))
    return float(native_columns[col] + 2.*offset)


@torch.no_grad()
def infer(cache, checkpoint, mount):
    """Consume only cropped front video, a frozen checkpoint, and the fixed mount."""
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoundaryNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with np.load(cache, allow_pickle=False) as source:
        images = source["images"]
        available = source["available"]
        keys = list(zip(source["run"], source["frame"], strict=True))
    assert len(set(keys)) == len(keys)
    output = np.full((len(keys), len(STATIONS), 2, 4), np.nan, dtype=np.float32)
    depth = STATIONS-mount["origin_body_m"][0]
    scale = .5*319.*mount["projection_x"]/depth
    flat_rows = (.5+.5*mount["projection_y"]*mount["origin_body_m"][2]/depth)*179.
    for start in range(0, len(keys), 48):
        indices = np.arange(start, min(start+48, len(keys)))
        indices = indices[available[indices]]
        if not len(indices):
            continue
        raw, heat = model(input_image(torch.from_numpy(images[indices]), device))
        meters, rows, support = (a.cpu().numpy() for a in predict(raw))
        heat = torch.sigmoid(heat).cpu().numpy()
        for i, index in enumerate(indices):
            for station, side in np.argwhere((support[i] >= SUPPORT_CUT) & (abs(meters[i]) <= 16.)):
                lateral = float(meters[i,station,side])
                y = float(rows[i,station,side])
                anchor = 159.5-scale[station]*(lateral-mount["origin_body_m"][1])
                if not 0. <= anchor <= 319.:
                    continue
                output[index,station,side,:2] = (lateral,y)
                for k, row in enumerate((y, flat_rows[station]), 2):
                    x = readout(heat[i,side], anchor, row)
                    if x is not None:
                        output[index,station,side,k] = mount["origin_body_m"][1] + (159.5-x)/scale[station]
    return keys, available, output

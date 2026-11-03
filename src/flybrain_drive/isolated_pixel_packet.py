"""Strict bounded immutable pixel transfer for isolated diagnostic, not actor admission."""
import time
import numpy as np
from flybrain_drive.immutable_camera_group import freeze_camera_group
from flybrain_drive.flyvis_camera import VIEW_ORDER

PIXEL_BYTES = 180*320*3

def pack_candidate(camera):
    candidate, selected, _ = camera.wait_for_candidate(max_age=.15, timeout=.04)
    group = freeze_camera_group(candidate, camera.samples_for(candidate), selected_at=selected)
    return {'candidate':candidate, 'owner_selected_at':selected,
            'pixels':{name:group['pixels'][name].tobytes() for name in VIEW_ORDER}}

def unpack_candidate(packet, *, received_at=None):
    if not isinstance(packet,dict) or set(packet)!= {'candidate','owner_selected_at','pixels'}:
        raise ValueError('Exact pixel packet required')
    candidate=packet['candidate']; pixels=packet['pixels']
    if not isinstance(pixels,dict) or set(pixels)!=set(VIEW_ORDER):raise ValueError('Exact four pixel payloads required')
    if any(not isinstance(pixels[n],bytes) or len(pixels[n])!=PIXEL_BYTES for n in VIEW_ORDER):raise ValueError('Exact bounded original RGB bytes required')
    now=time.perf_counter() if received_at is None else received_at
    if not isinstance(packet['owner_selected_at'],(int,float)) or not candidate['received']<=packet['owner_selected_at']<=now:raise ValueError('Causal owner and receiver times required')
    views={v['view_id']:v for v in candidate['views']}
    samples={name:dict(rgb=np.frombuffer(pixels[name],dtype=np.uint8).reshape(180,320,3),capture_id=candidate['capture_id'],generation=candidate['generation'],received=candidate['received'],render_started_lower=views[name]['render_started_lower']) for name in VIEW_ORDER}
    group=freeze_camera_group(candidate,samples,selected_at=now)
    return candidate, group

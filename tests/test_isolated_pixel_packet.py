import hashlib
import numpy as np
import pytest
from flybrain_drive.isolated_pixel_packet import unpack_candidate


def packet():
    names=('front','left','right','rear'); data=bytes(180*320*3)
    candidate=dict(source_kind='csp_four_camera',ideal_line_shown=False,aid_probe_only=False,generation='g',capture_id=3,received=1.,views=[dict(view_id=n,render_started_lower=.99,rgb_sha256=hashlib.sha256(data).hexdigest()) for n in names])
    return dict(candidate=candidate,owner_selected_at=1.01,pixels={n:data for n in names})

def test_received_pixels_are_immutable_and_verify_original_identity():
    candidate, group=unpack_candidate(packet(),received_at=1.02)
    assert candidate['capture_id']==3 and group['capture_id']==3
    assert all(not a.flags.writeable and a.shape==(180,320,3) for a in group['pixels'].values())

def test_pixel_corruption_and_stale_ipc_are_rejected():
    p=packet();p['pixels']['left']=b'X'+p['pixels']['left'][1:]
    with pytest.raises(ValueError,match='hash'):
        unpack_candidate(p,received_at=1.02)
    with pytest.raises(ValueError,match='age'):
        unpack_candidate(packet(),received_at=1.16)

def test_missing_view_rejected():
    p=packet();del p['pixels']['rear']
    with pytest.raises(ValueError,match='four'):
        unpack_candidate(p,received_at=1.02)

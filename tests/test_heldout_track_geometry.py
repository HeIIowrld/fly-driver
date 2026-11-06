from copy import deepcopy

import pytest

from scripts.restart_heldout_track import verify_geometry


def test_stationary_geometry_is_bound_to_exact_session_and_rows():
    request={"command":"sample-heldout-geometry"}
    before=dict(generation="same",epoch=3,track_id="magione",window_handle="4ULL",
                paused=False,in_menu=True,ai_controlled=False,camera_active=False)
    rows=[dict(progress=i/512,native_xyz=[i,1,-i],left_m=3,right_m=4) for i in range(512)]
    receipt=dict(status="heldout_geometry_sampled",command=request["command"],
                 geometry_schema="flybrain.csp.heldout_geometry.v1",actor_input=False,
                 before=deepcopy(before),rows=rows)
    assert verify_geometry(receipt,request,before) is rows
    receipt["before"]["epoch"]=4
    with pytest.raises(ValueError,match="Exact stationary"):
        verify_geometry(receipt,request,before)
    receipt["before"]["epoch"]=3
    rows[1]["right_m"]=float("inf")
    with pytest.raises(ValueError,match="positive side distances"):
        verify_geometry(receipt,request,before)

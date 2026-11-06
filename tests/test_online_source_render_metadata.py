import pytest
from flybrain_drive.camera_body_alignment import preserved_source_render_metadata


def test_online_source_metadata_does_not_alias_producer_dictionary():
    original=dict(source_body_before_render={'local_velocity':[1,2,3]},source_body_after_render={'local_velocity':[1,2,3]},render_started_clock=1.,render_call_finished_clock=1.001)
    kept=preserved_source_render_metadata(original)
    original['source_body_before_render']['local_velocity'][0]=999
    assert kept['source_body_before_render']['local_velocity']==[1,2,3]


def test_partial_source_witness_is_not_published_as_complete():
    with pytest.raises(ValueError):preserved_source_render_metadata(dict(source_body_before_render={},render_started_clock=1.))


def test_legacy_render_clock_alone_does_not_invent_source_evidence():
    assert preserved_source_render_metadata(dict(render_started_clock=1.,render_call_finished_clock=1.001)) is None

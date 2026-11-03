from flybrain_drive.online_neural_late_body import output_host_freshness


def test_new_body_does_not_hide_old_or_missing_held_body():
    args = dict(has_intent=True, selected_body_age=.012, render_ages=[.08]*4)
    assert output_host_freshness(**args, held_body_age=.049)
    assert not output_host_freshness(**args, held_body_age=.051)
    assert not output_host_freshness(**args, held_body_age=None)
    assert not output_host_freshness(**dict(args, selected_body_age=.051), held_body_age=.01)
    assert not output_host_freshness(**dict(args, render_ages=[.08, .08, .08, .151]), held_body_age=.01)
    assert not output_host_freshness(**dict(args, has_intent=False), held_body_age=.01)

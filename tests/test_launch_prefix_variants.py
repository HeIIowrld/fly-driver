import torch

from scripts.diagnose_ac_launch_prefix import prefix_variants


def test_counterfactual_sensor_masks_keep_original_clock_body_and_other_sources():
    original = dict(view_rgb=torch.ones(3, 4, 2, 3),
        view_contrast=torch.ones(3, 4, 2, 3), view_valid=torch.ones(3, 4, 2),
        contrast_valid=torch.ones(3, 4, 2), view_age=torch.ones(3, 4)*.05,
        organs=torch.ones(3, 132), organ_valid=torch.ones(3, 132))
    original['view_valid'][:, 3] = 0
    original['contrast_valid'][:, 3] = 0
    original['view_age'][:, 3] = 1.5
    modes, result = prefix_variants(original, {'temperature_missing': 54}, .15)
    for key, value in original.items():
        assert torch.equal(result[key][:, 0], value)
    column = modes.index('temperature_missing')
    assert not result['organs'][:, column, 54:90].any()
    assert not result['organ_valid'][:, column, 54:90].any()
    for start, stop in [(0, 54), (90, 132)]:
        assert torch.equal(result['organs'][:, column, start:stop],
                           original['organs'][:, start:stop])
    assert torch.equal(result['view_age'][:, column], original['view_age'])
    age = result['view_age'][:, modes.index('age_zero')]
    assert not age[:, :3].any() and (age[:, 3] == 1.5).all()
    assert (original['organs'] == 1).all() and (original['view_age'][:, :3] == .05).all()

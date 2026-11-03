import torch

from scripts.audit_native_student_replay import hide_visual_sources


def test_source_deprivation_masks_visual_values_times_and_validity_only():
    obs = dict(view_rgb=torch.arange(48.).reshape(1, 4, 4, 3),
        view_contrast=torch.ones(1, 4, 4, 3), view_valid=torch.ones(1, 4, 4),
        contrast_valid=torch.ones(1, 4, 4), view_age=torch.full((1, 4), .02),
        organs=torch.arange(132.)[None], organ_valid=torch.ones(1, 132))
    original = {k: v.clone() for k, v in obs.items()}
    hide_visual_sources(obs, [1, 2, 3], .15)
    for key in ('view_rgb', 'view_contrast', 'view_valid', 'contrast_valid'):
        assert torch.equal(obs[key][:, 0], original[key][:, 0])
        assert not obs[key][:, 1:].any()
    assert torch.equal(obs['view_age'][:, :1], original['view_age'][:, :1])
    assert torch.equal(obs['view_age'][:, 1:], torch.full((1, 3), 1.5))
    assert torch.equal(obs['organs'], original['organs'])
    assert torch.equal(obs['organ_valid'], original['organ_valid'])

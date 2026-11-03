"""Counterfactual diagnostics isolate image content and source timing without repairing masks."""

from types import SimpleNamespace

import pytest
import torch

from scripts.audit_native_student_replay import pixel_timing_ablation


@pytest.mark.parametrize('mode', [
    'pixels_training_mean', 'pixels_first_frame', 'valid_source_age_zero'])
def test_counterfactual_changes_only_declared_fields_and_keeps_body_and_validity(mode):
    obs = dict(view_rgb=torch.full((1, 2, 3, 3), .8),
               view_contrast=torch.full((1, 2, 3, 3), .1),
               view_valid=torch.ones(1, 2, 3), contrast_valid=torch.ones(1, 2, 3),
               view_age=torch.tensor([[.04, .5]]), organs=torch.ones(1, 2),
               organ_valid=torch.ones(1, 2))
    obs['view_valid'][:, 1] = 0
    obs['contrast_valid'][:, 1] = 0
    original = {k: v.clone() for k, v in obs.items()}
    model = SimpleNamespace(photometric_normalization=True,
                            photometric_mean=torch.full((2, 3, 6), .2))
    first = torch.full((1, 2, 3, 3), .4)
    pixel_timing_ablation(obs, model, mode, first)
    for key in ('view_valid', 'contrast_valid', 'organs', 'organ_valid'):
        torch.testing.assert_close(obs[key], original[key], rtol=0, atol=0)
    if mode == 'valid_source_age_zero':
        torch.testing.assert_close(obs['view_age'], torch.tensor([[0., .5]]))
        torch.testing.assert_close(obs['view_rgb'], original['view_rgb'])
    else:
        torch.testing.assert_close(obs['view_age'], original['view_age'])
        torch.testing.assert_close(obs['view_rgb'], torch.full_like(first,
            .2 if mode == 'pixels_training_mean' else .4))
        torch.testing.assert_close(obs['view_contrast'], torch.full_like(first,
            .2 if mode == 'pixels_training_mean' else 0.))

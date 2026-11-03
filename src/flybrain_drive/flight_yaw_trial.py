"""Explicit experiment boundary around the unchanged certified inference stream."""


def apply_yaw_mode(stream, mode):
    """Keep original sensor records/masks; zero only the declared yaw drive value."""
    if mode not in ('full', 'zero_yaw') or hasattr(stream, 'yaw_trial_mode'):
        raise ValueError('One explicit full/zero_yaw condition per original episode required')
    stream.yaw_trial_mode = mode
    if mode == 'full':
        return stream
    original = stream.flight._drive_tensor

    def drive(visual, visual_valid, body, body_valid, goals, scene_features, *, lesion=None):
        intervened = body.clone()
        intervened[:, 4] = 0.
        return original(visual, visual_valid, intervened, body_valid, goals, scene_features,
                        lesion=lesion)

    stream.flight._drive_tensor = drive
    return stream

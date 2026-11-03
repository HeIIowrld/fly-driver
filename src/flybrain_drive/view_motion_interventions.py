"""Offline feature interventions; never used as camera-loss handling."""

def mean_motion_views(scene, training_mean, views):
    """Replace selected views' eight pooled motion channels, preserving intensity."""
    if scene.shape != (1, 576) or training_mean.numel() != 576:
        raise ValueError("Expected four views, nine channels and sixteen bins")
    views = tuple(views)
    if len(set(views)) != len(views) or any(v not in range(4) for v in views):
        raise ValueError("Invalid or repeated view")
    result = scene.clone().reshape(1, 4, 9, 16)
    mean = training_mean.reshape(4, 9, 16)
    for view in views:
        result[:, view, 1:, :] = mean[view, 1:, :]
    return result.reshape(1, 576)

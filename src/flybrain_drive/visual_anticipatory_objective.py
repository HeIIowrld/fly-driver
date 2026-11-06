"""Training-only future steering supervision through existing muscle activity."""
import torch


def visual_anticipatory_loss(prediction,target,source_pedals,*,pre,
                              forecast=None,future_target=None):
    if prediction.shape!=(1,3) or target.shape!=(3,) or source_pedals.shape!=(2,):
        raise ValueError('One current three-axis command and original pedal anchor required')
    if not all(torch.isfinite(v).all() for v in (prediction,target,source_pedals)):
        raise ValueError('Finite qualified current supervision required')
    # Source pedals are a fixed engineering regression anchor, never actor input.
    loss=(64. if pre else 16.)*(prediction[0,0]-target[0].detach()).square()
    loss=loss+(prediction[0,1:]-target[1:].detach()).square().sum()
    loss=loss+64.*(prediction[0,1:]-source_pedals.detach()).square().sum()
    if (forecast is None)!=(future_target is None):raise ValueError('Explicit paired future prediction/target required')
    if forecast is not None:
        if not pre or forecast.numel()!=1 or future_target.numel()!=1:
            raise ValueError('Future auxiliary supervision is restricted to qualified pre-corner rows')
        if not torch.isfinite(forecast).all() or not torch.isfinite(future_target).all():raise ValueError('Finite future target required')
        loss=loss+16.*(forecast.reshape(())-future_target.detach().reshape(())).square()
    return loss

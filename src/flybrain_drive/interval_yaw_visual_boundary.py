"""Experimental interval-yaw visual port into the existing single flight graph.

This is an engineering hypothesis, initialized to zero, not a measured synapse,
physiological validation, or an actor-qualified input.
"""
import math
import re
import torch
from torch import nn
from .view_yaw_interval import ViewYawInterval


class IntervalYawVisualBoundary(nn.Module):
    def __init__(self, port_count, *, body_contract_sha256, device=None, dtype=torch.float32):
        super().__init__()
        if (type(port_count) is not int or port_count < 1 or not isinstance(body_contract_sha256, str)
                or re.fullmatch(r'[0-9a-f]{64}', body_contract_sha256) is None):
            raise ValueError('Matching ports and exact body contract hash required')
        self.body_contract_sha256 = body_contract_sha256
        self.projection = nn.Linear(256, port_count, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.projection.weight)

    def pack(self, intervals, *, body_context, decision_time):
        if not isinstance(intervals, tuple) or len(intervals) != 4:
            raise ValueError('Four ordered interval entries, including explicit None, required')
        yaw, valid = [], []
        for i, item in enumerate(intervals):
            if item is None:
                yaw.append(0.); valid.append(False); continue
            if not isinstance(item, ViewYawInterval):
                raise ValueError('Typed interval or explicit None required')
            values = (item.start_render_lower, item.end_render_lower, item.camera_receipt,
                      item.decision, item.yaw_left_mean_rad_s, item.yaw_left_integral_rad,
                      item.first_receipt, item.last_receipt)
            if (any(not isinstance(x, (int, float)) or isinstance(x, bool)
                    or not math.isfinite(x) for x in values)
                    or item.body_contract_sha256 != self.body_contract_sha256
                    or item.physical_time_verified is not False or item.actor_qualified is not False
                    or item.decision != decision_time or item.camera_receipt != body_context.camera_receipt_time
                    or item.end_render_lower != body_context.render_start_lowers[i]
                    or not 0 <= item.start_render_lower < item.end_render_lower <= item.camera_receipt <= item.decision
                    or item.end_render_lower-item.start_render_lower > .2
                    or not item.first_receipt <= item.start_render_lower
                    or not item.end_render_lower <= item.last_receipt <= item.decision
                    or type(item.first_packet) is not int or type(item.last_packet) is not int
                    or not 0 <= item.first_packet <= item.last_packet
                    or not math.isclose(item.yaw_left_integral_rad,
                                        item.yaw_left_mean_rad_s*(item.end_render_lower-item.start_render_lower),
                                        rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError('Interval provenance, clock or value differs')
            yaw.append(item.yaw_left_mean_rad_s); valid.append(True)
        return (torch.tensor([yaw], device=self.projection.weight.device, dtype=self.projection.weight.dtype),
                torch.tensor([valid], device=self.projection.weight.device, dtype=torch.bool))

    def forward(self, traces, yaw, valid):
        if (not isinstance(traces, tuple) or len(traces) != 3
                or any(not isinstance(x, torch.Tensor) or x.shape != (1,512)
                       or x.dtype != self.projection.weight.dtype or x.device != self.projection.weight.device
                       or not torch.isfinite(x).all() for x in traces)
                or not isinstance(yaw, torch.Tensor) or yaw.shape != (1,4)
                or yaw.dtype != self.projection.weight.dtype or yaw.device != self.projection.weight.device
                or not torch.isfinite(yaw).all() or not isinstance(valid, torch.Tensor)
                or valid.shape != (1,4) or valid.dtype != torch.bool or valid.device != yaw.device):
            raise ValueError('Finite interval yaw, explicit masks and four-view traces required')
        motion = (traces[0]-traces[1]).reshape(1,4,2,4,16)
        opponent = torch.stack((motion[:,:,:,1]-motion[:,:,:,0], motion[:,:,:,2]-motion[:,:,:,3]), dim=3)
        selected = torch.where(valid, yaw.clamp(-1,1), torch.zeros_like(yaw))
        return .5 * torch.tanh(self.projection((opponent * selected[:,:,None,None,None]).reshape(1,256)))

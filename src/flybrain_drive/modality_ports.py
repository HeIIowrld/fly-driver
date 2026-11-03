"""Explicit modality-to-input-port adapters; tuning is an engineering hypothesis."""

import numpy as np
import torch
from torch import nn

from .graph import validate_graph


class ModalityPortAdapter(nn.Module):
    """Visual features cannot directly enter organ ports or internal/output neurons.

    Organ feature indices must be declared per group. These learned projections do
    not establish physiological tuning of individual measured sensory neurons.
    """

    def __init__(self, graph, visual_ids, organ_groups, organ_size, visual_size,
                 visual_adapter=None, *, organ_validity_mode='learned_drive'):
        super().__init__()
        validate_graph(graph)
        if min(organ_size, visual_size) < 1:
            raise ValueError("Positive modality feature sizes required")
        if organ_validity_mode not in ('learned_drive', 'gate_only'):
            raise ValueError('Explicit organ validity encoding required')
        self.organ_validity_mode = organ_validity_mode
        self.size, self.organ_size, self.visual_size = len(graph["ids"]), organ_size, visual_size
        index = {str(body): i for i, body in enumerate(graph["ids"])}
        all_ids = list(map(str, visual_ids)) + [str(body) for group in organ_groups.values()
                                                for body in group["body_ids"]]
        inputs = {str(body) for body, role in zip(graph["ids"], graph["roles"], strict=True)
                  if role == "input"}
        if len(set(all_ids)) != len(all_ids) or set(all_ids) != inputs or not visual_ids:
            raise ValueError("Disjoint modality groups must cover exactly the graph input IDs")
        self.register_buffer("visual_ports", torch.tensor([
            index[str(body)] for body in visual_ids]))
        self.visual_adapter = (nn.Linear(visual_size, len(visual_ids), bias=False)
                               if visual_adapter is None else visual_adapter)
        self.organ_adapters = nn.ModuleList()
        self.group_names = tuple(sorted(organ_groups))
        for i, name in enumerate(self.group_names):
            group = organ_groups[name]
            features = np.asarray(group["feature_indices"])
            if (features.ndim != 1 or not len(features) or features.dtype.kind not in "iu"
                    or features.min() < 0 or features.max() >= organ_size
                    or len(set(features.tolist())) != len(features)):
                raise ValueError("Organ groups require unique valid feature indices")
            self.register_buffer(f"group_features_{i}", torch.tensor(features, dtype=torch.long))
            self.register_buffer(f"group_ports_{i}", torch.tensor([
                index[str(body)] for body in group["body_ids"]], dtype=torch.long))
            self.organ_adapters.append(nn.Linear(len(features) * 2, len(group["body_ids"]),
                                                   bias=False))

    def forward(self, visual, visual_valid, organs, organ_valid):
        batch = len(visual)
        if (visual.shape != (batch, self.visual_size) or visual_valid.shape != (batch, 1)
                or organs.shape != (batch, self.organ_size) or organ_valid.shape != organs.shape
                or not all(torch.isfinite(v).all() for v in
                           (visual, visual_valid, organs, organ_valid))
                or not all(torch.all((v == 0) | (v == 1)) for v in (visual_valid, organ_valid))):
            raise ValueError("Modality values/masks differ from the declared input contract")
        drive = visual.new_zeros((batch, self.size))
        drive = drive.index_add(1, self.visual_ports,
                                self.visual_adapter(visual * visual_valid) * visual_valid)
        for i, adapter in enumerate(self.organ_adapters):
            indices = getattr(self, f"group_features_{i}")
            values, valid = organs[:, indices], organ_valid[:, indices]
            flags = (valid if self.organ_validity_mode == 'learned_drive'
                     else torch.zeros_like(valid))
            encoded = torch.cat([values * valid, flags], dim=1)
            drive = drive.index_add(1, getattr(self, f"group_ports_{i}"), adapter(encoded))
        return drive

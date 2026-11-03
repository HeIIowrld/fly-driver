"""Contact-constrained electrical-current hypothesis on identified existing rate neurons.

Chemical counts remain measured. Electrical coupling and selected-cell time constants are
declared engineering hypotheses, not measured conductances or a validated voltage model.
There is no connection to vehicle controls and no direct motor-output residual.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .episode_storage import load_array
from .graph import sha256


class FastHaltereDynamics:
    def __init__(self, model, inventory, *, maximum_microstep=.001):
        root = Path(inventory)
        receipt = json.loads((root/'result.json').read_text(encoding='utf8'))
        proof = json.loads((root/'manifest.json').read_text(encoding='utf8'))['config']
        if (receipt['status'] != 'chemical_fast_route_inventory_completed'
                or proof['spec_sha256'] != model.spec_sha256
                or not 0 < maximum_microstep <= .001):
            raise ValueError('Exact chemical-route inventory and <=1ms internal step required')
        spec = model.spec
        if sha256(spec['graph_file']) != spec['graph_sha256']:
            raise ValueError('Original measured graph changed')
        with load_array(spec['graph_file'], allow_pickle=False) as archive:
            ids = archive['ids'].tolist()
            pre, post, weights = (archive[key] for key in ('pre', 'post', 'weights'))
        index = {body: n for n, body in enumerate(ids)}
        annotations = {str(row['bodyId']): row for row in
                       json.loads(Path(spec['annotations_file']).read_text(encoding='utf8'))}
        if sha256(spec['annotations_file']) != spec['annotations_sha256']:
            raise ValueError('Original annotations changed')
        targets = [row for row in receipt['targets'] if row['type'] == 'b1 MN']
        if len(targets) != 2 or {row['soma_side'] for row in targets} != {'L', 'R'}:
            raise ValueError('Both identified b1 neurons required; no b2 electrical assumption')
        edges = []
        for target in targets:
            motor = target['body_id']
            declared = target['direct_edges']
            if not declared or annotations[motor]['type'] != 'b1 MN':
                raise ValueError('Measured nonempty haltere-to-b1 contact support required')
            for edge in declared:
                source = edge['pre_id']
                matches = (pre == index[source]) & (post == index[motor])
                if (source not in spec['body_input_ids']
                        or annotations[source]['subclass'] != 'haltere'
                        or matches.sum() != 1 or float(weights[matches][0]) != edge['synapses']):
                    raise ValueError('Contact inventory differs from original measured IDs/counts')
                edges.append((source, motor, edge['synapses']/target['direct_chemical_synapses']))
        self.ids = sorted({body for source, target, _ in edges for body in (source, target)},
                          key=index.get)
        local = {body: n for n, body in enumerate(self.ids)}
        device = model.circuit.matrix.device
        adjacency = torch.zeros(len(self.ids), len(self.ids), device=device)
        for source, target, contact_weight in edges:
            a, b = local[source], local[target]
            adjacency[a, b] += contact_weight
            adjacency[b, a] += contact_weight
        self.laplacian = torch.diag(adjacency.sum(1))-adjacency
        self.positions = torch.tensor([index[body] for body in self.ids], device=device)
        self.chemical_matrix = model.circuit.matrix.coalesce().to_sparse_csr()
        self.model = model
        self.maximum_microstep = maximum_microstep
        self.original_matrix_identity = (id(model.circuit.matrix), model.circuit.matrix._version)
        self.contract = dict(schema='flybrain.fast_haltere_hypothesis.v1',
            inventory_result_sha256=sha256(root/'result.json'),
            measured_chemical_graph_sha256=spec['graph_sha256'], selected_ids=self.ids,
            motors=[row['body_id'] for row in targets], supported_contacts=len(edges),
            electrical_edge_rule='Undirected on existing haltere-b1 chemical contacts only',
            electrical_weights='Per-motor normalized contact counts, symmetrized; inferred proxy',
            state_semantics='Original bounded rate state used as normalized depolarization proxy',
            electrical_conductances_measured=False, fast_time_constants_measured=False,
            chemistry='Original net held per original nominal/fractional neural step',
            original_graph_changed=False, added_neurons=0, direct_vehicle_residual=False)

    @torch.no_grad()
    def advance(self, drive, state, elapsed, *, coupling, fast_tau):
        nominal = self.model.spec['dynamics']['dt']
        if (not math.isfinite(elapsed) or not 0 < elapsed <= nominal+1e-12
                or state.shape != drive.shape or state.ndim != 2
                or self.original_matrix_identity != (id(self.model.circuit.matrix),
                                                      self.model.circuit.matrix._version)):
            raise ValueError('Original positive subinterval, state and unchanged matrix required')
        if not bool(torch.all(self.model.circuit.state_mask == 1)):
            raise ValueError('State lesions require a separately verified integrator')
        coupling = torch.as_tensor(coupling, dtype=state.dtype, device=state.device).reshape(-1, 1)
        fast_tau = torch.as_tensor(fast_tau, dtype=state.dtype, device=state.device).reshape(-1, 1)
        if (coupling.shape[0] not in (1, len(state)) or fast_tau.shape[0] not in (1, len(state))
                or not torch.isfinite(coupling).all() or not torch.isfinite(fast_tau).all()
                or torch.any((coupling < 0) | (coupling > 1))
                or torch.any((fast_tau < .005) | (fast_tau > .05))):
            raise ValueError('Coupling[0,1], fast tau[5,50]ms and batch alignment required')
        net = torch.sparse.mm(self.chemical_matrix, state.T).T+drive
        target = torch.relu(torch.tanh(net))
        nominal_next = ((1-self.model.circuit.alpha)*state
                        + self.model.circuit.alpha*target)
        tau = self.model.spec['dynamics']['tau']
        nominal_alpha = float(np.float32(-np.expm1(-nominal/tau)))
        ratio = -np.expm1(-elapsed/tau)/nominal_alpha
        next_state = (state+ratio*(nominal_next-state)
                      if abs(ratio-1) > 1e-7 else nominal_next)
        fast = state[:, self.positions]
        held_net = net[:, self.positions]
        count = int(math.ceil(elapsed/self.maximum_microstep-1e-12))
        alpha = -torch.expm1(torch.as_tensor(-elapsed/count, device=state.device)/fast_tau)
        for _ in range(count):
            electrical_current = -coupling*(fast@self.laplacian.T)
            activated = torch.relu(torch.tanh(held_net+electrical_current))
            fast = (1-alpha)*fast+alpha*activated
        # This control is the untouched original recurrence, including fractional tick arithmetic.
        baseline = (coupling == 0) & (fast_tau == tau)
        next_state[:, self.positions] = torch.where(baseline,
            next_state[:, self.positions], fast)
        return next_state

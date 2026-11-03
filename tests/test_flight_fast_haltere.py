import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flybrain_drive.flight_fast_haltere import FastHaltereDynamics
from flybrain_drive.graph import sha256


def fixture(tmp_path):
    # Explicit tiny synthetic test graph, never presented as a biological dataset.
    ids = ['sL', 'sR', 'bL', 'bR']
    np.savez(tmp_path/'graph.npz', ids=np.array(ids), pre=[0, 1], post=[2, 3], weights=[2, 3])
    annotations = [dict(bodyId=body, type='b1 MN' if n >= 2 else 'sensory',
                        subclass='haltere' if n < 2 else 'motor') for n, body in enumerate(ids)]
    (tmp_path/'annotations.json').write_text(json.dumps(annotations))
    spec = dict(graph_file=str(tmp_path/'graph.npz'), graph_sha256=sha256(tmp_path/'graph.npz'),
        annotations_file=str(tmp_path/'annotations.json'),
        annotations_sha256=sha256(tmp_path/'annotations.json'),
        body_input_ids=ids[:2], dynamics=dict(dt=.01, tau=.05))
    targets = [dict(type='b1 MN', soma_side=side, body_id=ids[n+2],
                    direct_chemical_synapses=n+2,
                    direct_edges=[dict(pre_id=ids[n], synapses=n+2)])
               for n, side in enumerate(('L', 'R'))]
    (tmp_path/'result.json').write_text(json.dumps(
        dict(status='chemical_fast_route_inventory_completed', targets=targets)))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=dict(spec_sha256='synthetic'))))
    matrix = torch.sparse_coo_tensor([[2, 3], [0, 1]], [.1, .2], (4, 4)).coalesce()
    model = SimpleNamespace(spec=spec, spec_sha256='synthetic',
        circuit=SimpleNamespace(matrix=matrix, state_mask=torch.ones(4),
                                alpha=torch.tensor(float(np.float32(-np.expm1(-.2))))))
    return model, FastHaltereDynamics(model, tmp_path)


def test_baseline_preserves_whole_original_state_and_fractional_interval(tmp_path):
    model, fast = fixture(tmp_path)
    state = torch.tensor([[.1, .2, .3, .4]])
    drive = torch.tensor([[.2, .3, 0., 0.]])
    original_state, original_drive = state.clone(), drive.clone()
    for elapsed in (.01, .0073):
        target = torch.relu(torch.tanh(torch.sparse.mm(model.circuit.matrix, state.T).T+drive))
        alpha = model.circuit.alpha
        nominal = (1-alpha)*state+alpha*target
        ratio = -np.expm1(-elapsed/.05)/float(np.float32(-np.expm1(-.2)))
        expected = state+ratio*(nominal-state) if abs(ratio-1) > 1e-7 else nominal
        actual = fast.advance(drive, state, elapsed, coupling=0., fast_tau=.05)
        assert torch.equal(actual, expected)
    assert torch.equal(state, original_state) and torch.equal(drive, original_drive)


def test_diffusive_coupling_is_symmetric_passive_and_bounded_without_source_mutation(tmp_path):
    _, fast = fixture(tmp_path)
    laplacian = fast.laplacian
    assert torch.equal(laplacian, laplacian.T)
    assert torch.equal(laplacian@torch.ones(4), torch.zeros(4))
    assert torch.linalg.eigvalsh(laplacian).min() >= -1e-6
    state = torch.tensor([[.1, .2, .3, .4], [.4, .3, .2, .1]])
    assert torch.all((state*(state@laplacian.T)).sum(1) >= 0)
    drive = torch.tensor([[.2, .3, 0., 0.]]).expand(2, -1)
    for _ in range(100):
        state = fast.advance(drive, state, .01, coupling=[.1, 1.], fast_tau=.005)
    assert torch.isfinite(state).all() and ((state >= 0) & (state <= 1)).all()
    with pytest.raises(ValueError):
        fast.advance(drive, state, .02, coupling=1., fast_tau=.005)
    with pytest.raises(ValueError):
        fast.advance(drive, state, .01, coupling=-1., fast_tau=.005)

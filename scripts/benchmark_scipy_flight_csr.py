"""Offline frozen CPU CSR comparison; no gradient, vehicle or model fitting."""

import json
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.flight_embodied_navigation import EmbodiedVisualFlightCircuit
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    root = Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs')
    train = root/'20261029-t1437-embodied-joint-training'
    tr, manifest = read(train/'result.json'), read(train/'manifest.json')['config']
    plan = manifest['plan']
    source = read(Path(plan['source'])/'result.json')
    if sha256(tr['checkpoint']) != tr['checkpoint_sha256']:
        raise ValueError('Exact flight model checkpoint required')
    torch.set_num_threads(1)
    torch.manual_seed(863009)
    model = EmbodiedVisualFlightCircuit(plan['source_spec'], torch.load(
        source['checkpoint'], map_location='cpu', weights_only=True))
    model.load_state_dict(torch.load(tr['checkpoint'], map_location='cpu', weights_only=True)['model'], strict=True)
    model.eval().requires_grad_(False)
    circuit = model.backbone.circuit
    circuit.sparse_backend = 'cached_csr_cpu'
    matrix = circuit.matrix.coalesce().to_sparse_csr()
    if matrix.requires_grad or matrix.device.type != 'cpu':
        raise ValueError('Fixed CPU graph required')
    csr = scipy.sparse.csr_matrix((matrix.values().numpy().copy(), matrix.col_indices().numpy().copy(),
                                   matrix.crow_indices().numpy().copy()), shape=tuple(matrix.shape))
    if csr.nnz != matrix.values().numel():
        raise ValueError('Every original edge must be retained')
    state = torch.rand(1, circuit.size)
    drive = torch.rand_like(state) * circuit.input_mask * .02
    def original_message(x):
        return torch.sparse.mm(matrix, x.T).T
    def scipy_message(x):
        return torch.from_numpy((csr @ x.numpy().T).T.copy())
    with torch.no_grad():
        initial_difference = float((original_message(state)-scipy_message(state)).abs().max())
        reference, candidate = state.clone(), state.clone()
        max_drift = 0.
        for _ in range(100):
            _, reference = circuit._advance_tensor(drive, reference)
            incoming = scipy_message(candidate) + drive
            candidate = ((1-circuit.alpha)*candidate + circuit.alpha*torch.relu(torch.tanh(incoming))) * circuit.state_mask
            max_drift = max(max_drift, float((candidate-reference).abs().max()))
        if max_drift > 2e-6:
            raise ValueError('CPU CSR trajectory differs beyond declared tolerance')
        for _ in range(20):
            original_message(state); scipy_message(state)
        timings = {'torch_csr': [], 'scipy_csr': []}
        for i in range(200):
            operations = [('torch_csr', original_message), ('scipy_csr', scipy_message)]
            if i % 2:
                operations.reverse()
            for name, operation in operations:
                start = time.perf_counter_ns(); operation(state)
                timings[name].append((time.perf_counter_ns()-start)/1e6)
    run = begin_run(root/'20261031-frozen-cpu-csr-kernel-comparison',
                    {'program_sha256': sha256(__file__), 'flight_checkpoint_sha256': tr['checkpoint_sha256'],
                     'seed': 863009, 'threads': 1, 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
                     'scope': 'synthetic fixed input and bounded state; fixed graph inference only'})
    write_json(run/'timings.json', timings)
    result = {'status':'completed', 'nodes':circuit.size, 'edges':int(csr.nnz),
              'initial_message_max_abs_error':initial_difference, 'steps':100, 'state_max_abs_drift':max_drift,
              'milliseconds_p50_p95_max':{name:np.percentile(v,[50,95,100]).tolist() for name,v in timings.items()},
              'timings_sha256':sha256(run/'timings.json'), 'original_history_tested':False,
              'gradients_supported':False, 'live_or_vehicle_tested':False}
    write_json(run/'result.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

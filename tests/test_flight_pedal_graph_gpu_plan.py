"""Metadata-only parity-policy tests; never import torch/load a model/query a device."""

import ast
import json
from pathlib import Path

import numpy as np

from scripts.qualify_flight_pedal_graph_gpu_v2 import GATES, parity, write


def test_gradient_tolerance_is_per_cell_and_cpu_referenced_including_zero():
    cpu = np.array([[0., .1], [-.2, 10.]])
    limit = GATES['gradient_absolute']+GATES['gradient_relative']*abs(cpu)
    assert parity(cpu, cpu+limit*.99, absolute=1e-5, relative=1e-3)['passed']
    failed = cpu.copy()
    failed[0, 0] = 1.00001e-5
    result = parity(cpu, failed, absolute=1e-5, relative=1e-3)
    assert not result['passed'] and result['failing_cells'] == 1
    assert result['CPU_zero_cells'] == result['GPU_nonzero_at_CPU_zero'] == 1


def test_state_and_output_limits_are_not_relative_or_adapted_after_failure():
    assert not parity([.5], [.5+1.01e-6], absolute=1e-6)['passed']
    assert not parity([.5], [.5+1.01e-4], absolute=1e-4)['passed']
    assert not parity([0.], [np.nan], absolute=1e-4)['finite']
    assert not parity([0.], [[0.]], absolute=1e-4)['passed']


def test_failure_evidence_retains_nonfinite_without_success_masking(tmp_path):
    output = tmp_path/'failed.json'
    write(output, dict(passed=False, raw_gradients=[np.nan, float('inf'), 0.]))
    assert json.loads(output.read_bytes()) == dict(passed=False,
                                                  raw_gradients=['nan', 'inf', 0.])


def test_staging_has_no_top_level_torch_or_model_import_and_gpu_uses_actual_csr_route():
    path = Path('scripts/qualify_flight_pedal_graph_gpu_v2.py')
    source = path.read_text(encoding='utf8')
    tree = ast.parse(source)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all('torch' not in ast.unparse(n) and 'flybrain_drive' not in ast.unparse(n)
               for n in imports)
    assert "gpu_model.backbone.circuit.sparse_backend = 'cached_csr_gpu'" in source
    assert source.index('cpu_model, cpu_staging = cpu_staging_model()') < source.index(
        "gpu_model = gpu_model.to('cuda')")
    assert "CPU_hash_never_used_as_actual_GPU_quiet_fingerprint=True" in source
    assert "original_CSR_adjoint_present" in source and "torch.autograd.grad" in source
    assert "cpu_row['loss_basis'] == [d['loss']" in source

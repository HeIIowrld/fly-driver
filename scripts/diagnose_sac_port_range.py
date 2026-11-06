"""R1 training-only visual input/port saturation audit of the exact checkpoint."""
import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import assemble_student, read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    root = Path('runs/teacher-synchronized-20261102')
    checkpoint = root / 'student-fit-01/unified-sac-student.pt'
    if sha256(checkpoint) != '24a8f4e03614636fe6eede71137faae161c8e82dcc1f212dc4bd4a75fb5203d4':
        raise ValueError('Original checkpoint required')
    entries = read(root / 'student-split-01.json')['episodes'][:2]
    if any(e['role'] != 'train' for e in entries):
        raise ValueError('Training episodes only')
    model, _ = assemble_student()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'], strict=True)
    model.eval()
    run = begin_run(a.output, dict(seed=780308, program_sha256=sha256(__file__),
        checkpoint_sha256=sha256(checkpoint), episodes=entries, optimizer_updates=0,
        source_scope='Actual visual affine/sigmoid and preclamp temporal normalization; no recurrent lesion or biological tuning claim',
        actor_deployed=False, development_opened=False))
    scene = []
    for e in entries:
        d, c, evidence = load_episode(e['data'])
        if evidence['arrays_sha256'] != e['arrays_sha256']:
            raise ValueError('Original train source changed')
        scene.append(d['scene'][d['fit_mask']])
    x = torch.tensor(np.concatenate(scene), dtype=torch.float32)
    with torch.no_grad():
        normalized = (x - model.temporal.mean) / model.temporal.scale
        logits = model.visual(torch.cat([torch.zeros(len(x), 12), x], dim=1))
        ports = logits.sigmoid()
    np.savez_compressed(run / 'range.npz', visual_logits=logits.numpy(), visual_ports=ports.numpy(),
                        temporal_unclipped=normalized.numpy())
    groups = normalized.reshape(-1, 4, 9, 16)
    write_json(run / 'result.json', dict(status='completed_train_visual_range_audit',
        samples=len(x), temporal_clipped_fraction_by_view=[float((groups[:, i, 1:].abs() >= 5).float().mean()) for i in range(4)],
        visual_sigmoid_extreme_fraction=float(((ports < .01) | (ports > .99)).float().mean()),
        visual_sigmoid_derivative_quantiles=np.quantile((ports * (1-ports)).numpy(), [0, .05, .5, .95, 1]).tolist(),
        visual_port_training_std=ports.std(0).tolist(),
        visual_logits_quantiles=np.quantile(logits.numpy(), [0, .05, .5, .95, 1]).tolist(),
        arrays_sha256=sha256(run / 'range.npz'), optimizer_updates=0, vehicle_control_sent=False,
        scope='Train-only range audit; dynamic traces, physiological responses and causal driving contribution not established',
        termination_reason='Exact static input transformations on both train laps measured'))


if __name__ == '__main__':
    main()

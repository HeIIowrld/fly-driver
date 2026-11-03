"""Keep the original full auxiliary gate separate from component-level sensory evidence."""

import argparse
from pathlib import Path

import numpy as np

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from scripts.audit_exterior_preview_learning import auxiliary_gate, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--training', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan, root = read(a.plan), Path(a.training)
    data = Path(plan['data'])
    source, result = read(root/'manifest.json')['config'], read(root/'result.json')
    if (source['plan_sha256'] != sha256(a.plan)
            or source['dataset_sha256'] != sha256(data/'dataset.json')
            or source['program_sha256'] != plan['source_sha256'][
                'scripts/train_body_registered_preview.py']
            or result['status'] != 'completed_body_registered_preview_learning'
            or plan['preregistered_candidate'] != 'registered'
            or {r['condition'] for r in result['records']} != set(plan['conditions'])
            or any(r['epochs'] != 200 for r in result['records'])
            or len({r['initial_parameter_sha256'] for r in result['records']}) != 1
            or len({r['parameter_count'] for r in result['records']}) != 1):
        raise ValueError('Matched preregistered representations and final200 required')
    episodes = read(data/'dataset.json')['episodes']
    if sorted(e['role'] for e in episodes) != ['development', 'train', 'train', 'train']:
        raise ValueError('Every whole fixed-role episode required')
    dev = next(e['id'] for e in episodes if e['role'] == 'development')
    reports = {(r['condition'], r['intervention']): r['full'] for r in result['reports']
               if r['episode'] == dev and r['role'] == 'development'}
    scale = np.asarray(result['train_target_scale'])
    gate = auxiliary_gate(reports['registered', 'intact'],
                          reports['registered', 'neural_feature_mean'], scale)
    retinal_gate = auxiliary_gate(reports['registered', 'intact'],
                                  reports['registered', 'retinal_neural_mean'], scale)
    scores = {c+'/'+v: float(np.mean([(h['rmse']/s)**2 for h, s in zip(heads, scale, strict=True)]))
              for (c, v), heads in reports.items()}
    actual = scores['registered/intact']
    neural_feature_effect = actual < .99*scores['registered/neural_feature_mean']
    retinal_effect = retinal_gate['vision_contribution'] and actual < .99*scores[
        'registered/retinal_neural_mean']
    camera_effect = {v: scores['registered/'+v+'_neural_mean'] > actual*1.01
                     for v in ('front', 'left', 'right', 'rear')}
    beats_image = actual < scores['image_grid/intact']
    posture = actual < scores['registered_level/intact']
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        training_result_sha256=sha256(root/'result.json'), plan_sha256=sha256(a.plan),
        dataset_sha256=sha256(data/'dataset.json'), no_fitting=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_body_registered_preview_audit',
        original_six_head_gate=gate, full_auxiliary_candidate_passed=(
            gate['auxiliary_gate_passed'] and neural_feature_effect
            and retinal_effect and beats_image),
        standardized_mse=scores, component_evidence=dict(
            neural_feature_effect=neural_feature_effect,
            retinal_response_effect=retinal_effect, beats_matched_image_grid=beats_image,
            posture_registration_effect=posture, camera_neural_effects=camera_effect,
            multi_view_neural_effect=camera_effect['front'] and any(camera_effect[v] for v in (
                'left', 'right', 'rear'))), target_scales=scale.tolist(),
        known_development_diagnostic_only=True, physiological_function_verified=False,
        motor_transfer_qualified=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()

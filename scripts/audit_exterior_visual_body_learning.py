"""Report fixed sensory-fusion gates without fitting or granting driving ownership."""

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
    p.add_argument('--visual-audit', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plan, root, reference = read(a.plan), Path(a.training), Path(a.visual_audit)
    result, source = read(root/'result.json'), read(root/'manifest.json')['config']
    dataset = Path(plan['visual_data'])/'dataset.json'
    proof = read(reference/'result.json')
    if (source['plan_sha256'] != sha256(a.plan)
            or source['source_sha256'] != plan['source_sha256']['scripts/train_visual_body_preview.py']
            or read(reference/'manifest.json')['config']['dataset_sha256'] != sha256(dataset)
            or plan['preregistered_candidate'] != 'vision_body'
            or result['status'] != 'completed_visual_body_preview_comparison'
            or any(len(r['losses']) != 200 for r in result['records'])):
        raise ValueError('Declared final200 paired whole-history experiment required')
    episodes = read(dataset)['episodes']
    if sorted(e['role'] for e in episodes) != ['development', 'train', 'train', 'train']:
        raise ValueError('Original geographic roles required')
    expected = {e['id']: (e['role'], e['features_sha256'], e['targets_sha256']) for e in episodes}
    actual = {e['id']: (e['role'], e['visual_sha256'], e['targets_sha256'])
              for e in source['original_sources']}
    if actual != expected:
        raise ValueError('Whole visual/body run source identity differs')
    dev = next(e['id'] for e in episodes if e['role'] == 'development')
    scale = np.asarray(proof['train_only_target_scale'])
    reports = {(r['condition'], r['intervention']): r['full'] for r in result['reports']
               if r['episode'] == dev and r['role'] == 'development'}
    intact = reports[('vision_body', 'intact')]
    gate = auxiliary_gate(intact, reports[('vision_body', 'vision_absent')], scale)
    scores = {condition+'/'+intervention: float(np.mean([
        (h['rmse']/s)**2 for h, s in zip(heads, scale, strict=True)]))
        for (condition, intervention), heads in reports.items()}
    candidate = scores['vision_body/intact']
    body_contribution = candidate < scores['vision_body/body_absent']
    beats_single = all(candidate < scores[c+'/intact'] for c in ('vision_only', 'body_only'))
    run = begin_run(a.output, dict(arguments=vars(a), program_sha256=sha256(__file__),
        training_result_sha256=sha256(root/'result.json'), dataset_sha256=sha256(dataset),
        scale_reference_sha256=sha256(reference/'result.json'), no_fitting=True))
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'result.json', dict(status='completed_exterior_sensory_fusion_audit',
        visual_gate=gate, body_contribution=body_contribution,
        beats_both_matched_single_inputs=beats_single,
        fusion_auxiliary_gate_passed=gate['auxiliary_gate_passed'] and body_contribution and beats_single,
        standardized_mse=scores, development=dev, development_used_to_fit=False,
        flight_circuit_integration=False, physiological_function_verified=False,
        motor_transfer_qualified=False, actor_deployed=False, goal_achieved=False))


if __name__ == '__main__':
    main()

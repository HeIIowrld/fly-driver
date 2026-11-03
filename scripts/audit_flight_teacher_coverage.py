"""Count complete sealed teacher histories without changing labels, splits, times or models."""

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.dataset)
    source = json.loads((root/'dataset.json').read_text(encoding='utf8'))
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        dataset_manifest_sha256=sha256(root/'dataset.json'),
        signed_left_intent_rule='negative normalized applied steering',
        turn_threshold=.01, positive_pedal_threshold=.05,
        source_role='known historical source; no new blind evaluation claim'))
    result = dict(status='failed', goal_achieved=False, actuator_messages_sent=0,
                  ui_actions_sent=0, optimizer_updates=0, episodes=[])
    try:
        assert source['schema'] == 'flybrain.multiview.teacher.v1'
        assert source['training_admissible'] and source['labels_are_executed_applied_controls']
        assert source['actual_observation_clocks_preserved']
        for entry in source['episodes']:
            file = root/entry['file']
            assert sha256(file) == entry['sha256']
            with load_array(file, allow_pickle=False) as arrays:
                action, valid = arrays['action'], arrays['action_valid']
                clocks = arrays['observation_time']
                assert action.shape == valid.shape == (len(clocks), 3) and valid.dtype == bool
                assert len(clocks) == entry['steps'] and np.isfinite(clocks).all()
                assert np.all(np.diff(clocks) > 0) and np.isfinite(action[valid]).all()
                assert np.all(np.abs(action[valid]) <= 1)
                assert np.all(action[:, 1:][valid[:, 1:]] >= 0)
                left, mask = -action[:, 0], valid[:, 0]
                counts = dict(left=int(((left > .01) & mask).sum()),
                    right=int(((left < -.01) & mask).sum()),
                    neutral=int(((np.abs(left) <= .01) & mask).sum()),
                    throttle_gt005=int(((action[:, 1] > .05) & valid[:, 1]).sum()),
                    brake_gt005=int(((action[:, 2] > .05) & valid[:, 2]).sum()),
                    brake_gt0=int(((action[:, 2] > 0) & valid[:, 2]).sum()))
                assert counts['left']+counts['right']+counts['neutral'] == int(mask.sum())
                result['episodes'].append(dict(file=str(file), sha256=entry['sha256'],
                    observations=len(clocks), supervised_heads=valid.sum(0).tolist(),
                    counts=counts, original_intervals_above80ms=int((np.diff(clocks) > .08).sum()),
                    maximum_original_interval_seconds=float(np.diff(clocks).max()),
                    valid_action_min=[float(action[valid[:, k], k].min())
                                      if valid[:, k].any() else None for k in range(3)],
                    valid_action_max=[float(action[valid[:, k], k].max())
                                      if valid[:, k].any() else None for k in range(3)],
                    camera_aid_state_verified=entry.get('camera_aid_state_verified', False)))
        assert result['episodes']
        result.update(status='completed_sealed_teacher_coverage_audit',
            counts={key: sum(e['counts'][key] for e in result['episodes'])
                    for key in result['episodes'][0]['counts']},
            camera_aid_state_verified=source.get('camera_aid_state_verified', False),
            teacher_label_contract=source['teacher_label_contract'],
            labels_changed=0, observations_removed=0, splits_changed=False,
            steering_or_braking_competence_proven=False)
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
        write_json(run/'result.json', result)
        raise
    write_json(run/'result.json', result)
    counts = result['counts']
    lines = ['# 기존 교사 자료의 표본 범위', '',
        '원래 전체 이력과 라벨/유효성/시각/분할을 바꾸지 않은 집계다.',
        '이미 알려진 과거 자료이며 새 맹검 평가나 모델 학습이 아니다.', '',
        '| 항목 | 유효 표본 |', '|---|---|',
        *[f'| {key} | {value} |' for key, value in counts.items()], '',
        '좌/우는 원래 정규화 applied steering의 부호를 뒤집은 의도 ±0.01 기준이다.',
        '페달은 원래 [0,1] 값에서0.05 초과이며 brake_gt0도 함께 보고한다.',
        ('이 자료에는 유효한 제동 양성 표본이 없다. '
         '일반적인 제동 동작을 배웠다는 근거가 되지 않는다.'
         if counts['brake_gt0'] == 0 else
         '제동 양성 표본 수는 실제 제동 능력이나 미사용 조건의 일반화 검증을 뜻하지 않는다.'),
        '실제 AI 내부 명령이나 최적 레이싱 라인 정답을 측정한 자료라는 뜻도 아니다.', '',
        '[원본 해시와 전체 결과](result.json).']
    (run/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()

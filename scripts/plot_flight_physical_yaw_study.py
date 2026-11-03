"""Plot a completed audited study from original native samples; no simulator access."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.study)
    study = read(source/'result.json')
    assert study['status'] == 'completed_physical_yaw_study' and study['full_study_complete']
    assert len(study['blocks']) == 3 and len(study['pairs']) == 6
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
                                    source_sha256=sha256(source/'result.json')))
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    provenance = []
    trial_results = []
    for entry in study['blocks']:
        audit_root = Path(entry['audit'])
        assert sha256(audit_root/'result.json') == entry['audit_sha256']
        audit = read(audit_root/'result.json')
        block = read(Path(entry['original'])/'result.json')
        for audited in audit['audits']:
            result_path = Path(audited['path'])/'result.json'
            assert sha256(result_path) == audited['result_sha256']
            result = read(result_path)
            trial_results.append(result)
            trial = next(t for t in block['trials'] if t['index'] == audited['index'])
            assert sha256(Path(trial['path'])/'result.json') == trial['result_sha256']
            receipts_path = Path(trial['path'])/'native-receipts.json'
            receipts = read(receipts_path)
            times = np.array([r['received_monotonic_seconds'] for r in receipts])
            yaw = np.array([r['fields']['localAngularVel'][1] for r in receipts])
            assert np.all(np.diff(times) > 0)
            on, off = result['pulse_on_host_clock'], result['pulse_off_host_clock']
            relative = times-on  # Display axis only; original clocks remain in the archive.
            selected = (relative >= -.3) & (times <= off+1.6)
            error = yaw-result['physical_metrics']['baseline_yaw_left_rad_s']
            sign, mode = result['trial']['sign'], result['trial']['mode']
            ax = axes[entry['index'], 0 if sign == 1 else 1]
            ax.plot(relative[selected], error[selected], label=mode,
                    color='#2563eb' if mode == 'full' else '#dc2626', linewidth=1.3)
            provenance.append(dict(index=audited['index'], result=str(result_path),
                                   result_sha256=sha256(result_path),
                                   native=str(receipts_path), native_sha256=sha256(receipts_path)))
        for col, sign in enumerate((1, -1)):
            pair = next(p for p in study['pairs']
                        if p['block'] == entry['index'] and p['sign'] == sign)
            ax = axes[entry['index'], col]
            ax.axvspan(0, .28, color='grey', alpha=.12, label='nominal pulse')
            ax.axhline(0, color='black', linewidth=.5)
            ax.set_title(f"Block {entry['index']} / pulse {sign:+d}: "
                         f"IAE change {100*pair['iae_improvement_fraction']:+.2f}% "
                         f"({'valid' if pair['pair_valid'] else 'invalid pair'})")
            ax.grid(alpha=.2)
            ax.set_ylabel('Yaw minus own baseline [rad/s]')
            if entry['index'] == 2:
                ax.set_xlabel('Original host time minus pulse-on [s]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Physical yaw feedback: full vs zero_yaw, 6 km/h engineering pedals\n'
                 'Original native samples; no resampling or curve fitting', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(run/'physical-yaw-responses.png', dpi=160)
    plt.close(fig)
    passed = study['physical_yaw_disturbance_iae_gate_passed']
    lines = [
        '# 실제 회전 감각 피드백 비교 — 2026-10-29', '',
        f"사전 IAE 기준: **{'통과' if passed else '미통과'}**. "
        '전체 목표와 운전 후보는 미완료/미승격이다.', '',
        f"독립 시험 {len(trial_results)}회, "
        f"원본 관측 {sum(r['observations'] for r in trial_results):,}개를 감사했다. "
        f"유효 짝 {sum(p['pair_valid'] for p in study['pairs'])}/6, "
        f"IAE 감소 짝 {study['positive_iae_pairs']}/6, "
        f"개선율 중앙값 {100*study['median_iae_improvement_fraction']:.3f}%다.", '',
        '기준은 3블록·모든 짝 유효·6짝 모두 IAE 감소·중앙값 1% 이상이다. '
        '결과를 보고 바꾸지 않았다.', '',
        '| 블록 | 교란 부호 | full IAE (rad) | yaw0 IAE (rad) | 개선율 | 짝 유효 |',
        '|---|---|---|---|---|---|',
    ]
    for pair in study['pairs']:
        lines.append(f"| {pair['block']} | {pair['sign']:+d} | "
                     f"{pair['full_metrics']['iae_rad']:.8f} | "
                     f"{pair['zero_yaw_metrics']['iae_rad']:.8f} | "
                     f"{100*pair['iae_improvement_fraction']:.3f}% | {pair['pair_valid']} |")
    lines.extend(['', '회복 시간은 원래 표본으로 확인한 상한이다. '
        'IAE 개선이 모든 지표 개선을 뜻하지 않는다.',
        '', '| 블록 | 교란 | full 회복 상한 s | yaw0 회복 상한 s |', '|---|---|---|---|'])
    for pair in study['pairs']:
        recovery = [pair[key]['observed_recovery_seconds_upper']
                    for key in ('full_metrics', 'zero_yaw_metrics')]
        values = ['미관측' if value is None else f'{value:.3f}' for value in recovery]
        lines.append(f"| {pair['block']} | {pair['sign']:+d} | {values[0]} | {values[1]} |")
    lines.extend(['', '![원본 물리 응답](physical-yaw-responses.png)', '',
        '곡선은 각 시험의 원래 native 표본에서 자체 baseline을 뺀 값이다. '
        '표시 축만 pulse-on 기준 상대 시간이며, '
        '재표본화·곡선 적합·원본 시각 변경은 없다. 회색은 명목 280ms 구간이다.', '',
        '**범위:** 6km/h 시험용 페달과 작은 양방향 조향 교란이다. 몸 yaw 입력 제거와의 비교이며, '
        '초파리 회로만의 우월성·생리 재현·영상 경로 추종·학습한 가감속·유효 완주·'
        '타이어/셋업·최적 랩타임을 증명하지 않는다.', '',
        f"[원본 집계](<{(source/'result.json').resolve().as_posix()}>).", '',
        '최대 의도 재생 오차: '
        f"{max(r['intent_max_error'] for r in trial_results):.8g}. "
        f"80ms 초과 원본 간격 합계: {sum(r['intervals_above_80ms'] for r in trial_results)}.",
    ])
    if study.get('preserved_failed_study'):
        preserved = study['preserved_failed_study']
        lines.extend(['', f"앞선 실패 연구의 완료 블록 {preserved['preserved_blocks']}개/"
            f"실제 시험 {preserved['preserved_actual_trials']}회를 "
            '전체 감사·모든 짝과 함께 보존했다.',
            '준비에서 실패한 나머지 블록만 새 독립 에피소드로 수행했다. 실패를 삭제하거나 '
            '유리한 결과만 골라 상속하지 않았다.',
            '[보존한 실패 연구]('
            f"<{(Path(preserved['path'])/'result.json').resolve().as_posix()}>)."])
    plan = read(Path(study['blocks'][0]['original'])/'manifest.json')['config']['plan']
    if plan.get('fast_stream_config'):
        config = read(plan['fast_stream_config'])
        lines.extend(['', f"빠른 몸 경로 가설: `{config['hypothesis']}`. "
            '실측 화학 접촉을 바탕으로 가정한 전기 결합/시간 상수의 부품 시험이다. '
            '전기 전도도·전압·스파이크·날갯짓 위상의 생리 재현을 입증하지 않는다.',
            '원래 회로의 이전 미통과 연구와 이 새 동역학의 시험은 구분한다.'])
    (run/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    write_json(run/'result.json', dict(status='plotted_completed_study', goal_achieved=False,
        actuator_messages_sent=0, ui_actions_sent=0, original_samples_not_resampled=True,
        native_provenance=provenance, image_sha256=sha256(run/'physical-yaw-responses.png'),
        physical_yaw_disturbance_iae_gate_passed=passed,
        markdown_sha256=sha256(run/'summary.md')))


if __name__ == '__main__':
    main()

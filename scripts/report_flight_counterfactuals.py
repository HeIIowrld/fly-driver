"""Read-only plots and Markdown for the completed original-history decomposition."""

import argparse
import json
from pathlib import Path

import matplotlib

from flybrain_drive.episode_storage import load_array

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnosis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.diagnosis)
    result = read(source/'result.json')
    assert result['status'] == 'completed_original_history_decomposition'
    assert result['trials_completed'] == 6 and result['optimizer_updates'] == 0
    run = begin_run(args.output, dict(arguments=vars(args),
        source_result_sha256=sha256(source/'result.json'), scope=__doc__))
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    table = ['| 블록 | 교란 | 몸 효과 RMS | 시각 명령 효과 RMS | 몸 조향 변화 최대/교란 | '
             '동시점 억제 상관 | 최대 상관의 실제 중앙 지연 |',
             '|---|---|---|---|---|---|---|']
    maxima, delays, errors = [], [], []
    for entry in result['trials']:
        root = Path(entry['path'])
        assert sha256(root/'result.json') == entry['result_sha256']
        audit = read(root/'result.json')
        analysis = entry['analysis']
        assert (sha256(root/'counterfactuals.npz')
                == audit['counterfactual_diagnosis']['archive_sha256'])
        with load_array(root/'counterfactuals.npz', allow_pickle=False) as data:
            time, yaw, intents = data['original_clock'], data['yaw'], data['intents']
        on, off = audit['pulse_on_host_clock'], audit['pulse_off_host_clock']
        baseline = (time >= on-.3) & (time < on)
        view = (time >= on-.3) & (time <= off+1.6)
        body = intents[:, 0, 0]-intents[:, 1, 0]
        no_scene = intents[:, 3, 0]-intents[:, 4, 0]
        ax = axes[analysis['trial']['block'], 0 if analysis['trial']['sign'] == 1 else 1]
        ax.plot((time-on)[view], (body-body[baseline].mean())[view], color='#2563eb',
                label='Body effect, original scene', linewidth=1.4)
        ax.plot((time-on)[view], (no_scene-no_scene[baseline].mean())[view], color='#16a34a',
                label='Body effect, scene command lesioned', linestyle='--', linewidth=1)
        ax.axvspan(0, .28, color='grey', alpha=.12)
        ax.axhline(0, color='black', linewidth=.5)
        ax.set_title(f"Block {analysis['trial']['block']} / pulse {analysis['trial']['sign']:+d}")
        ax.set_ylabel('Left intent difference')
        ax.grid(alpha=.2)
        other = ax.twinx()
        other.plot((time-on)[view], (yaw-yaw[baseline].mean())[view], color='#ea580c',
                   label='Measured yaw minus baseline', alpha=.6, linewidth=1)
        other.set_ylabel('Yaw [rad/s]', color='#c2410c')
        if analysis['trial_index'] == 0:
            lines = ax.get_lines()[:2]+other.get_lines()
            ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc='upper right')
        if analysis['trial']['block'] == 2:
            ax.set_xlabel('Original host time minus pulse-on [s]')
        candidates = [lag for lag in analysis['lagged_correlations']
                      if lag['opposition_correlation'] is not None]
        best = max(candidates, key=lambda lag: lag['opposition_correlation'])
        delay = best['actual_lag_seconds_percentiles'][1]
        fraction = analysis['maximum_body_steering_change_fraction_of_pulse']
        maxima.append(fraction)
        delays.append(delay)
        errors.append(analysis['full_duplicate_max_error'])
        table.append(f"| {analysis['trial']['block']} | {analysis['trial']['sign']:+d} | "
            f"{analysis['pulse_body_intent_rms']:.7f} | "
            f"{analysis['pulse_scene_intent_rms']:.7f} | {100*fraction:.3f}% | "
            f"{analysis['lagged_correlations'][0]['opposition_correlation']:.3f} | "
            f"{1000*delay:.1f}ms |")
    figure.suptitle('Original-history body contribution and measured yaw\n'
                     'Input interventions only; no counterfactual vehicle trajectory', fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(run/'body-feedback-decomposition.png', dpi=160)
    plt.close(figure)
    lines = ['# 원래 이력의 몸·시각 명령 분해 — 2026-10-29', '',
        '3블록의 양방향 full 조건 6회를 모두 사용했다. 학습 가중치·차량 입력 변경은0이다. '
        '원래 조건의 복제 출력과 실시간 기준 출력의 최대 차이는 '
        f'{max(errors):.8g}다.', '', *table, '',
        'RMS와 조향 변화는 pulse-on 전300ms 평균을 뺀 뒤, '
        'pulse-on부터 실제 pulse-off+1.6초까지 계산했다. '
        '몸 조향 변화는 고정 차량 변환0.6을 적용하고 외부 교란0.05에 대한 비율로 표시했다.', '',
        '시각 명령 제거는 기존 scene_command 개입이다. '
        '정규화된 장면 증거를0으로 하고 학습 bias는 남긴다. '
        '실제 망막/시각 뉴런 제거와 동일하다고 주장하지 않는다.', '',
        '지연 표는 미리 정한9개 지연의 상관 중 최대값 위치를 **탐색적으로** 요약한 것이다. '
        '과거의 실제 관측만 선택했고 보간하지 않았다. 정확한 신경 전달 지연이나 인과적 전달함수의 '
        '측정값으로 해석하지 않는다. 모든 지연/실제 시간 차이는 원본 JSON에 보존했다.', '',
        '![몸 피드백과 원래 회전](body-feedback-decomposition.png)', '',
        '파랑/초록은 조향 의도 단위, 주황은 rad/s로 서로 다른 축이다. 원래 표본 그대로 표시했다. '
        '장면 개입을 해도 원래 영상·몸 궤적을 유지하므로 새로운 물리 주행 결과가 아니다.', '',
        '**판정:** 앞선 실제 안정화 시험4/6 개선·사전 기준 미통과는 유지한다. '
        '이 진단으로 생리 기능·실제 안정화·영상 자율주행·학습한 가감속·완주를 '
        '통과 처리하지 않는다.', '',
        '다음은 측정된 몸 회전 구동의 크기와 주파수별 지연을 '
        '원래 rate 회로의 통제 자극으로 확인하는 것이다. '
        '뉴런/몸 대응이나 생리 시간 상수를 결과에 맞춰 바꾸지 않는다.', '',
        f'[분해 원본](../../{source.as_posix()}/result.json).',
    ]
    (run/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    write_json(run/'result.json', dict(status='completed_decomposition_report', goal_achieved=False,
        actuator_messages_sent=0, ui_actions_sent=0, optimizer_updates=0,
        full_duplicate_maximum_error=max(errors),
        body_steering_fraction_of_pulse_range=[min(maxima), max(maxima)],
        exploratory_peak_correlation_actual_lag_range_seconds=[min(delays), max(delays)],
        physical_stability_repaired=False, markdown_sha256=sha256(run/'summary.md'),
        image_sha256=sha256(run/'body-feedback-decomposition.png')))


if __name__ == '__main__':
    main()

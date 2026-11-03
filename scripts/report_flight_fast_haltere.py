"""Contact-current prototype evidence: hypotheses, calibration and original histories."""

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
    for argument in ('probe', 'reference', 'numerical', 'histories', 'output'):
        parser.add_argument('--'+argument, required=True)
    args = parser.parse_args()
    raw = read(Path(args.probe)/'result.json')
    reference = read(Path(args.reference)/'result.json')
    numerical = read(Path(args.numerical)/'result.json')
    histories = read(Path(args.histories)/'result.json')
    assert raw['status'] == 'completed_fast_haltere_hypothesis_probe'
    assert reference['status'] == 'completed_relative_motor_reference_calibration'
    assert numerical['all_cases_passed']
    assert histories['status'] == 'completed_fast_original_history_audits'
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
        input_result_hashes={key: sha256(Path(getattr(args, key))/'result.json')
                            for key in ('probe', 'reference', 'numerical', 'histories')}))
    plan = read(Path(args.probe)/'manifest.json')['config']['plan']
    names = [row['name'] for row in plan['hypotheses']]
    figure, axes = plt.subplots(3, 1, figsize=(10, 10))
    table = ['| 가설 | 주파수 | 억제 의도 이득/(rad/s) | 위상 | 기준 입력 조향 의도: 전→후 |',
             '|---|---|---|---|---|']
    before_neutral, after_neutral = [], []
    for name in names:
        rows = [r for r in reference['cases'] if r['hypothesis'] == name
                and r['amplitude_rad_s'] == .02 and r['cadence_ticks'] == 4]
        original = [r for r in raw['cases'] if r['hypothesis'] == name
                    and r['amplitude_rad_s'] == .02 and r['cadence_ticks'] == 4]
        frequencies = [r['frequency_hz'] for r in rows]
        phases = np.rad2deg(np.unwrap(np.deg2rad(
            [r['opposing_intent_response']['phase_degrees'] for r in rows])))
        axes[0].loglog(frequencies, [r['gain_per_ideal_yaw'] for r in rows], marker='o', label=name)
        axes[1].semilogx(frequencies, phases, marker='o', label=name)
        before_neutral.append(original[0]['neutral_intent'][0])
        after_neutral.append(rows[0]['neutral_intent'][0])
        for row, phase in zip(rows, phases, strict=True):
            table.append(f"| {name} | {row['frequency_hz']:g}Hz | "
                f"{row['gain_per_ideal_yaw']:.5f} | {phase:.2f}° | "
                f"{original[0]['neutral_intent'][0]:+.5f} → {row['neutral_intent'][0]:+.5f} |")
    axes[0].set_ylabel('Opposing intent gain [per rad/s]')
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel('Opposing intent phase [degrees]')
    axes[1].axhline(-90, color='grey', linestyle='--', linewidth=.8)
    axes[1].set_xlabel('Synthetic yaw frequency [Hz]')
    for ax in axes[:2]:
        ax.grid(alpha=.2)
        ax.set_xticks([.5, 1, 2, 4, 8, 12], labels=['0.5', '1', '2', '4', '8', '12'])
    x = np.arange(len(names))
    axes[2].bar(x-.18, before_neutral, .36, label='Before motor reference')
    axes[2].bar(x+.18, after_neutral, .36, label='After relative motor reference')
    axes[2].set_xticks(x, labels=names, rotation=12)
    axes[2].set_ylabel('Left intent at fixed reference input')
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='y', alpha=.2)
    figure.suptitle('Contact-constrained fast haltere hypotheses: model-only\n'
                     '0.02 rad/s, 40 ms sensing; original visual policy bias retained')
    figure.tight_layout()
    figure.savefig(run/'fast-haltere-hypotheses.png', dpi=160)
    plt.close(figure)
    maximum_state_error = max(row['original_neuron_max_error'] for row in histories['trials'])
    zero_error = max(row['old_zero_yaw_intent_max_error'] for row in histories['trials'])
    lines = ['# 빠른 haltere 경로 후보 — 2026-10-29', '',
        '현재 연결망의 원래47감각뉴런과 좌·우 b1 운동뉴런, 총49개 상태를 사용했다. '
        '원래51개 화학 접촉이 있는 위치에만 대칭 전기 전류를 가정했다. '
        '화학 연결 수와 뉴런 ID는 유지했고 전기 전도도·5ms 시간 상수는 **미검증 가설**이다.', '',
        '원래 조건, 빠른 세포만, 결합0.1/0.3/1의5가설 ×24조건을 모두 보고한다. '
        '직접 차량 조향 보정이나 가중치 학습은 없다.', '',
        '원래 조건 대조: 주파수 실험 전체 뉴런 최대 오차 '
        f"{raw['full_neuron_baseline_max_error']:.8g}, 이전 출력 차이 "
        f"{raw['prior_frequency_intent_max_error']:.8g}다.", '',
        '중립 보정은 yaw0·외부 시각/조향 명령 주입0의 운동뉴런 기준을 원래 모델 기준에 맞췄다. '
        '시각·명령 정책, 몸 encoder, readout 이득/혼합·추진 기준은 유지했다. '
        '기존 조향 편향을0으로 만드는 보정이 아니다. 실제 뉴런 readout과 계산의 차이는 '
        f"{reference['actual_readout_equivalence_max_error']:.8g}다.", '',
        '1ms→0.5ms 수치 비교120조건 모두 사전 기준 통과: 최대 크기 차이 '
        f"{100*numerical['maximum_relative_gain_difference']:.4f}%, 위상 차이 "
        f"{numerical['maximum_phase_difference_degrees']:.4f}°, 기준 의도 차이 "
        f"{numerical['maximum_neutral_intent_difference']:.8g}다.", '',
        '원래6회·1,542관측을 모두 같은 영상·native bytes·시각으로 재생했다. '
        f'원래 조건의 전체 뉴런 최대 오차 {maximum_state_error:.8g}, '
        f'이전 yaw0 출력 대조 최대 오차 {zero_error:.8g}다. '
        '각 이력에서5가설×full/yaw0의10상태를 유지했다. '
        '15,420개 가상 출력은 새 실제 관측이 아니다.', '',
        '아래 표/그림은 진폭0.02rad/s·40ms 감각 갱신의 대표30조건이다. '
        '전체120조건은 원본 JSON에 있다.', '',
        *table, '', '![후보 비교](fast-haltere-hypotheses.png)', '',
        '**현재 판정:** 신경 회로 부품 후보 구현과 오프라인 검증이다. '
        '전기 연결/시간 상수의 생리 재현·새 실시간 스트림·실제 안정화·영상 경로 판단·학습한 가감속·'
        '유효 완주와 랩타임은 검증 전이다. '
        '원래 실제 안정화4/6 개선·기준 미통과와 후보 미승격은 유지한다.', '',
        '다음은 후보별 계약을 가진 별도 실시간 스트림 구현과 원래 이력 전체 대조, '
        '실시간 비용·상태 지속성 확인 후 새 실제 비교를 사전 선언하는 것이다. '
        '현재 GPU 부품 결과만으로 차량에 배포하지 않는다.', '',
    ]
    for key in ('probe', 'reference', 'numerical', 'histories'):
        lines.append(f"[{key} 원본](../../{Path(getattr(args, key)).as_posix()}/result.json).")
    (run/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    write_json(run/'result.json', dict(status='completed_fast_haltere_component_report',
        goal_achieved=False, actor_qualified=False, actuator_messages_sent=0, optimizer_updates=0,
        original_histories=6, original_observations=1542,
        original_neuron_max_error=maximum_state_error,
        previous_zero_yaw_intent_max_error=zero_error, physical_stability_repaired=False,
        image_sha256=sha256(run/'fast-haltere-hypotheses.png'),
        markdown_sha256=sha256(run/'summary.md')))


if __name__ == '__main__':
    main()

"""Plot the controlled model-only frequency response without simulator access."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.probe)
    result = json.loads((source/'result.json').read_text(encoding='utf8'))
    assert result['status'] == 'completed_model_yaw_frequency_probe'
    assert (result['parameter_versions_unchanged']
            and result['nonpersistent_csr_cache_values_verified'])
    assert sha256(source/'model-frequency-responses.npz') == result['archive_sha256']
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__,
                                    source_result_sha256=sha256(source/'result.json')))
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    table = ['| 진폭(rad/s) | 감각 갱신(ms) | 주파수(Hz) | 0.5Hz 대비 크기 | '
             '이상 회전 대비 위상(도) | 실제 유지 입력 대비 위상(도) | 적합 R² |',
             '|---|---|---|---|---|---|---|']
    for amplitude in (.005, .02):
        for cadence in (1, 4):
            rows = [r for r in result['cases'] if r['amplitude_rad_s'] == amplitude
                    and r['cadence_ticks'] == cadence]
            assert len(rows) == 6
            frequency = [r['frequency_hz'] for r in rows]
            label = f'{amplitude:g} rad/s, update {10*cadence} ms'
            axes[0].semilogx(frequency, [r['gain_relative_to_half_hz'] for r in rows],
                            marker='o', label=label)
            axes[1].semilogx(frequency, [r['phase_to_ideal_unwrapped_degrees'] for r in rows],
                            marker='o', label=label)
            for row in rows:
                table.append(f"| {amplitude:g} | {10*cadence} | {row['frequency_hz']:g} | "
                    f"{row['gain_relative_to_half_hz']:.4f} | "
                    f"{row['phase_to_ideal_unwrapped_degrees']:.2f} | "
                    f"{row['phase_relative_to_held_degrees']:.2f} | "
                    f"{row['opposing_intent_response']['r_squared']:.5f} |")
    axes[0].set_ylabel('Gain relative to 0.5 Hz')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Opposing intent phase vs ideal yaw [degrees]')
    axes[1].set_xlabel('Synthetic yaw frequency [Hz]')
    axes[1].axhline(-90, color='grey', linestyle='--', linewidth=.8)
    for ax in axes:
        ax.grid(alpha=.25)
        ax.set_xticks([.5, 1, 2, 4, 8, 12], labels=['0.5', '1', '2', '4', '8', '12'])
    figure.suptitle('Fixed flight rate circuit: controlled frequency response\n'
                     'Model-only synthetic stimuli, no actuator or weight updates')
    figure.tight_layout()
    figure.savefig(run/'yaw-frequency-response.png', dpi=160)
    plt.close(figure)
    text = ['# 고정 비행 회로의 회전 주파수 반응 — 2026-10-29', '',
        '24조건을 동시에 측정했다. 회로 dt10ms·공통 시간 상수50ms와 원래 가중치는 유지했다. '
        '감각 갱신10/40ms·진폭0.005/0.02rad/s·주파수0.5~12Hz를 사용했다.', '',
        '이는 새 합성 모델 시간의 실험이다. 실제 AC 관측이나 생리 반응으로 기록하지 않는다. '
        '원래 body baseline·유효성은 유지하고 yaw에만 합성 자극을 더했다. '
        '장면은 학습 정규화 평균으로 고정했다.', '',
        '2초 정착 후20초 자극 중 마지막8초의 기본파를 최소제곱으로 측정했다. '
        '출력은 회전을 억제하는 방향의 의도(-left intent)다. '
        '위상은 주파수 순서대로 unwrap했으며 '
        '위상 기준은 이상 사인과 실제 sample-hold 입력을 구분한다.', '',
        *table, '', '![주파수 반응](yaw-frequency-response.png)', '',
        '**판정:** 느린 rate 모델의 대역폭/위상 문제를 확인하는 증거다. '
        '실제 안정화 실패의 유일한 원인이나 초파리 생리 시간 상수를 확정하지 않는다. '
        '가중치·몸 경계·원본 그래프와 CSR 캐시 값은 불변을 확인했다. '
        '이 결과로 게인을 늘리거나 기존 물리 시험의 판정을 바꾸지 않았다.', '',
        f'[원본](../../{source.as_posix()}/result.json).',
    ]
    (run/'summary.md').write_text('\n'.join(text)+'\n', encoding='utf8')
    write_json(run/'result.json', dict(status='completed_model_frequency_report',
        goal_achieved=False, actuator_messages_sent=0, optimizer_updates=0,
        image_sha256=sha256(run/'yaw-frequency-response.png'),
        markdown_sha256=sha256(run/'summary.md')))


if __name__ == '__main__':
    main()

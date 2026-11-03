"""Compare saved component responses using the pinned author's analysis functions."""

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

from flybrain_drive.episode_storage import load_array
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def load_dataset(directory, config):
    directory = Path(directory)
    receipt = json.loads((directory / "result.json").read_text(encoding="utf8"))
    if receipt["status"] != "completed":
        raise ValueError("Requires terminal successful raw response run")
    path = directory / "responses.npz"
    if sha256(path) != receipt["responses_sha256"]:
        raise ValueError("Response archive hash mismatch")
    stimulus = json.loads((directory / "stimulus.json").read_text(encoding="utf8"))
    with load_array(path, allow_pickle=False) as raw:
        activity = raw["activity"].copy()
        names = raw["central_cell_types"].copy()
        dt = float(raw["dt"])
        valid_frames = (raw['valid_frames'].copy() if 'valid_frames' in raw
                        else np.full(activity.shape[0], activity.shape[1]))
    for index, length in enumerate(valid_frames):
        if not 0 < length <= activity.shape[1]:
            raise ValueError('Invalid finite stimulus length')
        if not np.isfinite(activity[index, :length]).all():
            raise ValueError('Nonfinite neural response inside actual stimulus')
    params = stimulus["parameters"]
    if dt != 0.005 or len(params) != activity.shape[0]:
        raise ValueError("Unexpected stimulus configuration")
    axes = {"angle": "angles", "speed": "speeds", "intensity": "intensities"}
    if stimulus["mode"] == "flash":
        axes = {"radius": "radius", "intensity": "dynamic_range"}
    for parameter, key in axes.items():
        if sorted({row[parameter] for row in params}) != sorted(config[key]):
            raise ValueError(f"Stimulus does not match declared {parameter} grid")
    combinations = [tuple(row[key] for key in axes) for row in params]
    expected_count = int(np.prod([len(config[key]) for key in axes.values()]))
    if len(set(combinations)) != expected_count or len(params) != expected_count:
        raise ValueError("Missing or duplicate stimulus combinations")
    return xr.Dataset(
        {"responses": (("network_id", "sample", "frame", "neuron"), activity[None])},
        coords={
            "network_id": [0],
            "sample": np.arange(activity.shape[0]),
            "frame": np.arange(activity.shape[1]),
            "neuron": np.arange(len(names)),
            "cell_type": ("neuron", names),
            # Exact convention in author's generic_responses, not a shifted peak window.
            "time": ("frame", np.arange(activity.shape[1]).astype(float) * dt - config["t_pre"]),
            **{key: ("sample", [row[key] for row in params]) for key in params[0]},
        },
        attrs={"config": dict(config, dt=dt)},
    )


def numpy_flash_index(dataset, radius):
    mask = (dataset.time.values >= -dataset.attrs["config"]["dt"]) & (
        dataset.time.values <= dataset.attrs["config"]["t_stim"]
    )
    sample = dataset.radius.values == radius
    response = dataset.responses.values[0, sample][:, mask].copy()
    response += np.abs(response.min(axis=(0, 1)))
    intensities = dataset.intensity.values[sample]
    on = response[intensities == 1].max(axis=(0, 1))
    off = response[intensities == 0].max(axis=(0, 1))
    return (on - off) / (on + off + 1e-16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash-run", required=True)
    parser.add_argument("--edge-run", required=True)
    parser.add_argument("--declaration", default="configs/flyvis-component-probe-20261029.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["MPLBACKEND"] = "Agg"
    from flyvis.analysis.flash_responses import flash_response_index
    from flyvis.analysis.moving_bar_responses import (
        direction_selectivity_index,
        get_time_masks,
        peak_responses,
        preferred_direction,
    )
    from flyvis.utils.groundtruth_utils import preferred_directions

    declaration = Path(args.declaration)
    config = json.loads(declaration.read_text(encoding="utf8"))
    source_files = {}
    for function in (flash_response_index, direction_selectivity_index, preferred_direction):
        installed = Path(inspect.getsourcefile(function))
        pinned = (
            Path("assets/external/flyvis-source-20261029/upstream/flyvis/analysis") / installed.name
        )
        if sha256(installed) != sha256(pinned):
            raise ValueError("Installed author analysis differs from pinned source")
        source_files[installed.name] = sha256(installed)
    run = begin_run(
        args.output,
        dict(
            arguments=vars(args),
            seed=None,
            config_sha256=sha256(declaration),
            source_commit=config["source_commit"],
            author_analysis_sha256=source_files,
            inputs={
                kind: {
                    name: sha256(Path(directory) / name)
                    for name in ("result.json", "stimulus.json", "responses.npz")
                }
                for kind, directory in [("flash", args.flash_run), ("edge", args.edge_run)]
            },
            actuator_messages_sent=0,
            optimizer_updates=0,
        ),
    )
    flash = load_dataset(args.flash_run, config["flash"])
    edge = load_dataset(args.edge_run, config["moving_edge"])
    offsets = config['moving_edge']['offsets']
    masks = get_time_masks(edge, offsets[0]*2.25/5.8, (offsets[1]-1)*2.25/5.8)
    if not np.isfinite(edge.responses.where(masks, other=0).values).all():
        raise ValueError('Analysis window intersects invalid neural response')
    types = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
    indices = {name: int(np.flatnonzero(edge.cell_type.values == name)[0]) for name in types}
    fri, parity = {}, {}
    for radius in config["flash"]["radius"]:
        author = flash_response_index(flash.copy(deep=True), radius).values.squeeze()
        independent = numpy_flash_index(flash, radius)
        np.testing.assert_allclose(author, independent, rtol=2e-6, atol=2e-7)
        parity[str(radius)] = float(np.max(np.abs(author - independent)))
        fri[str(radius)] = {name: float(author[index]) for name, index in indices.items()}
    rows = []
    for speed in np.unique(edge.speed.values):
        subset = edge.isel(sample=np.flatnonzero(edge.speed.values == speed))
        peaks = peak_responses(subset).squeeze("network_id")
        dsis = direction_selectivity_index(subset)
        directions = preferred_direction(subset).squeeze("network_id")
        for name, index in indices.items():
            maxima = {
                intensity: float(
                    peaks.isel(neuron=index).where(subset.intensity == intensity, drop=True).max()
                )
                for intensity in (0, 1)
            }
            contrast = (maxima[1] - maxima[0]) / (maxima[1] + maxima[0] + 1e-16)
            expected_intensity = 1 if name.startswith("T4") else 0
            vector = peaks.isel(neuron=index).where(
                subset.intensity == expected_intensity, drop=True
            )
            z = np.sum(vector.values * np.exp(1j * np.deg2rad(vector.angle.values)))
            expected_direction = preferred_directions[name]
            direction = float(
                np.rad2deg(directions.isel(neuron=index).sel(intensity=expected_intensity).item())
                % 360
            )
            independent = float(np.rad2deg(np.angle(z)) % 360)
            error = abs((direction - independent + 180) % 360 - 180)
            if error > 1e-5:
                raise ValueError("Direction metric parity failed")
            rows.append(
                dict(
                    speed=float(speed),
                    cell_type=name,
                    expected_intensity=expected_intensity,
                    edge_contrast_index=contrast,
                    expected_contrast_sign_matches=bool(
                        contrast > 0 if expected_intensity else contrast < 0
                    ),
                    peak_off=maxima[0],
                    peak_on=maxima[1],
                    dsi_expected_contrast=float(
                        dsis.isel(neuron=index).sel(intensity=expected_intensity)
                    ),
                    preferred_degrees=direction,
                    expected_degrees=expected_direction,
                    circular_error_degrees=abs((direction - expected_direction + 180) % 360 - 180),
                    direction_metric_parity_error_degrees=error,
                )
            )
    result = dict(
        status="completed",
        termination_reason="saved response analysis completed",
        source_commit=config["source_commit"],
        flash_index=fri,
        flash_parity_max_error=parity,
        moving_edge=rows,
        cardinal_pilot_only=len(np.unique(edge.angle)) == 4,
        physiological_reproduction_verified=False,
        AC_integration_verified=False,
        goal_achieved=False,
        actuator_messages_sent=0,
        optimizer_updates=0,
    )
    write_json(run / "result.json", result)
    import matplotlib.pyplot as plt

    speeds = sorted({row['speed'] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for axis, (key, title, cmap, low, high) in zip(axes, [
        ('edge_contrast_index', 'Edge contrast: ON positive / OFF negative', 'coolwarm', -1, 1),
        ('circular_error_degrees', 'Preferred direction error (degrees)', 'magma', 0, 180),
        ('dsi_expected_contrast', 'DSI at expected contrast', 'viridis', 0, 1),
    ], strict=True):
        matrix = np.array([[next(row[key] for row in rows
                                if row['cell_type'] == name and row['speed'] == speed)
                            for speed in speeds] for name in types])
        artist = axis.imshow(matrix, vmin=low, vmax=high, cmap=cmap, aspect='auto')
        axis.set_xticks(range(len(speeds)), [f'{speed:g}' for speed in speeds])
        axis.set_yticks(range(len(types)), types)
        axis.set_xlabel('Edge speed (columns/s)')
        axis.set_title(title, fontsize=10)
        figure.colorbar(artist, ax=axis, shrink=.8)
    figure.suptitle('Published model flow/0000/000: component responses, no vehicle integration')
    figure.savefig(run/'response-summary.png', dpi=160)
    plt.close(figure)
    lines = [
        "# flyvis 부품 반응 분석",
        "",
        '![부품 반응 요약](response-summary.png)',
        '',
        "저자 공개 모델 flow/0000/000의 저장 반응을 저자 분석 함수로 평가했다.",
        f"{len(np.unique(edge.angle))}방향·{len(np.unique(edge.speed))}속도 시험이다. "
        "전체 생리 재현·차량 통합·주행 성공을 뜻하지 않는다.",
        "",
        "| 속도(열/s) | 세포형 | 깜박임 지수(전시야) | 이동 경계 대비 지수 | "
        "기대 부호 일치 | 방향 오차(도) | DSI |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        name = row["cell_type"]
        lines.append(
            f"| {row['speed']:g} | {name} | {fri['-1'][name]:.4f} | "
            f"{row['edge_contrast_index']:.4f} | "
            f"{row['expected_contrast_sign_matches']} | {row['circular_error_degrees']:.2f} | "
            f"{row['dsi_expected_contrast']:.4f} |"
        )
    lines += [
        "",
        "극성 지수는 양수면 밝은 자극, 음수면 어두운 자극 선호다.",
        "이동 경계 대비 지수는 각 대비의 최대 반응으로 계산한 공학적 요약이며, "
        "저자의 FRI와 구별한다.",
        "DSI·선호 방향·반응 시간 마스크·깜박임 지수는 고정한 저자 함수를 직접 호출했다.",
        "깜박임은 독립 NumPy 계산과, 방향은 독립 복소 벡터 합 계산과도 대조했다.",
        "DSI는 저자 방식대로 두 대비 중 큰 방향별 반응합으로 정규화한다.",
        "방향 오차는 알려진 대비에서 계산했다. 단일 공개 모델의 반응이며 "
        "집단 전체나 생체 전체 검증은 아니다.",
        "기대 극성과 다른 결과도 보존하며 결과에 맞춰 모델이나 세포 이름을 바꾸지 않는다.",
        "",
        "출처: [저자 코드](https://github.com/TuragaLab/flyvis/tree/"
        + config["source_commit"]
        + "/flyvis/analysis), [논문](https://doi.org/10.1038/s41586-024-07939-3).",
        "",
    ]
    (run / "report.md").write_text("\n".join(lines), encoding="utf8")
    print(json.dumps(dict(status="completed", cells=rows, flash_parity_max_error=parity)))


if __name__ == "__main__":
    main()

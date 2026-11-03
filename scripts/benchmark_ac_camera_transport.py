"""Read-only finite lossless camera transport measurement, optionally paired same-shot RGB."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain_drive.ac_capture import WindowsACWindow
from flybrain_drive.ac_csp_vision import CSPVisionRecorder
from flybrain_drive.evidence import begin_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-format', choices=['png', 'bmp'], required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--verify-png-reference', action='store_true')
    parser.add_argument('--shared-transport', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 120:
        raise ValueError('Finite 1..120 group measurement required')
    config = json.loads(Path('configs/ac-csp-retina.json').read_text(encoding='utf8'))
    run = begin_run(args.output, dict(arguments=vars(args), scope=__doc__, seed=862953))
    result = dict(status='failed', ui_actions_sent=0, actuator_messages_sent=0,
                  goal_achieved=False)
    vision = None
    try:
        windows = WindowsACWindow().discover()
        if len(windows) != 1:
            raise ValueError('Exactly one reviewed engine required')
        vision = CSPVisionRecorder(windows[0].hwnd, config, max_frames=args.frames,
            image_format=args.image_format, verify_png_reference=args.verify_png_reference,
            shared_transport=args.shared_transport, period=.04 if args.shared_transport else .03)
        deadline = time.monotonic()+args.frames*.03+8
        while vision.thread.is_alive():
            if time.monotonic() > deadline:
                raise TimeoutError('Same camera request measurement deadline exceeded')
            time.sleep(.01)
        if vision.error:
            raise RuntimeError(str(vision.error))
        frames = [json.loads((vision.directory/f'frame-{i:04d}.json').read_text(encoding='utf8'))
                  for i in range(vision.count)]
        save = ([v['save_finished_clock']-v['render_call_finished_clock']
                 for f in frames for v in f['views']] if not args.shared_transport else None)
        render = [v['render_call_finished_clock']-v['render_started_clock']
                  for f in frames for v in f['views']]
        result.update(status='completed', frames=vision.count,
            paired_rgb_equal_images=vision.count*4 if args.verify_png_reference else None,
            save_seconds_percentiles=np.percentile(save, [50, 95, 100]).tolist()
                if save is not None else None,
            shared_transport=args.shared_transport,
            render_seconds_percentiles=np.percentile(render, [50, 95, 100]).tolist())
    except Exception as error:
        result.update(error=type(error).__name__, reason=str(error))
    finally:
        if vision:
            result['vision'] = vision.close()
            result['archive'] = vision.archive_to(run/'vision', bundle=True)
        write_json(run/'result.json', result)
    if result['status'] != 'completed':
        raise RuntimeError(result['reason'])


if __name__ == '__main__':
    main()

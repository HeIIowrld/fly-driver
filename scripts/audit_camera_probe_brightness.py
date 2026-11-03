"""Measure brightness and pixel changes in a retained four-view camera probe; not motion validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.probe)
    source = json.loads((root/'result.json').read_text(encoding='utf8'))
    archive = {x['file']:x['sha256'] for x in json.loads((root/'archive.json').read_text(encoding='utf8'))}
    if source['status'] != 'completed' or not source['all_hashes_verified']:
        raise ValueError('Completed hash-verified original camera probe required')
    by_view = {name:[] for name in ('front','left','right','rear')}
    for frame in source['frames']:
        for view in frame['views']:
            name=view['view_id']; path=root/'camera'/view['file']; payload=path.read_bytes()
            digest=hashlib.sha256(payload).hexdigest()
            if name not in by_view or digest != view['sha256'] or digest != archive['camera/'+view['file']]:
                raise ValueError('Original image identity mismatch')
            with Image.open(path) as image:
                pixels=np.asarray(image.convert('RGB'),dtype=np.float32)/255.
            by_view[name].append(pixels)
    out={}
    for name,frames in by_view.items():
        if len(frames) != len(source['frames']) or len(frames)<2:
            raise ValueError('All four original views required in every group')
        means=[float(x.mean()) for x in frames]
        differences=[float(np.abs(y-x).mean()) for x,y in zip(frames,frames[1:])]
        out[name]={'mean_rgb_fraction':means,'adjacent_mean_absolute_rgb_fraction':differences}
    path=Path(a.output);path.mkdir(parents=True,exist_ok=False)
    result={'source_result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
            'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'groups':len(source['frames']),'views':out,
            'physical_camera_exposure_verified':False,'independent_stationarity_verified':False,
            'motion_or_physiology_verified':False,'training_allowed':False}
    (path/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

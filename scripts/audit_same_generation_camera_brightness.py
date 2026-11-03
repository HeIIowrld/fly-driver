"""Compare archived same-generation first camera frames; no simulator control."""
import argparse,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--generation',required=True);a=p.parse_args()
 base=Path(a.base);entries=[]
 for archive in sorted(base.glob('*/vision/camera.zip')):
  root=archive.parent.parent
  try:
   result=json.loads((root/'result.json').read_text(encoding='utf8'));bundle=json.loads((archive.parent/'bundle.json').read_text(encoding='utf8'))
   with zipfile.ZipFile(archive) as z:
    meta=json.loads(z.read('frame-0000.json'))
    if meta.get('generation')!=a.generation:continue
    if sha256(archive)!=bundle['sha256']:raise ValueError('archive hash mismatch: '+str(archive))
    views={}
    for view in ('front','left','right','rear'):
     rgb=np.asarray(Image.open(io.BytesIO(z.read('0000-'+view+'.png'))).convert('RGB'))
     if rgb.shape!=(180,320,3):raise ValueError('unexpected RGB shape')
     views[view]={'mean':float(rgb.mean()),'nonzero_channels':int(np.count_nonzero(rgb))}
    entries.append(dict(run=root.name,source_result_sha256=sha256(root/'result.json'),archive_sha256=bundle['sha256'],script_clock=meta['script_clock'],views=views,training_admissible=False))
  except (FileNotFoundError,KeyError,zipfile.BadZipFile):continue
 entries.sort(key=lambda x:x['script_clock'])
 if len(entries)<2:raise ValueError('Need at least two same-generation archives')
 run=begin_run(a.output,dict(program_sha256=sha256(__file__),generation=a.generation));out=dict(status='completed_archived_brightness_comparison',records=entries,scope='First camera group per available archive only; camera motion, weather and exposure may differ. No causal or physical-time inference.',vehicle_control_sent=False,training_admissible=False)
 write_json(run/'result.json',out);print(json.dumps({'status':out['status'],'archives':len(entries),'first_clock':entries[0]['script_clock'],'last_clock':entries[-1]['script_clock']}))
if __name__=='__main__':main()

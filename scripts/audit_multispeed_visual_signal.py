"""Compare bend-vs-straight signals at sensory and flight-muscle stages."""
import json
from pathlib import Path
import numpy as np
from flybrain_drive.graph import sha256
from flybrain_drive.evidence import begin_run,write_json
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');pilot=B/'20261101-multispeed-symmetry-pilot-1';pr=json.loads((pilot/'result.json').read_text(encoding='utf8'));pf=pilot/'multispeed-pilot.npz';train=B/'20261101-frozen-symmetry-matched-speed-1';tr=json.loads((train/'result.json').read_text(encoding='utf8'));tf=train/'matched-speed-probe.npz'
if pr['status']!='completed_synthetic_multispeed_symmetry_pilot' or tr['status']!='completed_synthetic_frozen_symmetry_matched_speed' or sha256(pf)!=pr['output_sha256'] or sha256(tf)!=tr['output_sha256']:raise ValueError('verified stages required')
run=begin_run(B/'20261101-multispeed-visual-signal-audit-2',dict(program_sha256=sha256(__file__),pilot_sha256=sha256(pf),train_sha256=sha256(tf),optimizer_updates=0))
out=dict(status='failed',actor_qualified=False,vehicle_control_sent=False)
def rms(x):return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
try:
 rows=[]
 with np.load(pf,allow_pickle=False) as ep,np.load(tf,allow_pickle=False) as tp:
  for split,seed in [('fit',881403),('evaluation',881511)]:
   for speed in (1.5,3.,4.5):
    folder=(f'20261101-multispeed-eval-{seed}-{speed:g}' if split=='evaluation' else ('20261101-synthetic-road-yaw-881403' if speed==3. else f'20261101-synthetic-matched-speed-{speed:g}'));root=B/folder;rr=json.loads((root/'result.json').read_text(encoding='utf8'));cfg=json.loads((root/'manifest.json').read_text(encoding='utf8'))['config'];f=root/'responses.npz'
    if rr['status']!='completed_causal_synthetic_road_with_yaw' or rr['responses_sha256']!=sha256(f) or cfg['texture_seed']!=seed or (speed!=3. or split=='evaluation') and cfg['speed_m_s']!=speed:raise ValueError('video lineage mismatch')
    with np.load(f,allow_pickle=False) as z: retina=z['retina'][:,0,-10:].astype('float64');motion=z['motion'][:,0,-10:].astype('float64')
    muscle=(ep[f'{speed:g}_evaluation_muscles'] if split=='evaluation' else tp[f'{speed:g}_muscles']).astype('float64').reshape(3,10,6)
    detail=[]
    for idx in (1,2):
     dr=retina[idx]-retina[0];dm=motion[idx]-motion[0];dmu=muscle[idx]-muscle[0]
     detail.append(dict(bend=-1 if idx==1 else 1,retina_rms=rms(dr),retina_rms_per_view=[rms(dr[:,v]) for v in range(4)],motion_rms=rms(dm),t4_rms=rms(dm[:,:,0:4]),t5_rms=rms(dm[:,:,4:8]),muscle_rms=rms(dmu),muscle_mean_abs_per_output=np.mean(np.abs(dmu),axis=0).tolist()))
    rows.append(dict(split=split,seed=seed,speed_m_s=speed,response_sha256=sha256(f),bends=detail))
 out.update(status='completed_multispeed_visual_signal_audit',conditions=rows,interpretation='RMS of paired same-time zero-yaw synthetic bend-minus-straight samples; different units per stage, temporally correlated, not a transfer efficiency or physiological validation')
except Exception as e:out.update(error=type(e).__name__,reason=str(e))
finally:write_json(run/'result.json',out)
print(json.dumps(out,allow_nan=False))
if out['status']!='completed_multispeed_visual_signal_audit':raise SystemExit(1)

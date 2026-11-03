"""Apply synthetic-only linear diagnostic to fit-forbidden historical scenes."""
import json
from pathlib import Path
import numpy as np,torch
from flybrain_drive.bilateral_retinal_contrast_boundary import BilateralRetinalContrastBoundary
from flybrain_drive.episode_storage import load_array,resolve_payload_path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
B=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
root=B/'20261029-t1434-body21-flight-cache';manifest=root/'dataset.json';m=read(manifest);dataset=read(root/'result.json');diag=B/'20261101-bilateral-linear-capacity-1';result=read(diag/'result.json');f=diag/'diagnostic-linear-fit.npz'
if sha256(manifest)!=dataset['dataset_manifest_sha256'] or sha256(f)!=result['coefficients_sha256'] or result['status']!='completed_synthetic_bilateral_linear_capacity_diagnostic':raise ValueError('Verified sources required')
with np.load(f,allow_pickle=False) as z:coef=z['coefficients'].copy()
model=BilateralRetinalContrastBoundary(42,normalization_floor=.02).eval();rows=[]
for name in ('episode-03.npz','episode-04.npz','episode-06.npz','episode-07.npz'):
 e=next(x for x in m['episodes'] if x['file']==name);source=resolve_payload_path(root/name);record=read(Path(e['original']['recording'])/'result.json')
 if sha256(source)!=e['sha256'] or record.get('training_admissible') is not False:raise ValueError('Fit-forbidden original source required')
 with load_array(root/name,allow_pickle=False) as z:scene=z['scene_features'].copy()
 with torch.no_grad():features=model.features(torch.tensor(scene,dtype=torch.float32),torch.ones((len(scene),4),dtype=torch.bool)).numpy().astype(np.float64)
 pred=features@coef;rows.append(dict(episode=name,frames=len(scene),source_sha256=sha256(source),prediction_rms=float(np.sqrt(np.mean(pred**2))),abs_p95=float(np.quantile(abs(pred),.95)),over_half_fraction=float(np.mean(abs(pred)>.5))))
run=begin_run(B/'20261101-bilateral-linear-capacity-real-range-1',dict(program_sha256=sha256(__file__),diagnostic_sha256=sha256(f),dataset_sha256=sha256(manifest)));out=dict(status='completed_read_only_diagnostic_real_range',episodes=rows,actor_qualified=False,vehicle_control_sent=False,fit_on_real=False,all_views_assumed_valid=True,interpretation='Unbounded linear diagnostic predictions, not actual bounded actor output; no model update or proof of driving')
write_json(run/'result.json',out);print(json.dumps(out))

"""Summarize existing preassigned moving candidates without promoting or relabeling them."""
import argparse,json
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.training_source_guard import require_training_source

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
 base=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs');rows=[]
 for role in ('train','development'):
  prefix=f'20261031-pace-{role}-native-copy-candidate-00';root=base/prefix
  paths={'candidate':root/'result.json','recording':root/'recording/result.json','raw':base/(prefix+'-raw-audit')/'result.json','authority':base/(prefix+'-authority-audit')/'result.json','native':base/(prefix+'-native-audit')/'result.json'}
  d={k:read(v) for k,v in paths.items()};candidate,recording,raw,auth,native=[d[k] for k in ('candidate','recording','raw','authority','native')]
  if candidate['role']!=role or candidate['recording_result_sha256']!=sha256(paths['recording']) or auth['source_result_sha256']!=sha256(paths['recording']) or candidate['decisions']!=raw['decisions'] or candidate['decisions']!=auth['observations']:raise ValueError('Source role, count or original hash mismatch')
  if candidate['fit_allowed'] is not False or recording['training_admissible'] is not False or raw['continuous_40ms_training_admissible'] is not False:raise ValueError('Do not infer historical admission')
  reasons=[]
  if candidate['fit_allowed'] is not True:reasons.append('preassigned candidate denies fitting')
  if recording.get('training_admissible') is not True:reasons.append('recording does not admit training')
  if raw.get('continuous_40ms_training_admissible') is not True:reasons.append('continuous camera/body timing not admitted')
  if raw.get('expert_action_codec_verified') is not True:reasons.append('expert control codec not verified by raw audit')
  if auth.get('native_camera_source_time_synchronized') is not True or native.get('physical_sample_or_exposure_verified') is not True:reasons.append('physical camera/body sample synchronization unverified')
  if recording.get('expert_labels_available') is not True:reasons.append('expert labels unavailable or unverified')
  try:require_training_source(recording=recording,raw_audit=raw,label_audit={'training_admissible':False})
  except ValueError as e:guard_result='rejected: '+str(e)
  else:raise ValueError('Unqualified source must not be admitted')
  rows.append(dict(role=role,decisions=candidate['decisions'],source_paths={k:str(v) for k,v in paths.items()},source_sha256={k:sha256(v) for k,v in paths.items()},same_copy_native_ai_true=auth['native_ai_true'],physical_synchronization_verified=False,fit_allowed=False,blockers=reasons,guard=guard_result))
 run=begin_run(args.output,dict(program_sha256=sha256(__file__),guard_sha256=sha256('src/flybrain_drive/training_source_guard.py'),scope='existing preassigned source audit only',optimizer_updates=0))
 result=dict(status='completed_qualification_review',sources=rows,fit_permission_granted=False,optimizer_updates=0,vehicle_control_sent=False,interpretation='Existing historical candidates remain non-fit; no new data collection or proof of physical time')
 write_json(run/'result.json',result);print(json.dumps({'status':result['status'],'sources':[{'role':x['role'],'decisions':x['decisions'],'blockers':x['blockers']} for x in rows]},ensure_ascii=False))
if __name__=='__main__':main()

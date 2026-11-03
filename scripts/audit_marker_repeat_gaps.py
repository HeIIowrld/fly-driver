"""Summarize sampled packet ID gaps and nonmatching cached acceleration; diagnostic only."""
import base64,hashlib,json
from pathlib import Path
from flybrain_drive.ac_teacher_physics import decode_teacher_physics
OUT=Path('D:/CodexResearchRuns/flybrain-marker-repeat-gap-audit-20261102-01')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out={}
 for suffix in ('01','02'):
  root=Path(f'D:/CodexResearchRuns/flybrain-marker-motion-20261102-{suffix}');audit=Path(f'D:/CodexResearchRuns/flybrain-cached-native-value-match-20261102-{suffix}');result=json.loads((root/'result.json').read_text());a=json.loads((audit/'result.json').read_text());frames=json.loads((root/'diagnostic-frames.json').read_text());records=json.loads((root/'native-copy-windows.json').read_text())
  if sha(root/'result.json')!=a['source_result_sha256'] or sha(root/'native-copy-windows.json')!=result['native_copy_sha256'] or sha(root/'diagnostic-frames.json')!=result['metadata_sha256']:raise ValueError('source mismatch')
  ids=sorted({x['authority']['packet_id'] for x in records});gaps=[{'after':a,'before':b,'missing_id_count':b-a-1} for a,b in zip(ids,ids[1:]) if b-a>1];bad=[x for x in a['rows'] if not x['matching_all_three_fields']];wanted={p for row in bad for p in row['packet_ids']};native={}
  for rec in records:
   auth=rec['authority'];raw=base64.b64decode(auth['prefix_base64'],validate=True)
   if hashlib.sha256(raw).hexdigest()!=auth['prefix_sha256']:raise ValueError('native hash mismatch')
   if auth['packet_id'] in wanted:
    s=decode_teacher_physics(raw,rec['copy']['second_copy_end']);native[s.physics.packet_id]=list(s.physics.fields['accG'])
  cases=[]
  for frame in sorted({x['frame'] for x in bad}):
   record=frames[frame];path=Path(record['source_file'])
   if sha(path)!=record['sha256']:raise ValueError('metadata hash mismatch')
   view=next(x for x in json.loads(path.read_text())['views'] if x['view_id']=='front');e=view['render_marker_diagnostic'];m=next(x for x in bad if x['frame']==frame and x['view']=='front');pid=m['packet_ids'][0]
   cases.append({'frame':frame,'packet_id':pid,'speed_kmh':e['speed_kmh'],'csp_acceleration_g':e['acceleration_g'],'native_acceleration_g':native[pid]})
  out[suffix]={'unique_sampled_packet_ids':len(ids),'gaps':gaps,'nonmatching_views':len(bad),'nonmatching_distinct_frames':len(cases),'cases':cases,'source_sha256':sha(root/'result.json'),'value_match_sha256':sha(audit/'result.json')}
 report={'status':'completed_packet_gap_and_acceleration_diagnostic','runs':out,'program_sha256':sha(Path(__file__)),'training_admissible':False,'physical_exposure_sync_verified':False,'interpretation':'A gap means an ID not observed in sampled copies, not proof of engine packet loss. Numeric correspondence and speed/gyro uniqueness are sample-bound, not proof of common physical tick or exposure.'}
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

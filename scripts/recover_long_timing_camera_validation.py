"""Separate certificate for a false old1200-count validator rejection.

Never edit raw failed results. This only permits a subsequent independent full
lap/pixel/body audit, and does not itself admit training or sensor timestamps.
"""
import argparse,hashlib,json
from pathlib import Path
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.ac_csp_vision import validate_source_timing_ledger
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

ERROR='Source timing proof unavailable: Source timing original boundary clocks/capture count differ'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recording',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.recording);parent=read(root/'result.json');cr=read(root/'camera/result.json');tr=read(root/'teacher/result.json')
 cm=read(root/'camera/manifest.json')['config']['arguments'];rm=read(root/'manifest.json')['config'];owner=read(root/'camera/ownership-plan.json')
 plan=Path('runs/teacher-synchronized-20261102/camera-alignment-qualification-plan-live-mmf.json')
 run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),seed=780212,validator_sha256=sha256('src/flybrain_drive/ac_csp_vision.py')))
 result=dict(status='failed',training_admissible=False,student_optimizer_updates=0,actuator_messages_sent=0,ui_actions_sent=0)
 try:
  if (parent['status']!='failed_camera_demonstration' or cr['status']!='failed' or cr.get('reason')!=ERROR
      or cr['camera_close']['error']!=ERROR or cr.get('closing_error') or cr.get('error')
      or tr['status']!='curriculum_completed' or parent['teacher_exitcode']!=0 or parent['camera_exitcode']!=0
      or cr['native_body_close']['status']!='completed' or cr['native_body_close'].get('forced_termination')
      or cm['frames']!=1600 or cm['period']!=.09 or cm['source_body_recording'] is not True
      or cm['source_timing_diagnostic'] is not True or rm['alignment_plan_sha256']!=sha256(plan)
      or cm['expected_lua_sha256']!=read(plan)['source_candidate_sha256']):raise ValueError('Only original explicitly pinned long-recording count false rejection is recoverable')
  with CameraArchive(root/'camera/vision') as archive:
   raw=archive.read('source-timing-v1.json');ledger=json.loads(raw);consumer=archive.json('consumer-result.json')
   source=archive.read('deployed-camera.lua');frames=cr['frames'];terminal=cr['camera_close']['producer_terminal']
   if (not 1200<frames<=1600 or consumer['error'] is not None or consumer['frames']!=frames
       or consumer['producer_terminal']!=terminal or terminal['frames']!=frames or terminal['status'] not in ('stopped','completed')
       or hashlib.sha256(source).hexdigest()!=cm['expected_lua_sha256']
       or not 0<=ledger['capture_closed_clock']-ledger['started_clock']<=cm['frames']*cm['period']+5):raise ValueError('Original capture/consumer/source or declared finite duration differs')
   try:validate_source_timing_ledger(ledger,owner['request_id'],owner['generation'],expected_capture_count=frames)
   except ValueError as error:
    if str(error)!='Source timing original boundary clocks/capture count differ':raise
   else:raise ValueError('Original1200 count failure was not reproduced')
   proof=validate_source_timing_ledger(ledger,owner['request_id'],owner['generation'],expected_capture_count=frames,maximum_capture_count=1600)
   if proof['status']!='complete':raise ValueError('Incomplete or overflowed original timing ledger')
  result.update(status='recovered_original_camera_timing_validation',eligible_for_independent_audit=True,
   source_result_sha256=sha256(root/'result.json'),camera_result_sha256=sha256(root/'camera/result.json'),teacher_result_sha256=sha256(root/'teacher/result.json'),
   original_camera_bundle_sha256=cr['camera_archive']['bundle_sha256'],source_timing_sha256=hashlib.sha256(raw).hexdigest(),
   source_plan_sha256=sha256(plan),source_candidate_sha256=cm['expected_lua_sha256'],original_frames=frames,
   source_timing=proof,original_source_flags_unchanged=True,sensor_age_bounds_unchanged=True,
   interpretation='Only the post-capture validator count cap was stale relative to the already declared1600/90ms150s profile. Full independent lap/body/pixel qualification still required.')
 except Exception as error:result.update(error=type(error).__name__,reason=str(error))
 write_json(run/'result.json',result);print(json.dumps(result))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()

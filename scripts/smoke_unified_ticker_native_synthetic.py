"""Read-only native receipt and synthetic-scene ticker smoke test; never drives."""
import argparse
import base64
import json
import time
from pathlib import Path
import torch
from flybrain_drive.ac_image_buffer import ExistingNamedMapping
from flybrain_drive.ac_native_physics import PHYSICS_PREFIX, decode_static_identity_prefix
from flybrain_drive.ac_teacher_status import AI_STATUS_PREFIX
from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256
from flybrain_drive.temporal_body_context import HostFreshnessLimits, per_view_temporal_body_context
from flybrain_drive.unified_host_guard import UnifiedHostGuard
from flybrain_drive.unified_live_ticker import UnifiedLiveBodyTicker
from flybrain_drive.unified_multirate_shadow import UnifiedMultirateShadow
from flybrain_drive.virtual_navigation_body import NavigationBodyEncoder, decode_navigation_body_prefix
from scripts.record_ac_airflow_timed_diagnostic import NativeCopySampler
from test_unified_flight_sensorimotor import make_model


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 run=begin_run(a.output,dict(script_sha256=sha256(__file__),ticker_sha256=sha256('src/flybrain_drive/unified_live_ticker.py'),synthetic_scene=True,synthetic_graph=True,actor=False,seed=863009))
 result={'status':'failed','vehicle_control_sent':False,'camera_request_sent':False,'training_admissible':False,'synthetic_scene_and_clock_bounds':True,'physical_exposure_sync_verified':False}
 sampler=ticker=None
 try:
  torch.set_num_threads(1);torch.manual_seed(863009)
  with ExistingNamedMapping('Local\\acpmf_static',200) as m: identity=decode_static_identity_prefix(m.read())
  model=make_model().eval().requires_grad_(False);model.freshness_limits=HostFreshnessLimits(.05,.15)
  shadow=UnifiedMultirateShadow(UnifiedHostGuard(model),goals=torch.tensor([[0.,1.]]))
  sampler=NativeCopySampler(lambda:ExistingNamedMapping('Local\\acpmf_physics',AI_STATUS_PREFIX.size));sampler.start()
  ticker=UnifiedLiveBodyTicker(shadow,native_rows=sampler.rows,native_identity=identity,episode_id='synthetic-scene-native-smoke');ticker.start()
  encoder=NavigationBodyEncoder('configs/ac-virtual-navigation-body-v2.json',native_identity=identity)
  deadline=time.perf_counter()+.5;next_frame=0.;frames=0
  while time.perf_counter()<deadline:
   if sampler.error or ticker.error:raise RuntimeError('sampler/ticker: '+str((sampler.error,ticker.error)))
   now=time.perf_counter()
   if now>=next_frame:
    row=sampler.rows[-1];raw=base64.b64decode(row['authority']['prefix_base64'],validate=True)
    sample=decode_navigation_body_prefix(raw[:PHYSICS_PREFIX.size],row['copy']['second_copy_end'])
    body=encoder.encode(sample,episode_id='synthetic-scene-native-smoke',decision_monotonic_seconds=now)
    # These render bounds are expressly synthetic and cannot certify a camera or sensor.
    frame=per_view_temporal_body_context((None,)*4,body,render_start_lowers=(now-.02,)*4,camera_receipt_time=now-.01,decision_time=now)
    ticker.publish(torch.zeros((1,576)),frame,frames);frames+=1;next_frame=now+.04
   time.sleep(.002)
  result.update(status='synthetic_smoke_completed',scenes=frames)
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:
  if ticker is not None:
   try:ticker.stop()
   except Exception as e:result.update(status='failed',ticker_stop_error=str(e))
   write_json(run/'ticker-events.json',ticker.events);result.update(ticker_error=ticker.error,ticker_events_sha256=sha256(run/'ticker-events.json'),shadow_outputs=sum(x['status']=='shadow_output' for x in ticker.events))
   if ticker.error:result['status']='failed'
  if sampler is not None:
   try:sampler.stop();write_json(run/'native-copy.json',sampler.rows);result.update(native_error=sampler.error,native_sha256=sha256(run/'native-copy.json'))
   except Exception as e:result.update(status='failed',sampler_stop_error=str(e))
  write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()

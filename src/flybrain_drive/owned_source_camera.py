"""Identity and cancellation for a separate camera recorder; no actuator API."""
import json
import os
import re
import time
from pathlib import Path

from .graph import sha256
from .ac_published_state import read_published_json


class OwnedSourceCamera:
    def __init__(self, plan_path, *, generation, window_handle, installed_source):
        self.plan_path=Path(plan_path).resolve()
        self.plan=json.loads(self.plan_path.read_text(encoding='utf8'))
        row=self.plan
        root=(Path(os.environ['LOCALAPPDATA'])/'flybrain/camera-coordination').resolve()
        if (root not in self.plan_path.parent.parents or self.plan_path.name!='ready.json'
                or row.get('schema')!='flybrain.owned.source_body_camera.v1'
                or row.get('generation')!=generation or row.get('window_handle')!=window_handle
                or not re.fullmatch(r'mmf-stream-[0-9a-f]{32}',row.get('request_id',''))
                or sha256(installed_source)!=row.get('source_sha256')):
            raise ValueError('Owned source-camera identity/source differs')
        for key,name in [('trigger_file','trigger'),('stop_file','stop'),('health_file','health.json')]:
            if Path(row[key]).resolve()!=self.plan_path.parent/name:
                raise ValueError('Camera coordination path escaped the owned directory')
        if Path(row['trigger_file']).exists() or Path(row['stop_file']).exists():
            raise ValueError('Fresh untriggered camera ownership required')
        self.triggered=False
        self.check(False,None)

    def check(self, active, request_id):
        path=Path(self.plan['health_file'])
        row,modified=read_published_json(path)
        self.validate_health(row,modified,active,request_id)

    def validate_health(self,row,modified,active,request_id):
        published=row.get('observed_host_time')
        age=time.perf_counter()-published if type(published) in (float,int) else float('inf')
        epsilon=2*time.get_clock_info('perf_counter').resolution
        if (not -epsilon <= age <= 2
                or row.get('request_id')!=self.plan['request_id'] or row.get('error')
                or row.get('state') not in ('prepared','capturing','completed')
                or active and request_id!=self.plan['request_id']):
            raise RuntimeError('Owned camera freshness/identity/health failed: '+json.dumps(dict(
                qpc_publication_age_seconds=age,filesystem_wall_age_seconds=time.time()-modified,
                camera_active=active,camera_request_id=request_id,expected_request_id=self.plan['request_id'],
                health=row),allow_nan=True))

    def trigger_near_initial_finish(self, progress, crossings):
        if not self.triggered and (progress>=.97 or crossings>=1):
            Path(self.plan['trigger_file']).touch(exist_ok=False)
            self.triggered=True

    def start_paused_warmup(self):
        if self.triggered:
            raise RuntimeError('Camera start may only be requested once')
        Path(self.plan['trigger_file']).touch(exist_ok=False)
        self.triggered=True
        deadline=time.perf_counter()+8
        while True:
            row,_=read_published_json(self.plan['health_file'])
            self.check(False,None)
            if row.get('frames',0)>=3 and row.get('state')=='capturing':
                return
            if time.perf_counter()>deadline:
                raise TimeoutError('Prepared paused camera warmup not verified')
            time.sleep(.01)

    def request_stop(self):
        Path(self.plan['stop_file']).touch(exist_ok=True)

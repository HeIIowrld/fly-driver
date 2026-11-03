"""Build candidate allowing only explicitly owned paused MMF resume."""
from pathlib import Path
from scripts.build_camera_release_witness_candidate import build as build_v3
DEST=Path('integrations/ac_csp/candidates/FlyBrainCamera-owned-mmf-resume-v4.lua')
OLD='''  local ownsCamera = isMotion and active and request.camera_request_id == active.request.id
    and active.deadline - now >= motionSeconds + ((isSpeed60 or isPace) and 9 or 5)
    and now - (active.last_capture_clock or -1e9) <= .15
'''
NEW=OLD+'''  -- Narrow lifecycle handoff: a previously started, owned MMF stream with
  -- actual fresh frames, adequate remaining budget and no stop request.
  local ownsPausedMMFResume = request.command == 'resume' and request.owned_mmf_resume == true
    and before.paused and not before.in_menu and active and active.request
    and request.camera_request_id == active.request.id
    and active.request.mmf_metadata == true and active.request.shared_transport == true
    and active.request.host_persistence_protocol == 'immutable_ram_v3'
    and active.mmfAck ~= nil and active.mmfFrame ~= nil and active.frames >= 1
    and active.request.frames - active.frames >= 2 and active.deadline - now >= 2
    and now - (active.last_capture_clock or -1e9) <= .2
    and not io.fileExists(active.directory .. '/stop')
'''
def build():
 s=build_v3().decode('utf8')
 for old,new in [(OLD,NEW),('and not before.is_online and (not before.camera_active or ownsCamera) and not before.error','and not before.is_online and (not before.camera_active or ownsCamera or ownsPausedMMFResume) and not before.error'),('and (request.camera_request_id == nil or ownsCamera)','and (request.camera_request_id == nil or ownsCamera or ownsPausedMMFResume)\n    and (request.owned_mmf_resume == nil or ownsPausedMMFResume)'),('capabilities = {disable_ideal_line = ac.trySimKeyPressCommand ~= nil,','capabilities = {owned_mmf_resume = true, disable_ideal_line = ac.trySimKeyPressCommand ~= nil,')]:
  if s.count(old)!=1:raise ValueError('Exact v3 anchor required')
  s=s.replace(old,new)
 return s.encode('utf8')
if __name__=='__main__':DEST.write_bytes(build())

"""Build non-deployed v3: generation-specific release witness, no admission."""
from pathlib import Path
from scripts.build_camera_orphan_observation_candidate import build as build_v2
DEST=Path('integrations/ac_csp/candidates/FlyBrainCamera-release-witness-v3.lua')
OLD='''            assert(not io.fileExists(observation), 'Prior orphan observation requires host review')
            write(observation, {schema='flybrain.csp.camera.orphan_observation.v1',
              status='prior_generation_unresolved', request_id=id,
              expected_generation=request.expected_generation, observer_generation=generation,
              original_terminal_present=false, stop_present=true, old_process_exit_verified=false,
              prior_camera_started='unknown', prior_frames='unknown', actor_admissible=false})
'''
NEW='''            if io.fileExists(observation) then
              local existing = JSON.parse(assert(io.load(observation), 'Cannot read prior orphan observation'))
              assert(type(existing) == 'table' and existing.schema == 'flybrain.csp.camera.orphan_observation.v1'
                and existing.status == 'prior_generation_unresolved' and existing.request_id == id
                and existing.expected_generation == request.expected_generation
                and existing.actor_admissible == false, 'Prior orphan observation differs; host review required')
            else
              write(observation, {schema='flybrain.csp.camera.orphan_observation.v1',
                status='prior_generation_unresolved', request_id=id,
                expected_generation=request.expected_generation, observer_generation=generation,
                original_terminal_present=false, stop_present=true, old_process_exit_verified=false,
                prior_camera_started='unknown', prior_frames='unknown', actor_admissible=false})
            end
'''
TAIL='''ac.onRelease(function()
  if brakeSettle then pcall(ac.tryToPause, true) end
  stopExpertProbe(os.preciseClock(), 'app release')
  closeActive()
  flushSourceTiming() -- best effort only; skip if release cleanup is still unobserved
  if heartbeatTransport then heartbeatTransport:close() end -- keep host lease/fence
end)
'''
NEWTAIL='''ac.onRelease(function()
  local hadActive = active ~= nil
  local activeID = active and active.request and active.request.id or nil
  if brakeSettle then pcall(ac.tryToPause, true) end
  stopExpertProbe(os.preciseClock(), 'app release')
  closeActive()
  flushSourceTiming() -- best effort only; skip if release cleanup is still unobserved
  if heartbeatTransport then heartbeatTransport:close() end -- keep host lease/fence
  -- Local callback evidence only: never invent a prior request terminal or claim
  -- old engine process exit. Failure remains observable as a missing witness.
  pcall(function()
    local token = generation:gsub('[^a-zA-Z0-9_-]', '_')
    local path = root .. '/release-' .. token .. '.json'
    if io.fileExists(path) then return end
    write(path, {schema='flybrain.csp.camera.release_observation.v1',
      generation=generation, active_before=hadActive, active_request_id=activeID,
      active_after=(active ~= nil), actor_admissible=false,
      prior_generation_resolved=false, engine_exit_verified=false})
  end)
end)
'''
def build():
 s=build_v2().decode('utf8')
 if s.count(OLD)!=1 or s.count(TAIL)!=1:raise ValueError('Exact v2 anchors required')
 return s.replace(OLD,NEW).replace(TAIL,NEWTAIL).encode('utf8')
if __name__=='__main__':DEST.write_bytes(build())

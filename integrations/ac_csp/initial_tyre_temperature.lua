-- Offline episode preparation only; CSP documents that BOTH setters invalidate lap.
-- No thermostat, camera, steering, pedals, AI, or lifecycle side effects here.
local M = {}
local appliedGeneration, appliedEpoch = nil, nil
local wakeGeneration, wakeEpoch = nil, nil
local function finite(x)
  return type(x) == 'number' and x == x and math.abs(x) ~= math.huge
end
local function callable(x) return type(x) == 'function' or type(x) == 'cdata' end
function M.available(api)
  return type(api) == 'table' and callable(api.setTyresBlankets)
    and callable(api.setTyresTemperature)
end
function M.lastEpoch() return appliedEpoch end
local function liveProof(request)
  local p = request.native_stationary_packets
  return request.native_stationary_verified == true and type(p) == 'table' and #p == 3
    and finite(p[1]) and finite(p[2]) and finite(p[3])
    and p[1] >= 0 and p[1] % 1 == 0 and p[2] % 1 == 0 and p[3] % 1 == 0
    and p[1] < p[2] and p[2] < p[3]
end
local function validateBase(request, state, now, api)
  if type(request) ~= 'table' or type(state) ~= 'table' then return false end
  return request.schema == 'flybrain.csp.session.request.v1'
    and type(request.id) == 'string' and #request.id == 32
    and request.id:match('^[a-f0-9]+$') ~= nil
    and type(state.generation) == 'string' and #state.generation > 0
    and finite(state.epoch) and state.epoch >= 0 and state.epoch % 1 == 0
    and request.generation == state.generation and request.epoch == state.epoch
    and not (appliedGeneration == state.generation and appliedEpoch == state.epoch)
    and finite(now) and finite(request.expires_clock)
    and request.expires_clock >= now and request.expires_clock <= now + 5
    and request.destination_flags == 15
    and request.remove_blankets == true and request.released_controller_verified == true
    and finite(request.temperature_c) and request.temperature_c >= 20 and request.temperature_c <= 120
    and request.expected_car_id == 'bmw_z4_gt3' and request.expected_track_id == 'monza'
    and state.car_id == request.expected_car_id and state.track_id == request.expected_track_id
    and state.in_menu == false and state.is_online == false
    and state.ai_controlled == false and state.camera_active == false
    and state.expert_probe_active == false and state.brake_settle_active == false
    and state.physics_api_allowed == true and not state.error and not state.pending_receipt_id
    and finite(state.speed_kmh) and math.abs(state.speed_kmh) < .1
    and finite(state.gas) and state.gas >= 0 and state.gas < .02
    and finite(state.brake) and state.brake > .98 and state.brake <= 1
    and request.camera_request_id == nil and request.owned_mmf_resume == nil
    and M.available(api)
end
function M.validate(request, state, now, api)
  return validateBase(request,state,now,api) and request.command == 'set-tyres-temperature-initial'
    and (request.phase == 'episode_initialization' and state.paused == true
      or request.phase == 'episode_initialization_live' and state.paused == false
        and state.live == true and liveProof(request))
end
function M.wakeAvailable(api) return M.available(api) and callable(api.awakeCar) end
function M.lastWakeEpoch() return wakeEpoch end
function M.validateWake(request,state,now,api)
  return validateBase(request,state,now,api) and request.command == 'wake-thermal-native-initial'
    and request.phase == 'episode_initialization_live' and state.paused == false and state.live == true
    and M.wakeAvailable(api) and not (wakeGeneration == state.generation and wakeEpoch == state.epoch)
end
function M.prepareNative(request,state,now,api)
  if not M.validateWake(request,state,now,api) then return {status='native_wake_guard_rejected',calls={}} end
  wakeGeneration,wakeEpoch=state.generation,state.epoch
  local ok,value=pcall(api.awakeCar,0)
  return {status=ok and value ~= false and 'native_wake_returned_readback_required' or 'native_wake_unknown',
    calls={{name='awakeCar',call_returned=ok,returned_false=ok and value == false or false,error=not ok and tostring(value) or nil}},
    current_lap_excluded=true,temperature_setter_called=false,native_readiness_verified=false,same_epoch_wake_retry_prohibited=true}
end
function M.apply(request, state, now, api)
  if not M.validate(request, state, now, api) then
    return {status='thermal_guard_rejected', calls={}, current_lap_excluded=true}
  end
  -- Consume BEFORE first setter. A partial/unknown outcome must never be retried
  -- in this session, even under a new request ID. Start a new explicit episode.
  appliedGeneration, appliedEpoch = state.generation, state.epoch
  local calls = {}
  local function call(name, ...)
    local ok, value = pcall(api[name], ...)
    calls[#calls+1] = {name=name, call_returned=ok,
      returned_false=ok and value == false or false, error=not ok and tostring(value) or nil}
    return ok and value ~= false
  end
  local blankets = call('setTyresBlankets', 0, nil, false)
  local temperature = false
  if blankets then temperature = call('setTyresTemperature', 0, nil, request.temperature_c, 15) end
  return {status=blankets and temperature and 'thermal_calls_returned_readback_required'
      or 'thermal_partial_or_unknown', calls=calls,
    target_temperature_c=request.temperature_c, destination_flags=15,
    current_lap_excluded=true, sdk_documents_lap_invalidation=true,
    actual_lap_validity_observed=false, temperature_applied_verified=false,
    native_readback_required=true, same_epoch_retry_prohibited=true}
end
return M

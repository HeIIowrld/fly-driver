-- Read-only diagnostic candidate. No actor output, files, controls, or module-load I/O.
-- ac.getWindVelocity() and car.velocity are candidate values pending independent proof.
local M = {}
local function finite(x)
  return type(x) == 'number' and x == x and x ~= math.huge and x ~= -math.huge
end
local function vec3(v)
  assert(v and finite(v.x) and finite(v.y) and finite(v.z), 'finite vec3 required')
  return {v.x, v.y, v.z}
end
local function vec2(v)
  assert(v and finite(v.x) and finite(v.y), 'finite vec2 required')
  return {v.x, v.y}
end

function M.capture(ac, preciseClock, generation)
  assert(type(generation) == 'string' and #generation > 0, 'source generation required')
  local started = preciseClock()
  local car, sim = ac.getCar(0), ac.getSim()
  assert(car and sim, 'car and sim required')
  local value = {schema='flybrain.csp.airflow.diagnostic.v1', generation=generation,
    wind_velocity_candidate_m_s=vec3(ac.getWindVelocity()),
    weather_wind_velocity_candidate_kmh=vec2(sim.windVelocityKmh),
    weather_wind_speed_candidate_kmh=sim.windSpeedKmh,
    vehicle_velocity_candidate_m_s=vec3(car.velocity),
    body_forward_candidate=vec3(car.look), body_left_candidate=vec3(car.side),
    body_up_candidate=vec3(car.up), car_physics_timestamp_ms=car.timestamp,
    sim_frame=tostring(sim.frame), started_source_clock=started,
    actor_admissible=false, source_verified=false}
  value.finished_source_clock = preciseClock()
  assert(finite(started) and finite(value.finished_source_clock)
    and value.finished_source_clock >= started and finite(value.car_physics_timestamp_ms)
    and finite(value.weather_wind_speed_candidate_kmh), 'finite ordered clocks required')
  return value
end

return M

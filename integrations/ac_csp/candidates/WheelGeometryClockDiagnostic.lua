-- Read-only geometry diagnostic: no controls, actor input or module-load I/O.
local M = {}
local function finite(x)
  return type(x) == 'number' and x == x and x ~= math.huge and x ~= -math.huge
end
local function vec(v)
  assert(v and finite(v.x) and finite(v.y) and finite(v.z), 'finite vec3 required')
  return {v.x, v.y, v.z}
end
function M.capture(ac, preciseClock, generation)
  assert(type(generation) == 'string' and #generation > 0, 'source generation required')
  local started = preciseClock()
  local car, sim = ac.getCar(0), ac.getSim()
  assert(car and sim and car.wheels, 'car wheels and sim required')
  local sim_before = sim.time
  local physics_before = car.timestamp
  local wheels = {}
  for i=0,3 do
    local w=car.wheels[i]
    assert(w and finite(w.load) and w.load >= 0, 'finite nonnegative wheel load required')
    wheels[i+1] = {native_index=i, position_world=vec(w.position),
      contact_world=vec(w.contactPoint), outside_world=vec(w.outside), load_candidate=w.load}
  end
  local physics_after = car.timestamp
  local sim_after = sim.time
  local finished = preciseClock()
  assert(finite(started) and finite(finished) and finished >= started
    and finite(physics_before) and finite(physics_after) and finite(sim_before)
    and finite(sim_after) and sim_after >= sim_before, 'finite ordered clocks required')
  return {schema='flybrain.csp.wheel-geometry.diagnostic.v1', generation=generation,
    car_position_world=vec(car.position), body_forward_candidate=vec(car.look),
    body_side_candidate=vec(car.side), body_up_candidate=vec(car.up), wheels=wheels,
    car_physics_timestamp_ms=physics_before, car_physics_timestamp_after_ms=physics_after,
    sim_time_before_ms=sim_before, sim_time_after_ms=sim_after, sim_frame=tostring(sim.frame),
    started_source_clock=started, finished_source_clock=finished,
    state_update_domain='csp_graphics_frame', sim_update_domain='csp_graphics_frame',
    native_tick_link_verified=false, actor_admissible=false, source_verified=false}
end
return M

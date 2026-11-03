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
  local wheels = {}
  for i=0,3 do
    local w=car.wheels[i]
    assert(w and finite(w.load) and w.load >= 0, 'finite nonnegative wheel load required')
    wheels[i+1] = {native_index=i, position_world=vec(w.position),
      contact_world=vec(w.contactPoint), outside_world=vec(w.outside), load_candidate=w.load}
  end
  local finished = preciseClock()
  assert(finite(started) and finite(finished) and finished >= started
    and finite(car.timestamp), 'finite ordered clocks required')
  return {schema='flybrain.csp.wheel-geometry.diagnostic.v1', generation=generation,
    car_position_world=vec(car.position), body_forward_candidate=vec(car.look),
    body_side_candidate=vec(car.side), body_up_candidate=vec(car.up), wheels=wheels,
    car_physics_timestamp_ms=car.timestamp, sim_frame=tostring(sim.frame),
    started_source_clock=started, finished_source_clock=finished,
    actor_admissible=false, source_verified=false}
end
return M

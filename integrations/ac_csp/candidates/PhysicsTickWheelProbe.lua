-- Research-only physics callback observer. No file/network I/O or control writes.
-- Relative sequence and accumulated dt are not native packet IDs or wall time.
local M = {}
local function finite(x)
  return type(x) == 'number' and x == x and x ~= math.huge and x ~= -math.huge
end
local function vec(v)
  assert(v and finite(v.x) and finite(v.y) and finite(v.z), 'finite physics vector required')
  return {v.x, v.y, v.z}
end
function M.new(capacity)
  assert(type(capacity) == 'number' and capacity == math.floor(capacity)
    and capacity >= 1 and capacity <= 2048, 'bounded capacity required')
  local state = {capacity=capacity, entries={}, count=0, sequence=0, elapsed=0, failed=false}
  function state:step(ac, dt)
    assert(not self.failed, 'explicit reset required after failure')
    local ok, err = pcall(function()
      assert(finite(dt) and dt > 0 and dt <= .1, 'bounded physics dt required')
      assert(self.count < self.capacity, 'capture capacity exhausted')
      local car=ac.accessCarPhysics()
      assert(car and car.wheels and finite(car.speedKmh) and car.speedKmh >= 0,
        'finite physics car required')
      local loads={}
      for i=0,3 do
        local wheel=car.wheels[i]
        assert(wheel and finite(wheel.load) and wheel.load >= 0, 'finite wheel load required')
        loads[i+1]=wheel.load
      end
      local gforces=vec(car.gForces)
      local angular=vec(car.localAngularVelocity)
      local seq=self.sequence+1
      local elapsed=self.elapsed+dt
      self.entries[self.count+1]={sequence=seq, dt=dt, elapsed=elapsed,
        speed_kmh=car.speedKmh, wheel_load_native_order=loads,
        g_forces_unregistered=gforces, angular_velocity_unregistered=angular,
        native_packet_id=nil, physical_time_verified=false, actor_admissible=false}
      self.sequence, self.elapsed, self.count=seq,elapsed,self.count+1
    end)
    if not ok then self.failed=true; error(err, 0) end
  end
  function state:reset()
    self.entries, self.count, self.sequence, self.elapsed, self.failed={},0,0,0,false
  end
  return state
end
return M

-- Read-only, offline diagnostic markers; track progress is never actor input.
local M={}
local function finite(x) return type(x)=='number' and x==x and x~=math.huge and x~=-math.huge end
function M.start(ac, clock, generation, markers)
  assert(type(generation)=='string' and #generation>0, 'generation required')
  assert(type(markers)=='table' and #markers>=1 and #markers<=4, 'one to four markers required')
  for i=1,#markers do
    assert(finite(markers[i]) and markers[i]>=0 and markers[i]<1, 'normalized marker required')
    for j=1,i-1 do assert(markers[i]~=markers[j], 'unique markers required') end
  end
  local s={generation=generation,events={},disposers={},closed=false}
  function s:stop()
    if self.closed then return end
    self.closed=true
    local failed=false
    for _,dispose in ipairs(self.disposers) do
      local ok=pcall(dispose)
      if not ok then failed=true end
    end
    self.disposers={}
    assert(not failed, 'marker unsubscribe failed')
  end
  local ok,err=pcall(function()
    for i=1,#markers do
      local index,progress=i,markers[i]
      local dispose=ac.onTrackPointCrossed(0,progress,function(carIndex,physicsMs)
        if s.closed or s.events[index] then return end
        local arrival=clock();local sim=ac.getSim()
        if carIndex~=0 or not finite(physicsMs) or not finite(arrival) or not sim
          or not finite(sim.time) then return end
        s.events[index]={marker_index=index,progress=progress,physics_event_ms=physicsMs,
          callback_sim_time_ms=sim.time,callback_source_clock=arrival,
          actor_admissible=false,physical_sync_verified=false}
      end)
      assert(type(dispose)=='function', 'disposable marker subscription required')
      s.disposers[#s.disposers+1]=dispose
    end
  end)
  if not ok then s:stop(); error(err,0) end
  function s:snapshot()
    local rows={}
    for i=1,#markers do if self.events[i] then rows[#rows+1]=self.events[i] end end
    return {schema='flybrain.csp.physics-marker.diagnostic.v1',generation=self.generation,
      events=rows,actor_admissible=false,physical_sync_verified=false}
  end
  return s
end
return M

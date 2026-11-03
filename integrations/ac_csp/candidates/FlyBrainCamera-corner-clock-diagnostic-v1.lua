local airflowDiagnostic = (function()
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

end)()
-- Read-only camera acquisition plus a separate explicit offline session lifecycle service.
-- No student steering/pedal or key injection. Teacher AI setters, including a finite
-- brake intervention, require explicit owned probes.
local root = ac.getFolder(ac.FolderID.AppDataLocal) .. '/flybrain/csp-camera'
io.createDir(root)
local requestPath = root .. '/request.json'
local statusPath = root .. '/status.json'
local active, lastID, nextPoll, nextStatus = nil, nil, 0, 0
local message = 'Idle: waiting for bounded capture request'
local terminalState, terminalReason = 'idle', nil
local generation = tostring(ac.getSim().windowHandle) .. ':' .. tostring(os.preciseClock())
local lastPing = nil

-- Opt-in finite diagnostic only. Original source clocks, no hot-loop timing I/O.
local sourceTiming, pendingSourceTiming = nil, nil
local sourceTimingQuiescent
local SOURCE_TIMING_MAX_RECORDS = 131072
local function timingStart()
  if sourceTiming then return {owner=sourceTiming, started=os.preciseClock()} end
end
local function timingFinish(stage, token, succeeded, detail)
  if not token or token.owner.records_frozen then return end
  local finished, owner = os.preciseClock(), token.owner
  if #owner.records < SOURCE_TIMING_MAX_RECORDS then
    -- Fixed columns avoid a separate hash table for every measured call.
    owner.records[#owner.records+1] = {stage, token.started, finished,
      finished-token.started, succeeded, owner.update_index,
      active and active.frames or false, detail or false}
  else owner.overflow_records = owner.overflow_records + 1 end
end
local function timingCall(stage, fn, ...)
  if not sourceTiming then return fn(...) end -- original default behavior
  local token = timingStart()
  local ok, value = pcall(fn, ...)
  local detail = stage == 'write.io.save' and select(1, ...) or nil
  timingFinish(stage, token, ok and value ~= false, detail)
  if not ok then error(value, 0) end -- observation only; preserve failure
  return value
end
local function flushSourceTiming()
  if not pendingSourceTiming then return end
  local owner = pendingSourceTiming
  if not owner.records_frozen then
    owner.records_frozen, owner.frozen_clock = true, os.preciseClock()
  end
  local quiescent = sourceTimingQuiescent()
  if not quiescent then return end -- do not delay lost-camera/control cleanup
  pendingSourceTiming = nil -- one terminal attempt; never hot-loop retry
  owner.quiescence = quiescent
  owner.finished_clock = os.preciseClock()
  owner.record_count = #owner.records
  owner.flushed_after_capture = true
  -- This diagnostic write is deliberately outside the measured capture/update.
  local ok, problem = pcall(function()
    assert(io.save(owner.directory .. '/source-timing-v1.json', JSON.stringify(owner), true),
      'Cannot commit terminal source timing')
  end)
  if not ok then message = 'Source timing terminal flush failed: ' .. tostring(problem) end
end

local function write(path, value)
  local text = timingCall('write.JSON.stringify', JSON.stringify, value)
  local saved = timingCall('write.io.save', io.save, path, text, true)
  assert(saved, 'Cannot commit camera metadata')
end

-- Lifecycle commands are separate from camera requests and never come from the actor.
local sessionRoot = ac.getFolder(ac.FolderID.AppDataLocal) .. '/flybrain/csp-session'
io.createDir(sessionRoot)
local sessionEpoch, sessionPoll, sessionHeartbeat = 0, 0, 0
local sessionError, pendingSessionReceipt = nil, nil
local heartbeatWriteFailures = 0
local expertProbe, brakeSettle = nil, nil

-- Candidate-only embedded protocol; exact deployed base bound by builder.
local heartbeatProtocol = (function()
-- Standalone candidate only: not required by or deployed with FlyBrainCamera.
-- No camera dependency, filesystem I/O, API setters, or module-load mappings.
-- Host must first exclusively create/retain BOTH slots under its owned lease.
-- host_created_slots=true is a required declaration, not proof by itself: future
-- integration must bind the host exclusive-creation receipt to this owner/generation.
local ffi, bit = require('ffi'), require('bit')
local M = {}
M.MAGIC, M.VERSION, M.CAPACITY = 0x46424832, 1, 8192
M.SNAPSHOT_CAPACITY, M.PERIOD = 6144, .25
M.LAYOUT = 'unsigned int magic; unsigned int version; unsigned int sequence; '
  .. 'unsigned int length; unsigned int crc32; char payload[8192];'
local ZERO_PAYLOAD = string.rep('\0', M.CAPACITY)
local crcTable = {}
for i = 0, 255 do
  local value = i
  for _ = 1, 8 do
    value = bit.band(value, 1) ~= 0
      and bit.bxor(bit.rshift(value, 1), 0xEDB88320) or bit.rshift(value, 1)
  end
  crcTable[i] = value
end

local function finite(value)
  return type(value) == 'number' and value == value and math.abs(value) ~= math.huge
end

function M.crc32(text)
  assert(type(text) == 'string', 'CRC32 requires immutable bytes')
  local crc = -1
  for i = 1, #text do
    crc = bit.bxor(bit.rshift(crc, 8), crcTable[bit.band(bit.bxor(crc, text:byte(i)), 255)])
  end
  crc = bit.bnot(crc)
  return crc < 0 and crc + 4294967296 or crc
end

function M.channel_name(owner_id, index)
  assert(type(owner_id) == 'string' and #owner_id == 32 and owner_id:match('^[0-9a-f]+$'),
    'Fresh 32-hex host owner ID required')
  assert(index == 0 or index == 1, 'Exactly two session heartbeat slots')
  return 'FlyBrainSessionHeartbeat_' .. owner_id .. '_' .. tostring(index)
end

local Publisher = {}
Publisher.__index = Publisher

function Publisher:close()
  if self.closed then return end
  self.closed, self.retired = true, true
  -- References opened by this instance are ours to close, including on failure.
  -- Disposing a reference never clears or resets a pre-existing map's bytes.
  for index = 0, 1 do
    if self.slots[index] then
      ac.disposeMemoryMappedFile(self.slots[index])
      self.slots[index] = nil
    end
  end
  self.status = 'closed'
end

function Publisher:record()
  return {schema='flybrain.session_heartbeat.publisher.v1', owner_id=self.owner_id,
    generation=self.generation, status=self.status, closed=self.closed, retired=self.retired,
    published_count=self.published_count, maximum_snapshots=self.max_snapshots,
    started_clock=self.started_clock, deadline_clock=self.deadline_clock,
    last_source_clock=self.last_source_clock, last_error=self.last_error,
    period_seconds=M.PERIOD, camera_independent=true, actuator_messages_sent=0,
    filesystem_writes=0, host_exclusive_creation_required=true}
end

function Publisher:publish(snapshot)
  assert(not self.closed and not self.retired, 'Heartbeat publisher closed or retired')
  local now = os.preciseClock()
  if not finite(now) or now < self.started_clock then
    self.status, self.retired = 'failed', true
    self.last_error = 'Source clock changed origin or is invalid'
    error(self.last_error, 0)
  end
  if now > self.deadline_clock then
    self.status, self.retired = 'expired', true
    return nil, 'expired'
  end
  if self.published_count >= self.max_snapshots then
    self.status, self.retired = 'budget_exhausted', true
    return nil, 'budget_exhausted'
  end
  if now < self.next_due_clock then return nil, 'not_due' end
  local wireSource
  local ok, result = pcall(function()
    assert(type(snapshot) == 'table' and snapshot.schema == 'flybrain.csp.session.status.v1'
      and snapshot.generation == self.generation, 'Original same-generation session snapshot required')
    local source = snapshot.script_clock
    assert(finite(source) and source >= 0 and source <= now
      and (self.last_source_clock == nil or source > self.last_source_clock),
      'Original source clock must be finite, strictly increasing and not future')
    local body = JSON.stringify(snapshot)
    assert(type(body) == 'string' and #body > 0 and #body <= M.SNAPSHOT_CAPACITY,
      'Original snapshot JSON exceeds bounded payload')
    local count = self.published_count + 1
    local packet = JSON.stringify({schema='flybrain.session_heartbeat.packet.v1',
      owner_id=self.owner_id, generation=self.generation, sequence=count,
      source_clock=source, snapshot_json=body})
    assert(type(packet) == 'string' and #packet > 0 and #packet <= M.CAPACITY,
      'Heartbeat envelope exceeds bounded payload')
    -- Bind ACKs to the actual immutable emitted bytes. Internal raw clock still
    -- owns monotonic/deadline checks; no tolerance is applied to host echo.
    local emitted = JSON.parse(packet)
    local nested = JSON.parse(emitted.snapshot_json)
    wireSource = emitted.source_clock
    assert(finite(wireSource) and nested.script_clock == wireSource
      and emitted.snapshot_json == body and emitted.sequence == count,
      'Emitted envelope/body wire source clock differs')
    assert(self.last_wire_source_clock == nil or wireSource > self.last_wire_source_clock,
      'Emitted source clock regressed/repeated after serialization')
    assert(math.abs(wireSource-source) <= self.wire_clock_rounding_margin_ns/1000000000,
      'Serialization error exceeds existing source-clock rounding margin')
    local crc, slot = M.crc32(packet), self.slots[count % 2]
    -- Publish odd before any header/payload mutation and even only after CRC+bytes.
    slot.sequence = count * 2 - 1
    slot.magic, slot.version, slot.length, slot.crc32 = M.MAGIC, M.VERSION, #packet, crc
    ffi.copy(slot.payload, packet, #packet)
    slot.sequence = count * 2
    self.published_count, self.last_source_clock = count, source
    self.last_wire_source_clock = wireSource
    self.last_wire_delta_seconds = wireSource-source
    self.next_due_clock, self.status = now + M.PERIOD, 'publishing'
    return count
  end)
  if not ok then
    self.status, self.retired, self.last_error = 'failed', true, tostring(result)
    error(result, 0) -- never retry an uncertain publication in this instance
  end
  return result, nil, wireSource
end

function M.new(config)
  assert(type(config) == 'table' and config.host_created_slots == true,
    'Bound host exclusive-creation declaration required before opening slots')
  M.channel_name(config.owner_id, 0)
  assert(type(config.generation) == 'string' and #config.generation > 0
    and #config.generation <= 128, 'Explicit bounded generation required')
  assert(finite(config.duration_seconds) and config.duration_seconds >= M.PERIOD
    and config.duration_seconds <= 60, 'Finite heartbeat lifetime of at most60 seconds required')
  assert(type(config.max_snapshots) == 'number' and config.max_snapshots % 1 == 0
    and config.max_snapshots >= 1 and config.max_snapshots <= 256,
    'Finite at most256 heartbeat snapshots required')
  assert(type(ac.writeMemoryMappedFile) == 'function'
    and type(ac.disposeMemoryMappedFile) == 'function', 'Installed owned MMF API required')
  assert(type(config.wire_clock_rounding_margin_ns) == 'number'
    and config.wire_clock_rounding_margin_ns % 1 == 0
    and config.wire_clock_rounding_margin_ns >= 1
    and config.wire_clock_rounding_margin_ns <= 1000000,
    'Existing cached clock rounding margin required; never widen it')
  local now = os.preciseClock()
  assert(finite(now) and now >= 0, 'Original source clock required')
  local self = setmetatable({owner_id=config.owner_id, generation=config.generation,
    wire_clock_rounding_margin_ns=config.wire_clock_rounding_margin_ns,
    max_snapshots=config.max_snapshots, started_clock=now, deadline_clock=now+config.duration_seconds,
    next_due_clock=now, published_count=0, status='ready', slots={}, closed=false,
    retired=false}, Publisher)
  local ok, problem = pcall(function()
    for index = 0, 1 do
      local slot = ac.writeMemoryMappedFile(M.channel_name(self.owner_id, index), M.LAYOUT, false)
      self.slots[index] = slot
      assert(tonumber(slot.magic) == 0 and tonumber(slot.version) == 0
        and tonumber(slot.sequence) == 0 and tonumber(slot.length) == 0
        and tonumber(slot.crc32) == 0 and ffi.string(slot.payload, M.CAPACITY) == ZERO_PAYLOAD,
        'Host-precreated slot is not pristine; never reset existing contents')
    end
  end)
  if not ok then self:close(); error(problem, 0) end
  return self
end

return M

end)()
local heartbeatService = (function()
-- Candidate lifecycle glue. No module-load I/O, no camera or new control API.
-- The caller supplies the original cleanup and quiescent/synchronous sink.
local ffi = require('ffi')
local M = {}
M.CONTROL_LAYOUT = 'unsigned int magic; unsigned int version; unsigned int sequence; '
  .. 'unsigned int length; unsigned int crc32; char payload[2048];'
local CONTROL_MAGIC, CONTROL_CAPACITY, MAX_CONTROL = 0x46424843, 2048, 512
local CONTROL_SCHEMA = 'flybrain.session_heartbeat.control.v1'
local Service = {}
Service.__index = Service

local function controlName(owner, direction)
  return 'FlyBrainSessionHeartbeatControl_' .. owner .. '_' .. direction
end

function Service:ack(phase, extra)
  assert(self.ack_count < MAX_CONTROL, 'Finite control ACK budget exhausted')
  local value = {schema=CONTROL_SCHEMA, owner_id=self.plan.owner_id,
    generation=self.plan.generation, lease_sha256=self.plan.lease_sha256,
    phase=phase, original_clock=os.preciseClock(),
    committed_sequence=self.committed_sequence,
    committed_source_clock=self.committed_source_clock,
    last_raw_source_clock_text=self.last_raw_source_clock_text,
    last_wire_source_clock_text=self.last_wire_source_clock_text,
    last_wire_delta_text=self.last_wire_delta_text}
  for key, item in pairs(extra or {}) do value[key] = item end
  local text = JSON.stringify(value)
  assert(#text > 0 and #text <= CONTROL_CAPACITY, 'Bounded lifecycle ACK required')
  self.ack_count = self.ack_count + 1
  local slot = self.lua_control
  slot.sequence = self.ack_count*2-1
  slot.magic, slot.version, slot.length, slot.crc32 = CONTROL_MAGIC, 1, #text, self.proto.crc32(text)
  ffi.copy(slot.payload, text, #text)
  slot.sequence = self.ack_count*2
end

function Service:readHost()
  local slot = self.host_control
  -- SDK return T may be a struct or struct*. Never sizeof(pointer), cast or
  -- copy the enclosing object. Existing deployed protocol uses these fields.
  local function sample()
    local sequence = tonumber(slot.sequence)
    local magic, version = tonumber(slot.magic), tonumber(slot.version)
    local length, crc = tonumber(slot.length), tonumber(slot.crc32)
    local payload = ffi.string(slot.payload, CONTROL_CAPACITY)
    if sequence ~= tonumber(slot.sequence) then return nil end
    return {sequence=sequence,magic=magic,version=version,length=length,crc=crc,payload=payload}
  end
  local first, second = sample(), sample()
  if not first or not second then return nil end
  for key, value in pairs(first) do if value ~= second[key] then return nil end end
  local sequence, length = first.sequence, first.length
  if sequence == 0 or sequence % 2 ~= 0 then return nil end
  assert(first.magic == CONTROL_MAGIC and first.version == 1
    and sequence/2 >= 1 and sequence/2 <= MAX_CONTROL
    and length >= 1 and length <= CONTROL_CAPACITY, 'Stable host control header differs')
  local text = first.payload:sub(1, length)
  assert(self.proto.crc32(text) == first.crc, 'Host control CRC differs')
  local value = JSON.parse(text)
  assert(value.schema == CONTROL_SCHEMA and value.owner_id == self.plan.owner_id
    and value.generation == self.plan.generation
    and value.lease_sha256 == self.plan.lease_sha256, 'Foreign host control identity')
  if sequence/2 <= self.host_sequence then return nil end
  self.host_sequence = sequence/2
  return value
end

function Service:fail(reason)
  if self.phase == 'failed' then return end
  self.phase, self.failure = 'failed', tostring(reason)
  -- Original owned cleanup, once. A transport failure never enables a second
  -- activation, widens a timer/cap, or silently resumes synchronous heartbeat.
  pcall(self.cleanup, self.failure)
  pcall(self.ack, self, 'failed', {reason=self.failure})
end

function Service:tick(now)
  if self.phase == 'closed' or self.phase == 'failed' then return end
  local ok, problem = pcall(function()
    assert(now <= self.plan.deadline_source_clock, 'Original heartbeat lease expired')
    local command = self:readHost()
    if command then
      assert(command.command == 'run' or command.command == 'retire'
        or command.command == 'writer_fenced' or command.command == 'release_confirmed',
        'Unknown heartbeat host command')
      local count, source = command.committed_sequence, command.committed_source_clock
      assert(type(count) == 'number' and count % 1 == 0
        and count >= self.committed_sequence and count <= self.publisher.published_count,
        'Host cannot acknowledge an uncommitted or regressing source sequence')
      if count > 0 then
        assert(source == self.published_clocks[count], 'Host ACK must retain original source clock')
        self.committed_sequence, self.committed_source_clock = count, source
        self.committed_raw_source_clock = self.published_raw_clocks[count]
      else assert(source == nil, 'Uncommitted source clock cannot refresh freshness') end
      if command.command == 'retire' then self.retire_requested = true end
      if command.command == 'writer_fenced' then
        assert(self.phase == 'final' and count == self.final_sequence,
          'Only the committed final source sequence can hand off to sync')
        assert(self.quiescent(), 'Synchronous return requires original quiescence')
        local state = self.state()
        local text = self.sync_commit(state)
        self.phase = 'sync_return'
        self:ack('sync_return', {sync_crc32=self.proto.crc32(text), sync_length=#text,
          sync_source_clock=state.script_clock, final_sequence=self.final_sequence})
      elseif command.command == 'release_confirmed' then
        assert(self.phase == 'sync_return', 'Lease release cannot precede synchronous return')
        self.phase = 'closed'
        self:close()
        return
      end
    end
    if self.phase ~= 'sync_return' then
      -- ACK delivery itself does not redate the source snapshot.
      local original = self.committed_raw_source_clock or self.started_clock
      assert(now-original <= 2, 'Host commit/source freshness timeout')
    end
  end)
  if not ok then self:fail(problem) end
end

function Service:heartbeat(state, force)
  if self.phase == 'closed' then return false end
  if self.phase == 'sync_return' then return true end -- freeze ACK-bound SYNC bytes until release
  if self.phase == 'final' or self.phase == 'failed' then return true end
  local ok, problem = pcall(function()
    assert(state.epoch == self.plan.epoch, 'Session epoch changed while armed; never rebase')
    -- Original pollSession resets sessionHeartbeat=0 for task transitions and
    -- receipt recovery. Its forced refresh must not be throttled a second time.
    if force then self.publisher.next_due_clock = 0 end
    local count, why, wireSource = self.publisher:publish(state)
    assert(count or why == 'not_due', 'Heartbeat publisher retired: '..tostring(why))
    if count then
      self.published_clocks[count] = wireSource
      self.published_raw_clocks[count] = state.script_clock
      self.last_raw_source_clock_text = string.format('%.17g', state.script_clock)
      self.last_wire_source_clock_text = string.format('%.17g', wireSource)
      self.last_wire_delta_text = string.format('%.17g', wireSource-state.script_clock)
      if self.retire_requested and self.quiescent() then
        self.final_sequence, self.final_source_clock, self.phase = count, wireSource, 'final'
        self:ack('final', {final_sequence=count, final_source_clock=wireSource})
      else self.phase = 'publishing'; self:ack('publishing') end
    end
  end)
  if not ok then self:fail(problem) end
  return true -- own the sink even after failure; never silent sync fallback
end

function Service:close()
  if self.closed then return end
  self.closed = true
  if self.publisher then self.publisher:close() end
  if self.host_control then ac.disposeMemoryMappedFile(self.host_control) end
  if self.lua_control then ac.disposeMemoryMappedFile(self.lua_control) end
end

function M.new(proto, plan, lease, callbacks)
  assert(plan.schema == 'flybrain.session_heartbeat.arm.v1'
    and lease.schema == 'flybrain.session_heartbeat.file_lease.v1'
    and plan.owner_id == lease.owner_id and plan.generation == lease.generation
    and plan.deadline_qpc_ns == lease.deadline_qpc_ns
    and plan.registered_qpc_ns == lease.registered_qpc_ns
    and plan.clock_domain == 'host_perf_counter_QueryPerformanceCounter_ns'
    and lease.clock_domain == plan.clock_domain
    and plan.host_created_slots == true and plan.host_ready == true
    and type(plan.lease_sha256) == 'string'
    and #plan.lease_sha256 == 64, 'Bound exclusive same-generation lease/arm plan required')
  local now = os.preciseClock()
  assert(type(plan.deadline_source_clock) == 'number' and plan.deadline_source_clock > now
    and plan.deadline_source_clock <= now+60, 'Original finite source deadline required')
  assert(callbacks.quiescent(), 'Paused AIoff quiescence required before transport arm')
  local self = setmetatable({proto=proto, plan=plan, started_clock=now,
    state=callbacks.state, quiescent=callbacks.quiescent,
    cleanup=callbacks.cleanup, sync_commit=callbacks.sync_commit,
    phase='armed', host_sequence=0, ack_count=0, committed_sequence=0,
    published_clocks={}, published_raw_clocks={}, closed=false}, Service)
  local ok, problem = pcall(function()
    self.host_control = ac.readMemoryMappedFile(controlName(plan.owner_id, 'host'), M.CONTROL_LAYOUT, false)
    self.lua_control = ac.writeMemoryMappedFile(controlName(plan.owner_id, 'lua'), M.CONTROL_LAYOUT, false)
    assert(tonumber(self.lua_control.sequence) == 0 and tonumber(self.lua_control.magic) == 0,
      'Host-precreated Lua control map must be pristine; never reset it')
    self.publisher = proto.new({owner_id=plan.owner_id, generation=plan.generation,
      wire_clock_rounding_margin_ns=plan.clock_rounding_margin_ns,
      duration_seconds=plan.deadline_source_clock-now, max_snapshots=plan.max_snapshots,
      host_created_slots=true})
    self:tick(now)
    assert(self.phase ~= 'failed', self.failure)
    assert(self.host_sequence > 0, 'Initial host control bytes are not ready')
    self:ack('armed')
  end)
  if not ok then self:close(); error(problem, 0) end
  return self
end

return M

end)()
local heartbeatLeasePath = sessionRoot .. '/status.json.flybrain-heartbeat-owner'
local heartbeatArmPath = sessionRoot .. '/heartbeat-transport.json'
local heartbeatTransport, heartbeatArmPoll, heartbeatArmOwner = nil, 0, nil
local heartbeatPending = nil
local heartbeatActiveLeaseText, heartbeatCompletedLease = nil, nil
-- A reload cannot assume the old external OS writer was cancelled.
local heartbeatOrphanGuard = io.fileExists(heartbeatLeasePath)
ac.onSessionStart(function() sessionEpoch = sessionEpoch + 1 end)

local function optionalBoolean(getter)
  local ok, value = pcall(getter)
  if ok and type(value) == 'boolean' then return value end
  return nil -- unavailable is distinct from a verified false value
end

local function optionalVector(getter)
  local ok, value = pcall(function()
    local v = getter()
    if v == nil then return nil end
    return {v.x, v.y, v.z}
  end)
  if not ok or not value then return nil end
  for i = 1, 3 do
    if type(value[i]) ~= 'number' or value[i] ~= value[i]
        or math.abs(value[i]) == math.huge then return nil end
  end
  return value -- diagnostic only; never infer a missing render origin from car pose
end

local function sessionState()
  local timing = timingStart()
  local sim, car = ac.getSim(), ac.getCar(0)
  local state = {schema = 'flybrain.csp.session.status.v1', generation = generation,
    epoch = sessionEpoch, window_handle = tostring(sim.windowHandle),
    script_clock = os.preciseClock(), is_online = sim.isOnlineRace,
    in_menu = sim.isInMainMenu, paused = sim.isPaused, live = sim.isLive,
    speed_kmh = car and car.speedKmh or nil, gas = car and car.gas or nil,
    brake = car and car.brake or nil, car_id = ac.getCarID(0), track_id = ac.getTrackID(),
    camera_active = active ~= nil,
    camera_request_id = active and active.request.id,
    ai_controlled = optionalBoolean(function() return car.isAIControlled end),
    ideal_line_shown = optionalBoolean(function() return sim.idealLineShown end),
    physics_api_allowed = optionalBoolean(function() return physics.allowed() end),
    expert_probe_active = expertProbe ~= nil or brakeSettle ~= nil,
    brake_settle_active = brakeSettle ~= nil,
    expert_probe_id = expertProbe and expertProbe.id,
    external_task = expertProbe and expertProbe.pace and {
      schema='flybrain.external_speed_task.v1', phase=expertProbe.pace_phase,
      desired_speed_m_s=expertProbe.pace_goal_kmh/3.6,
      source='prospectively fixed engineering pace task', actor_input=true} or nil,
    error = sessionError or (pendingSessionReceipt and pendingSessionReceipt.last_error),
    pending_receipt_id = pendingSessionReceipt and pendingSessionReceipt.value.id,
    heartbeat_write_failures = heartbeatWriteFailures,
    renderer_origin_shift_diagnostic = optionalVector(function() return sim.originShift end),
    receipt_protocol = 'accepted_plus_immutable_terminal_v2',
    capabilities = {disable_ideal_line = ac.trySimKeyPressCommand ~= nil,
      restart_engine = type(ac.restartAssettoCorsa) == 'function',
      restart = ac.tryToRestartSession ~= nil,
      start = ac.tryToStart ~= nil, pause = ac.tryToPause ~= nil, expert_probe = true,
      expert_motion_probe = true, expert_collect30 = true,
      expert_motion_speed30 = true, expert_collect30_speed30 = true,
      expert_motion_speed60 = true, expert_collect30_speed60 = true,
      expert_pace60_100_stop30 = type(physics) == 'table'
        and type(physics.setAIStopCounter) == 'function',
      expert_brake_pulse30 = type(physics) == 'table'
        and type(physics.setAIStopCounter) == 'function',
      expert_speed_recovery15 = true, expert_speed20_recovery15 = true,
      expert_lateral_recovery15 = true, expert_moving_handoff3 = true,
      expert_moving_handoff30 = true, brake_settle_after_teacher = true},
    api_types = {restart_engine = type(ac.restartAssettoCorsa),
      restart = type(ac.tryToRestartSession), start = type(ac.tryToStart),
      pause = type(ac.tryToPause)}}
  timingFinish('sessionState', timing, true)
  return state
end

local function flushSessionReceipt()
  if not pendingSessionReceipt then return end
  local pending = pendingSessionReceipt
  local ok, err = pcall(write, pending.path, pending.value)
  if ok then
    pendingSessionReceipt = nil
    sessionHeartbeat = 0 -- publish recovery before another command can use stale metadata
  else pending.last_error = tostring(err) end
end

local function finite(v)
  return type(v) == 'number' and v == v and math.abs(v) ~= math.huge
end

sourceTimingQuiescent = function()
  if active or expertProbe or brakeSettle or pendingSessionReceipt then return nil end
  local ok, state = pcall(function()
    local sim, car = ac.getSim(), ac.getCar(0)
    if not car or sim.isPaused ~= true or car.isAIControlled ~= false
        or not finite(car.speedKmh) or math.abs(car.speedKmh) >= .1
        or not finite(car.gas) or car.gas >= .02
        or not finite(car.brake) or car.brake <= .98 then return nil end
    return {observed_clock=os.preciseClock(), paused=true, ai_controlled=false,
      speed_kmh=car.speedKmh, gas=car.gas, brake=car.brake,
      camera_inactive=true, expert_probe_inactive=true, brake_settle_inactive=true,
      pending_session_receipt_inactive=true}
  end)
  return ok and state or nil
end

local function probeSnapshot()
  local state = sessionState()
  local car = ac.getCar(0)
  state.steer_degrees = car and car.steer or nil
  state.wheels_outside = car and car.wheelsOutside or nil
  local trackLength = ac.getSim().trackLengthM
  if car and finite(car.splinePosition) and finite(car.lapCount) and finite(trackLength)
      and car.splinePosition >= 0 and car.splinePosition <= 1
      and car.lapCount >= 0 and car.lapCount % 1 == 0 and trackLength > 0 then
    state.diagnostic_route = {spline_position=car.splinePosition, lap_count=car.lapCount,
      track_length_m=trackLength,
      lap_valid=optionalBoolean(function() return car.isLapValid end), actor_input=false}
  end
  local ok, position = pcall(function()
    local p = ac.getCar(0).position
    return {p.x, p.y, p.z}
  end)
  if ok then
    for i = 1, 3 do
      local v = position[i]
      if not finite(v) then return state end
    end
    state.position = position
  end
  return state
end

-- Corrective recovery only: let a known stopped teacher's native packet settle
-- before pausing again. No autopilot, steering, throttle or brake setters.
local function pollBrakeSettle(now)
  if not brakeSettle then return end
  local p, state = brakeSettle, probeSnapshot()
  local squared = 0
  if state.position then
    for i=1,3 do squared = squared + (state.position[i]-p.before.position[i])^2 end
  else squared = math.huge end
  p.maximum_delta = math.max(p.maximum_delta, math.sqrt(squared))
  if #p.samples < 128 then
    table.insert(p.samples, {script_clock=now, speed_kmh=state.speed_kmh,
      gas=state.gas, brake=state.brake, paused=state.paused, ai_controlled=state.ai_controlled})
  end
  local safe = state.epoch == p.before.epoch and not state.is_online
    and state.car_id == p.before.car_id and state.track_id == p.before.track_id
    and state.ai_controlled == false and not state.camera_active
    and finite(state.speed_kmh) and math.abs(state.speed_kmh) < .2
    and finite(state.gas) and state.gas < .02
    and finite(state.brake) and state.brake > .98 and p.maximum_delta <= .02
  if not safe then p.failed = 'Brake-settle envelope changed' end
  if not p.pause_issued and (p.failed or now >= p.deadline
      or now-p.started >= .12 and math.abs(state.speed_kmh or 1e9) < .01) then
    p.pause_issued = true -- even an uncertain API call is never repeated
    p.settled = safe and not state.paused and now <= p.deadline
      and now-p.started >= .12 and math.abs(state.speed_kmh or 1e9) < .01
    local ok, returned = pcall(ac.tryToPause, true)
    p.pause_ok = ok and returned == true
    if not p.pause_ok then sessionError = 'Brake-settle pause outcome unknown' end
  end
  if p.pause_issued then
    local after = probeSnapshot()
    -- tryToPause can return before the next simulation update publishes paused.
    -- Observe the original request; never repeat the setter or report a stale after-state.
    if not after.paused and now < p.deadline+1 then return end
    if not after.paused then sessionError = 'Brake-settle pause was not observed' end
    local passed = p.settled and p.pause_ok and after.paused
      and after.ai_controlled == false and math.abs(after.speed_kmh or 1e9) < .01
      and (after.gas or 1e9) < .02 and (after.brake or -1) > .98
    pendingSessionReceipt = {path=p.terminal_path, value={
      status=passed and 'api_returned_true' or 'api_returned_false',
      command='brake-settle-after-teacher', id=p.id, before=p.before, after=after,
      predecessor_id=p.predecessor_id, duration_seconds=now-p.started,
      maximum_position_delta_m=p.maximum_delta, samples=p.samples,
      brake_settle_passed=passed, control_setters_sent=0, ui_actions_sent=0,
      actuator_messages_sent=0, reason=p.failed}}
    brakeSettle = nil
  end
end

local function beginBrakeSettle(request, before, now, resultPath, terminalPath)
  local id = request.terminal_teacher_request_id
  local previous
  if type(id) == 'string' and #id == 32 and id:match('^[a-f0-9]+$') then
    local text = io.load(sessionRoot .. '/' .. id .. '.terminal.json')
    if text then
      local ok, value = pcall(JSON.parse, text)
      if ok then previous = value end
    end
  end
  local after = previous and previous.after
  local valid = request.schema == 'flybrain.csp.session.request.v1'
    and request.generation == generation and request.epoch == sessionEpoch
    and finite(request.expires_clock) and request.expires_clock >= now
    and request.expires_clock <= now+3 and not before.error
    and not before.is_online and not before.camera_active and before.paused
    and not before.in_menu and before.ai_controlled == false and before.position
    and before.car_id == 'bmw_z4_gt3' and before.track_id == 'monza'
    and math.abs(before.speed_kmh or 1e9) < .1
    and (before.gas or 1e9) < .02 and (before.brake or -1) > .98
    and request.native_residual_after_teacher_verified == true
    and finite(request.native_residual_speed_kmh)
    and request.native_residual_speed_kmh >= .1 and request.native_residual_speed_kmh < .2
    and previous and previous.status == 'api_returned_true'
    and previous.command == 'expert-collect30-speed30' and after
    and after.window_handle == before.window_handle and after.epoch == before.epoch
    and after.paused and after.ai_controlled == false
    and after.car_id == before.car_id and after.track_id == before.track_id
  if not valid then
    write(resultPath, {status='rejected', id=request.id, reason='Unverified teacher residual recovery'})
    return
  end
  write(resultPath, {status='accepted', id=request.id, command=request.command, before=before})
  brakeSettle = {id=request.id, before=before, started=now, deadline=now+.35,
    terminal_path=terminalPath, predecessor_id=id, maximum_delta=0, samples={}}
  local ok, returned = pcall(ac.tryToPause, false)
  if not ok or returned ~= true then brakeSettle.failed = 'Resume outcome unknown' end
end

local function probeCall(name, callback)
  local started = os.preciseClock() -- callback boundary, not physical engine time
  local ok, err = pcall(callback)
  local finished = os.preciseClock()
  table.insert(expertProbe.calls, {name = name, returned = ok,
    started_clock = started, finished_clock = finished,
    script_clock = os.preciseClock(),
    error = not ok and tostring(err) or nil})
  if not ok then
    expertProbe.api_unknown = true
    sessionError = 'Expert probe API outcome unknown: ' .. tostring(err)
  end
  return ok
end

local function stopExpertProbe(now, reason)
  if not expertProbe or expertProbe.cleanup_issued then return end
  expertProbe.cleanup_issued = true -- never replay an uncertain setter
  expertProbe.cleanup_clock = now
  expertProbe.stop_deadline = now + (expertProbe.high_speed and 8
    or (expertProbe.motion and 4 or 1.5))
  expertProbe.stop_reason = reason
  if not expertProbe.motion and not ac.getSim().isPaused then
    expertProbe.pause_violation = true
    probeCall('emergency_pause', function() ac.tryToPause(true) end)
  end
  probeCall('disable_ai', function() physics.setCarAutopilot(false, false) end)
  if expertProbe.brake_pulse_issued then
    probeCall('clear_ai_brake_pulse', function() physics.setAIStopCounter(0, 0) end)
  end
  if expertProbe.pace_stop_issued and not expertProbe.pace_stop_released then
    expertProbe.pace_stop_released = true -- never retry an uncertain clear
    probeCall('clear_pace_stop_on_cleanup', function() physics.setAIStopCounter(0, 0) end)
  end
end

-- Explicit external speed task, not an optimal racing policy or natural AI label.
local function pollPaceTask(now, state)
  local p = expertProbe
  if p.pace_stop_issued and not p.pace_stop_released then
    p.pace_brake_peak = math.max(p.pace_brake_peak, state.brake or 0)
    p.pace_min_speed = math.min(p.pace_min_speed, state.speed_kmh)
    if state.speed_kmh < 1 and (state.brake or 0) > .05 then p.pace_stop_seen = true end
    if state.speed_kmh > 20 and (state.brake or 0) > .05
        and state.ai_controlled == true and not state.paused then
      p.pace_moving_brake_seen = true
    end
  end
  local elapsed = now-p.started_clock
  if elapsed >= 5 and not p.pace_boost_issued then
    if elapsed >= 15 then stopExpertProbe(now, 'missed pace acceleration phase'); return end
    p.pace_boost_issued = true
    p.pace_phase, p.pace_goal_kmh = 'accelerate100', 100
    if probeCall('pace_set100', function() physics.setAITopSpeed(0, 100) end) then
      p.current_speed_hint = 100
    end
    table.insert(p.pace_transitions, {script_clock=now, phase=p.pace_phase, desired_speed_m_s=100/3.6})
    sessionHeartbeat = 0
  elseif elapsed >= 15 and not p.pace_stop_issued then
    if elapsed >= 23 or state.speed_kmh < 40 or state.speed_kmh > 110 then
      p.motion_violation = true
      stopExpertProbe(now, 'pace stop requires observed40to110kmh'); return
    end
    p.pace_stop_issued, p.pace_stop_clock = true, now
    p.pace_before_stop_speed, p.pace_min_speed = state.speed_kmh, state.speed_kmh
    p.pace_phase, p.pace_goal_kmh = 'stop0', 0
    table.insert(p.pace_transitions, {script_clock=now, phase=p.pace_phase, desired_speed_m_s=0})
    probeCall('pace_force_stop8', function() physics.setAIStopCounter(0, 8) end)
    sessionHeartbeat = 0
  elseif elapsed >= 23 and not p.pace_restart_issued then
    if not p.pace_stop_seen or not p.pace_moving_brake_seen or p.pace_brake_peak <= .05
        or p.pace_min_speed > p.pace_before_stop_speed-5 then
      p.motion_violation = true
      stopExpertProbe(now, 'pace moving brake and stop response not observed'); return
    end
    p.pace_restart_issued, p.pace_stop_released = true, true
    p.pace_phase, p.pace_goal_kmh = 'accelerate60', 60
    table.insert(p.pace_transitions, {script_clock=now, phase=p.pace_phase, desired_speed_m_s=60/3.6})
    if probeCall('pace_release_stop', function() physics.setAIStopCounter(0, 0) end)
        and probeCall('pace_set60', function() physics.setAITopSpeed(0, 60) end) then
      p.current_speed_hint = 60
    end
    sessionHeartbeat = 0
  end
end

local function pollExpertProbe(now)
  if not expertProbe then return end
  local state = probeSnapshot()
  if state.ai_controlled == true then expertProbe.seen_enabled = true end
  if expertProbe.motion then
    expertProbe.maximum_speed = math.max(expertProbe.maximum_speed, math.abs(state.speed_kmh or 1e9))
    if now >= expertProbe.next_sample and #expertProbe.samples < expertProbe.sample_cap then
      expertProbe.next_sample = now + .02
      table.insert(expertProbe.samples, {script_clock=now, position=state.position,
        speed_kmh=state.speed_kmh, gas=state.gas, brake=state.brake,
        steer_degrees=state.steer_degrees, ai_controlled=state.ai_controlled,
        paused=state.paused, wheels_outside=state.wheels_outside,
        diagnostic_route=state.diagnostic_route,
        teacher_speed_hint_kmh=expertProbe.current_speed_hint,
        external_task_phase=expertProbe.pace and expertProbe.pace_phase or nil,
        desired_speed_m_s=expertProbe.pace and expertProbe.pace_goal_kmh/3.6 or nil,
        phase=expertProbe.cleanup_issued and 'stopping' or 'driving'})
    end
  end
  if state.position then
    local squared = 0
    for i = 1, 3 do
      squared = squared + (state.position[i] - expertProbe.before.position[i]) ^ 2
    end
    expertProbe.maximum_position_delta = math.max(expertProbe.maximum_position_delta,
                                                 math.sqrt(squared))
  else expertProbe.position_unknown = true end
  if expertProbe.motion then
    if not expertProbe.cleanup_issued then
      if io.fileExists(expertProbe.cancel_path) then
        expertProbe.motion_violation = true
        stopExpertProbe(now, 'owned client cancellation')
      elseif state.epoch ~= expertProbe.before.epoch or state.is_online
          or state.car_id ~= expertProbe.before.car_id or state.track_id ~= expertProbe.before.track_id
          or not state.position or state.ai_controlled ~= true or state.paused
          or not finite(state.speed_kmh) or not finite(state.gas)
          or not finite(state.brake) or not finite(state.steer_degrees)
          or expertProbe.maximum_speed > expertProbe.hard_speed
          or expertProbe.maximum_position_delta > expertProbe.displacement_limit
          or (state.wheels_outside or 4) > 2 then
        expertProbe.motion_violation = true
        stopExpertProbe(now, 'motion envelope or identity changed')
      elseif expertProbe.camera_request_id and (not active
          or active.request.id ~= expertProbe.camera_request_id
          or now - (active.last_capture_clock or -1e9) > .15) then
        expertProbe.motion_violation = true
        stopExpertProbe(now, 'owned camera ended or became stale')
      elseif now >= expertProbe.expires or expertProbe.api_unknown then
        stopExpertProbe(now, 'owned motion probe deadline')
      elseif expertProbe.recovery and not expertProbe.retarget_issued
          and now >= expertProbe.retarget_clock then
        expertProbe.retarget_issued = true -- never repeat a possibly applied setter
        expertProbe.retarget_issued_clock = now
        if probeCall('set_recovery15_speed_limit', function() physics.setAITopSpeed(0, 15) end) then
          expertProbe.current_speed_hint = 15
        end
      elseif expertProbe.brake_pulse and not expertProbe.brake_pulse_issued
          and now >= expertProbe.brake_pulse_at and state.speed_kmh >= 25
          and state.speed_kmh <= 35 then
        expertProbe.brake_pulse_issued = true -- one setter attempt, including unknown outcome
        expertProbe.brake_pulse_clock = now
        expertProbe.brake_pulse_before_speed = state.speed_kmh
        expertProbe.brake_pulse_min_speed = state.speed_kmh
        probeCall('force_ai_brake_pulse', function() physics.setAIStopCounter(0, .4) end)
      elseif expertProbe.pace then
        pollPaceTask(now, state)
      end
    end
    if expertProbe.brake_pulse_issued and not expertProbe.cleanup_issued
        and now <= expertProbe.brake_pulse_clock + 1.5 then
      expertProbe.brake_pulse_peak = math.max(expertProbe.brake_pulse_peak, state.brake or 0)
      expertProbe.brake_pulse_min_speed = math.min(expertProbe.brake_pulse_min_speed, state.speed_kmh)
    end
    if expertProbe.recovery and not expertProbe.cleanup_issued
        and expertProbe.current_speed_hint == 15 and state.ai_controlled == true
        and now > expertProbe.retarget_issued_clock then
      if (state.speed_kmh or 0) > expertProbe.recovery_threshold then
        expertProbe.saw_recovery_overspeed = true
      end
      if expertProbe.saw_recovery_overspeed and (state.speed_kmh or 1e9) <= 17 then
        expertProbe.recovery_settled = true
      end
    end
    if expertProbe.lateral_recovery and not expertProbe.cleanup_issued
        and state.ai_controlled == true and not state.paused then
      expertProbe.last_driving_wheels_outside = state.wheels_outside
      if state.wheels_outside == 0 and expertProbe.maximum_position_delta > 5 then
        expertProbe.lateral_recovered = true
      end
    end
    if expertProbe.cleanup_issued and state.ai_controlled == false and not state.paused
        -- Leave margin for the independently sampled native physics packet before
        -- freezing publication with pause. Client admission stays strictly <0.1.
        and math.abs(state.speed_kmh or 1e9) < .01
        and (state.gas or 1e9) < .02 and (state.brake or -1) > .98
        and not expertProbe.pause_issued then
      expertProbe.pause_issued = true
      probeCall('pause_after_brake_hold', function()
        assert(ac.tryToPause(true), 'Pause API returned false')
      end)
    end
  elseif not state.paused or math.abs(state.speed_kmh or 1e9) >= .1 then
    expertProbe.pause_violation = true
    stopExpertProbe(now, 'stationary/pause condition changed')
  elseif now >= expertProbe.expires or expertProbe.api_unknown then
    stopExpertProbe(now, 'owned probe deadline')
  end
  if not expertProbe.cleanup_issued then return end
  if now > expertProbe.cleanup_clock and state.ai_controlled == false and state.paused then
    if not expertProbe.restore_issued then
      expertProbe.restore_issued = true
      probeCall('restore_default_ai_speed_limit', function() physics.setAITopSpeed(0, 1e9) end)
      if expertProbe.steer_multiplier == .9 then
        probeCall('restore_default_ai_steer_multiplier', function()
          physics.setAISteerMultiplier(0, 1)
        end)
      end
    end
    local passed = expertProbe.seen_enabled and not expertProbe.pause_violation
      and not expertProbe.position_unknown
      and not expertProbe.api_unknown
      and (expertProbe.motion and not expertProbe.motion_violation
        and expertProbe.maximum_position_delta > .1
        and expertProbe.maximum_position_delta <= expertProbe.displacement_limit
        and expertProbe.maximum_speed <= expertProbe.hard_speed
        and math.abs(state.speed_kmh or 1e9) < .1
        and (state.gas or 1e9) < .02 and (state.brake or -1) > .98
        or not expertProbe.motion and expertProbe.maximum_position_delta <= .02)
    if expertProbe.recovery and not (expertProbe.saw_recovery_overspeed
        and expertProbe.recovery_settled) then passed = false end
    local brakePulsePassed = expertProbe.brake_pulse_issued
      and expertProbe.brake_pulse_peak > .05
      and expertProbe.brake_pulse_min_speed <= expertProbe.brake_pulse_before_speed - 1
    if expertProbe.brake_pulse and not brakePulsePassed then passed = false end
    local pacePassed = expertProbe.pace_boost_issued and expertProbe.pace_stop_issued
      and expertProbe.pace_restart_issued and expertProbe.pace_stop_seen
      and expertProbe.pace_moving_brake_seen
      and expertProbe.maximum_speed >= 80
      and expertProbe.pace_brake_peak > .05
      and expertProbe.pace_min_speed <= (expertProbe.pace_before_stop_speed or 0)-5
    if expertProbe.pace and not pacePassed then passed = false end
    if expertProbe.lateral_recovery and not (expertProbe.lateral_recovered
        and expertProbe.last_driving_wheels_outside == 0
        and state.wheels_outside == 0) then passed = false end
    pendingSessionReceipt = {path = expertProbe.terminal_path, value = {
      status = expertProbe.api_unknown and 'api_outcome_unknown'
        or (passed and 'api_returned_true' or 'api_returned_false'),
      id = expertProbe.id, command = expertProbe.command, before = expertProbe.before, after = state,
      probe_seen_enabled = expertProbe.seen_enabled,
      stationary_probe_passed = not expertProbe.motion and passed,
      motion_probe_passed = expertProbe.motion and passed,
      moving_handoff_passed = expertProbe.moving_handoff and passed,
      student_owner_pid = expertProbe.student_owner_pid,
      maximum_position_delta_m = expertProbe.maximum_position_delta,
      maximum_speed_kmh = expertProbe.maximum_speed, telemetry = expertProbe.samples,
      ai_steer_multiplier = expertProbe.steer_multiplier,
      requested_ai_speed_hint_kmh = expertProbe.speed_hint,
      label_speed_hint_kmh = not expertProbe.pace
        and (expertProbe.recovery and 15 or expertProbe.speed_hint) or nil,
      explicit_pace_task = expertProbe.pace or false,
      pace_task_passed = expertProbe.pace and pacePassed and passed or false,
      pace_contract = expertProbe.pace and 'flybrain.external_speed_task.v1' or nil,
      pace_transitions = expertProbe.pace_transitions,
      pace_before_stop_speed_kmh = expertProbe.pace_before_stop_speed,
      pace_min_speed_kmh = expertProbe.pace_min_speed,
      pace_brake_peak = expertProbe.pace_brake_peak,
      pace_moving_brake_seen = expertProbe.pace_moving_brake_seen,
      recovery15_passed = expertProbe.recovery and passed,
      brake_pulse_passed = expertProbe.brake_pulse and brakePulsePassed and passed or false,
      brake_pulse_issued_clock = expertProbe.brake_pulse_clock,
      brake_pulse_before_speed_kmh = expertProbe.brake_pulse_before_speed,
      brake_pulse_min_speed_kmh = expertProbe.brake_pulse_min_speed,
      brake_pulse_peak = expertProbe.brake_pulse_peak,
      brake_pulse_duration_seconds = expertProbe.brake_pulse and .4 or nil,
      recovery_start_threshold_kmh = expertProbe.recovery_threshold,
      lateral_recovery15_passed = expertProbe.lateral_recovery and passed,
      last_driving_wheels_outside = expertProbe.last_driving_wheels_outside,
      motion_hard_speed_kmh = expertProbe.hard_speed,
      maximum_displacement_guard_m = expertProbe.displacement_limit,
      api_calls = expertProbe.calls, stop_reason = expertProbe.stop_reason,
      speed_limit_enforcement_tested = false, driving_tested = expertProbe.motion == true,
      ui_actions_sent = 0, actuator_messages_sent = 0}}
    expertProbe = nil
  elseif now >= expertProbe.stop_deadline then
    if expertProbe.motion and not state.paused and not expertProbe.pause_issued then
      expertProbe.pause_issued = true
      probeCall('emergency_pause', function() ac.tryToPause(true) end)
    end
    sessionError = 'Expert probe teardown state not verified; do not retry activation'
    pendingSessionReceipt = {path = expertProbe.terminal_path, value = {
      status = 'api_outcome_unknown', id = expertProbe.id, command = expertProbe.command,
      before = expertProbe.before, after = state, api_calls = expertProbe.calls,
      telemetry = expertProbe.samples,
      reason = sessionError, ui_actions_sent = 0, actuator_messages_sent = 0}}
    expertProbe = nil
  end
end


local function heartbeatTransitionBusy()
  return heartbeatOrphanGuard or heartbeatPending ~= nil or heartbeatCompletedLease ~= nil
    or heartbeatTransport and (
    heartbeatTransport.committed_sequence == 0 or heartbeatTransport.phase == 'failed'
    or heartbeatTransport.phase == 'final' or heartbeatTransport.phase == 'sync_return')
end

local function pollHeartbeatTransport(now)
  if heartbeatOrphanGuard then
    sessionError = 'Heartbeat orphan ownership guard: inspect/fence old host; no sync fallback'
    return true
  end
  if heartbeatTransport then
    heartbeatTransport:tick(now)
    if heartbeatTransport.phase == 'closed' then
      -- Only a verified same-writer fence+SYNC return+release can reach closed.
      -- Keep heartbeatArmOwner so the old arm file can never replay.
      if heartbeatActiveLeaseText then
        heartbeatCompletedLease = {raw=heartbeatActiveLeaseText, deadline_clock=now+2}
        heartbeatActiveLeaseText = nil
      end
      heartbeatTransport, heartbeatArmPoll = nil, 0
    else
      if heartbeatTransport.phase == 'failed' then sessionError = heartbeatTransport.failure end
      return true
    end
  end
  if heartbeatCompletedLease then
    if now > heartbeatCompletedLease.deadline_clock then
      sessionError = 'Heartbeat completed-owner lease removal timeout; no new startup'
      return true
    end
    if not io.fileExists(heartbeatLeasePath) then heartbeatCompletedLease = nil
    else
      local ok, text = pcall(io.load, heartbeatLeasePath)
      if not ok or not text or text == heartbeatCompletedLease.raw then
        return true -- release_confirmed precedes unlink; this is the known completed owner
      end
      heartbeatCompletedLease = nil -- different fresh lease, never infer an unknown old owner
    end
  end
  if heartbeatPending and heartbeatPending.failed then return true end
  if not heartbeatPending and io.fileExists(heartbeatLeasePath) then
    heartbeatPending = {started_clock=now, deadline_clock=now+2}
  end
  if heartbeatPending and now > heartbeatPending.deadline_clock then
    heartbeatPending.failed = true
    sessionError = 'Heartbeat matching ready arm plan timeout; original2s wait not extended'
    return true
  end
  if now < heartbeatArmPoll then return heartbeatPending ~= nil end
  heartbeatArmPoll = now + .1
  local loadOK, leaseText = pcall(io.load, heartbeatLeasePath)
  if not heartbeatPending then return false end
  if not loadOK or not leaseText then return true end -- never infer cancellation from missing I/O
  local leaseOK, lease = pcall(JSON.parse, leaseText)
  if not leaseOK or type(lease) ~= 'table' then return true end -- partial exclusive-create write
  local leaseValid = lease.schema == 'flybrain.session_heartbeat.file_lease.v1'
    and type(lease.owner_id) == 'string' and #lease.owner_id == 32
    and lease.owner_id:match('^[0-9a-f]+$') and lease.generation == generation
    and lease.clock_domain == 'host_perf_counter_QueryPerformanceCounter_ns'
    and type(lease.registered_qpc_ns) == 'number' and lease.registered_qpc_ns >= 0
    and type(lease.deadline_qpc_ns) == 'number'
    and lease.deadline_qpc_ns > lease.registered_qpc_ns
    and lease.deadline_qpc_ns-lease.registered_qpc_ns <= 60000000000
  if not leaseValid or heartbeatPending.lease_text and heartbeatPending.lease_text ~= leaseText then
    heartbeatPending.failed = true
    sessionError = 'Heartbeat stable foreign/invalid/changed lease; no takeover or sync fallback'
    return true
  end
  heartbeatPending.lease_text = leaseText -- raw bytes, never a reserialized lease
  local armOK, armText = pcall(io.load, heartbeatArmPath)
  if not armOK or not armText then return true end
  local planOK, plan = pcall(JSON.parse, armText)
  if not planOK or type(plan) ~= 'table' or plan.owner_id ~= lease.owner_id then return true end
  if plan.host_ready ~= true then return true end
  local planValid = plan.schema == 'flybrain.session_heartbeat.arm.v1'
    and plan.generation == generation and plan.epoch == sessionEpoch
    and plan.owner_id ~= heartbeatArmOwner
    and plan.lease_crc32 == heartbeatProtocol.crc32(leaseText)
    and type(plan.lease_sha256) == 'string' and #plan.lease_sha256 == 64
    and plan.deadline_qpc_ns == lease.deadline_qpc_ns
    and plan.registered_qpc_ns == lease.registered_qpc_ns
  if not planValid then
    heartbeatPending.failed = true
    sessionError = 'Heartbeat matching ready arm binding differs; no retry or sync fallback'
    return true
  end
  heartbeatArmOwner = plan.owner_id
  heartbeatTransport = heartbeatService.new(heartbeatProtocol, plan, lease, {
    state=sessionState, quiescent=sourceTimingQuiescent,
    cleanup=function(reason)
      stopExpertProbe(os.preciseClock(), 'heartbeat transport: '..reason)
    end,
    sync_commit=function(state)
      local text = timingCall('write.JSON.stringify', JSON.stringify, state)
      assert(timingCall('write.io.save', io.save, sessionRoot .. '/status.json', text, true),
        'Cannot commit synchronous heartbeat handoff')
      return text
    end})
  heartbeatActiveLeaseText = leaseText
  heartbeatPending = nil
  sessionHeartbeat = 0
  return true
end

local function pollSession(now)
  -- Retry metadata only, never an accepted API side effect.
  flushSessionReceipt()
  local transportOwnsSink = pollHeartbeatTransport(now)
  if now >= sessionHeartbeat then
    local forceHeartbeat = sessionHeartbeat == 0
    sessionHeartbeat = now + .25
    local timing = timingStart()
    local ok = pcall(function()
      if heartbeatTransport and transportOwnsSink then
        heartbeatTransport:heartbeat(sessionState(), forceHeartbeat)
      elseif not transportOwnsSink then write(sessionRoot .. '/status.json', sessionState()) end
    end)
    timingFinish('session_heartbeat', timing, ok)
    if not ok then heartbeatWriteFailures = heartbeatWriteFailures + 1 end
  end
  if transportOwnsSink and heartbeatTransitionBusy() then return end
  if pendingSessionReceipt or expertProbe or brakeSettle then return end
  if now < sessionPoll then return end
  sessionPoll = now + (active and .015 or .1)
  local text = io.load(sessionRoot .. '/request.json')
  if not text then return end
  local request = JSON.parse(text)
  assert(type(request.id) == 'string' and request.id:match('^[a-f0-9]+$')
    and #request.id == 32, 'Owned lifecycle command ID required')
  local resultPath = sessionRoot .. '/' .. request.id .. '.json'
  local terminalPath = sessionRoot .. '/' .. request.id .. '.terminal.json'
  if io.fileExists(resultPath) or io.fileExists(terminalPath) then return end -- never replay
  local before = sessionState()
  if request.command == 'brake-settle-after-teacher' then
    beginBrakeSettle(request, probeSnapshot(), now, resultPath, terminalPath)
    return
  end
  local isWarm20 = request.command == 'expert-speed20-recovery15'
  local isBrakePulse = request.command == 'expert-brake-pulse30'
  local isPace = request.command == 'expert-pace60-100-stop30'
  local isSpeed60 = request.command == 'expert-motion-speed60'
    or request.command == 'expert-collect30-speed60'
  local isRecovery = request.command == 'expert-speed-recovery15' or isWarm20
  local isLateralRecovery = request.command == 'expert-lateral-recovery15'
  local isHandoff30 = request.command == 'expert-moving-handoff30'
  local isHandoff = request.command == 'expert-moving-handoff3' or isHandoff30
  local isFast = request.command == 'expert-motion-speed30'
    or request.command == 'expert-collect30-speed30' or isRecovery or isBrakePulse or isSpeed60 or isPace
  local isCollect = request.command == 'expert-collect30'
    or request.command == 'expert-collect30-speed30' or isRecovery or isLateralRecovery
    or isHandoff30 or isBrakePulse or request.command == 'expert-collect30-speed60' or isPace
  local isMotion = request.command == 'expert-motion-probe' or isCollect or isFast or isHandoff
  local isProbe = request.command == 'expert-probe' or isMotion
  local isDisableLine = request.command == 'disable-ideal-line'
  local isEngineRestart = request.command == 'restart-engine'
  local motionSeconds = isCollect and 30 or 3
  local ownsCamera = isMotion and active and request.camera_request_id == active.request.id
    and active.deadline - now >= motionSeconds + ((isSpeed60 or isPace) and 9 or 5)
    and now - (active.last_capture_clock or -1e9) <= .15
  if isProbe then before = probeSnapshot() end
  local valid = request.schema == 'flybrain.csp.session.request.v1'
    and request.generation == generation and request.epoch == sessionEpoch
    and type(request.expires_clock) == 'number' and request.expires_clock >= now
    and request.expires_clock <= now + 5
    and not before.is_online and (not before.camera_active or ownsCamera) and not before.error
    and (request.camera_request_id == nil or ownsCamera)
    and (not isCollect or ownsCamera and request.ai_steer_multiplier == .9)
    and (not isFast or request.ai_steer_multiplier == .9)
    and (not isRecovery or request.recovery_target_kmh == 15 and request.warmup_seconds == 3)
    and (not isBrakePulse or type(physics.setAIStopCounter) == 'function'
      and request.brake_pulse_seconds == .4 and request.brake_pulse_after_seconds == 3)
    and (not isPace or type(physics.setAIStopCounter) == 'function'
      and request.pace_contract == 'flybrain.external_speed_task.v1'
      and request.pace_boost_after_seconds == 5 and request.pace_stop_after_seconds == 15
      and request.pace_restart_after_seconds == 23 and request.pace_stop_seconds == 8)
    and (not isLateralRecovery or request.lateral_recovery == true)
    and (not isHandoff or ownsCamera and request.ai_steer_multiplier == .9
      and request.student_owner_verified == true and finite(request.student_owner_pid)
      and request.student_owner_pid > 0 and request.student_owner_pid % 1 == 0
      and before.live and not before.paused
      and finite(before.speed_kmh) and before.speed_kmh >= 16 and before.speed_kmh <= 18.5
      and finite(before.gas) and before.gas >= 0 and before.gas <= 1
      and finite(before.brake) and before.brake >= 0 and before.brake <= 1)
    and before.car_id == 'bmw_z4_gt3' and before.track_id == 'monza'
    and (isHandoff or before.speed_kmh and math.abs(before.speed_kmh) < .1
      and before.gas and before.gas < .02 and before.brake and before.brake > .98)
    and not before.ai_controlled
    and ((not isProbe and (request.command == 'restart' or request.command == 'start'
      or request.command == 'pause' or request.command == 'resume'
      or isEngineRestart and before.paused and before.ai_controlled == false
        and type(ac.restartAssettoCorsa) == 'function'
        and request.released_controller_verified == true
        and finite(request.old_engine_pid) and request.old_engine_pid > 0
        and request.old_engine_pid % 1 == 0 and request.race_ini == nil
      or isDisableLine and (before.paused or request.allow_live_stationary == true and before.live)
        and type(before.ideal_line_shown) == 'boolean'
        and ac.trySimKeyPressCommand ~= nil))
      or (isProbe and (before.paused or isHandoff) and before.ai_controlled == false
        and before.physics_api_allowed == true and type(before.position) == 'table'
        and request.speed_limit_kmh == (isPace and 100 or isSpeed60 and 60
          or (isWarm20 and 20 or (isFast and 30 or 15)))
        and (not isMotion or request.ai_steer_multiplier == nil
          or request.ai_steer_multiplier == 1 or request.ai_steer_multiplier == .9)
        and (not isMotion and request.hold_seconds == .5
          or isMotion and request.hold_seconds == motionSeconds and before.in_menu == false
            and type(before.wheels_outside) == 'number'
            and before.wheels_outside == (isLateralRecovery and 1 or 0))))
  if not valid then
    write(resultPath, {status = 'rejected', id = request.id,
      reason = 'Expired, changed, active or nonstationary session', before = before})
    return
  end
  -- Commit before the side effect. Unknown outcomes are not automatically retried.
  write(resultPath, {status = 'accepted', id = request.id, command = request.command,
    before = before})
  if isEngineRestart then
    -- Durable intent is not a success receipt: the external owner must observe the
    -- original process handle exit and a new engine/producer. Never replay this ID.
    write(terminalPath, {status='engine_restart_intent_committed', id=request.id,
      command=request.command, old_engine_pid=request.old_engine_pid, before=before,
      ui_actions_sent=0, actuator_messages_sent=0})
    local ok, err = pcall(function() ac.restartAssettoCorsa() end)
    write(sessionRoot .. '/' .. request.id .. '.engine-call.json', {
      status=ok and 'void_api_returned' or 'api_outcome_unknown',
      id=request.id, error=not ok and tostring(err) or nil})
    return
  end
  if isProbe then
    expertProbe = {id = request.id, command=request.command, motion=isMotion,
      before = before, expires = now + request.hold_seconds,
      terminal_path = terminalPath, calls = {}, seen_enabled = false,
      camera_request_id=request.camera_request_id,
      cancel_path=sessionRoot .. '/' .. request.id .. '.stop',
      maximum_position_delta = 0, maximum_speed=0, samples={}, next_sample=now,
      sample_cap=isCollect and 2048 or 512,
      displacement_limit=isPace and 1100 or isSpeed60 and (isCollect and 720 or 80) or isWarm20 and 220
        or (isCollect and (isFast and 360 or 180) or (isFast and 40 or 20)),
      speed_hint=isPace and 60 or isSpeed60 and 60 or (isWarm20 and 20 or (isFast and 30 or 15)),
      hard_speed=isPace and 120 or isSpeed60 and 80 or (isWarm20 and 25 or (isFast and 40 or 20)),
      high_speed=isSpeed60 or isPace, started_clock=now}
    expertProbe.pace = isPace
    if isPace then
      expertProbe.pace_phase, expertProbe.pace_goal_kmh = 'accelerate60', 60
      expertProbe.pace_brake_peak, expertProbe.pace_stop_seen = 0, false
      expertProbe.pace_moving_brake_seen = false
      expertProbe.pace_transitions = {{script_clock=now, phase='accelerate60', desired_speed_m_s=60/3.6}}
    end
    expertProbe.steer_multiplier = request.ai_steer_multiplier or 1
    expertProbe.current_speed_hint = expertProbe.speed_hint
    expertProbe.brake_pulse = isBrakePulse
    expertProbe.brake_pulse_at = now + 3
    expertProbe.brake_pulse_peak = 0
    expertProbe.recovery = isRecovery
    expertProbe.recovery_threshold = isWarm20 and 18 or 20
    expertProbe.lateral_recovery = isLateralRecovery
    expertProbe.moving_handoff = isHandoff
    expertProbe.student_owner_pid = isHandoff and request.student_owner_pid or nil
    expertProbe.retarget_clock = isRecovery and now + 3 or nil
    if isMotion then
      -- Autopilot initialization may reset per-AI hints: configure after enabling,
      -- while still paused. The speed envelope remains independently observed.
      local enabled = probeCall('enable_ai', function() physics.setCarAutopilot(true, false) end)
      if enabled and expertProbe.steer_multiplier == .9 then
        enabled = probeCall('set_ai_steer_multiplier', function()
          physics.setAISteerMultiplier(0, .9)
        end)
      end
      if enabled and probeCall('set_ai_speed_limit', function()
        physics.setAITopSpeed(0, expertProbe.speed_hint)
      end) and not isHandoff then
        probeCall('resume_owned_ai', function()
          assert(ac.tryToPause(false), 'Resume API returned false')
        end)
      end
    elseif probeCall('set_ai_speed_limit', function() physics.setAITopSpeed(0, 15) end) then
      probeCall('enable_ai', function() physics.setCarAutopilot(true, false) end)
    end
    sessionHeartbeat = 0 -- publish the newly observed AI owner on the next update
    return
  end
  local success
  if request.command == 'restart' then success = ac.tryToRestartSession()
  elseif request.command == 'start' then success = ac.tryToStart(true)
  elseif isDisableLine then
    -- Sim API command only, never an OS key or window action. Do not toggle if already off.
    success = before.ideal_line_shown == false or ac.trySimKeyPressCommand('Ideal Line')
  else success = ac.tryToPause(request.command == 'pause') end
  pendingSessionReceipt = {path = terminalPath, value = {
    status = success and 'api_returned_true' or 'api_returned_false',
    id = request.id, command = request.command, before = before, after = sessionState(),
    ui_actions_sent = 0, actuator_messages_sent = 0}}
  flushSessionReceipt()
end

local function vector(v) return {v.x, v.y, v.z} end

-- BEGIN owned MMF metadata protocol: tested separately with real LuaJIT FFI.
local mmfLayout = 'unsigned int magic; unsigned int version; unsigned int sequence; '
  .. 'unsigned int length; char digest[64]; char payload[32768];'
local function mmfDigest(text)
  local digest = ac.checksumSHA256(text)
  assert(type(digest) == 'string', 'MMF SHA256 unavailable')
  if #digest == 32 then
    digest = digest:gsub('.', function(c) return string.format('%02x', string.byte(c)) end)
  end
  digest = digest:lower()
  assert(#digest == 64 and digest:match('^[0-9a-f]+$'), 'Invalid MMF SHA256 encoding')
  return digest
end
local function mmfPublish(slot, index, text)
  local ffi = require('ffi')
  assert(type(text) == 'string' and #text > 0 and #text <= 32768, 'MMF payload budget')
  assert(index >= 0 and index < 4000 and index % 1 == 0, 'MMF capture index')
  local digest = mmfDigest(text)
  slot.sequence = index * 2 + 1
  slot.magic, slot.version, slot.length = 0x46424D31, 1, #text
  ffi.copy(slot.digest, digest, 64)
  ffi.copy(slot.payload, text, #text)
  slot.sequence = index * 2 + 2
end
local function mmfRead(slot, index)
  local ffi = require('ffi')
  local sequence, length = tonumber(slot.sequence), tonumber(slot.length)
  if sequence == 0 or sequence % 2 ~= 0 or sequence < index * 2 + 2 then return nil end
  assert(sequence == index * 2 + 2, 'MMF ACK skipped an unacknowledged capture')
  assert(slot.magic == 0x46424D31 and slot.version == 1 and length > 0 and length <= 32768,
    'MMF ACK header differs')
  local text, digest = ffi.string(slot.payload, length), ffi.string(slot.digest, 64)
  if tonumber(slot.sequence) ~= sequence or tonumber(slot.length) ~= length then return nil end
  assert(mmfDigest(text) == digest, 'Stable MMF ACK checksum differs')
  return text
end
-- END owned MMF metadata protocol.

local function closeActive()
  if active then
    if active.mmfFrame then ac.disposeMemoryMappedFile(active.mmfFrame) end
    if active.mmfAck then ac.disposeMemoryMappedFile(active.mmfAck) end
    for _, view in ipairs(active.views) do
      if view.shot then view.shot:dispose() end
    end
    if active.source_timing then
      active.source_timing.capture_closed_clock = os.preciseClock()
      active.source_timing.active_capture_closed = true
      active.source_timing.final_capture_count = active.frames
      pendingSourceTiming = active.source_timing
      sourceTiming = nil
    end
    active = nil
  end
end

local function status(state, reason)
  message = state .. (reason and (': ' .. tostring(reason)) or '')
  write(statusPath, {schema = 'flybrain.csp.camera.status.v1', state = state,
    reason = reason, request_id = lastID, patch_version = ac.getPatchVersion(),
    patch_version_code = ac.getPatchVersionCode(), render_frame = ac.getSim().frame,
    generation = generation, window_handle = ac.getSim().windowHandle,
    script_clock = os.preciseClock(), actuator_messages_sent = 0})
end

-- BEGIN explicit MMF camera status lifecycle policy v1.
local CAMERA_STATUS_POLICY = 'mmf_lifecycle_only_v1'
local function acceptCameraStatusPolicy(request)
  if request.camera_status_policy == nil then return nil end
  assert(type(request.camera_status_policy) == 'string'
    and request.camera_status_policy == CAMERA_STATUS_POLICY, 'Unknown camera status policy')
  assert(request.mmf_metadata == true and request.shared_transport == true,
    'Lifecycle-only status requires explicit shared MMF transport')
  return CAMERA_STATUS_POLICY
end
local function cameraStatusPolicyEcho(owner)
  if owner and owner.camera_status_policy == CAMERA_STATUS_POLICY then
    return {schema='flybrain.csp.camera.status_policy_echo.v1', policy=CAMERA_STATUS_POLICY,
      request_id=owner.request.id, generation=generation}
  end
  return nil
end
local function writeCameraTerminal(owner, path, value)
  if owner.camera_status_policy == CAMERA_STATUS_POLICY then
    value.camera_status_policy_echo = cameraStatusPolicyEcho(owner)
  end
  write(path, value)
end
-- END explicit MMF camera status lifecycle policy v1.
local function begin(request, now)
  local acceptedStatusPolicy = acceptCameraStatusPolicy(request)
  assert(request.deferred_save_probe == nil or type(request.deferred_save_probe) == 'boolean',
    'Deferred save probe switch must be boolean')
  assert(not request.deferred_save_probe or (request.airflow_diagnostic == true
    and request.image_format == 'png' and not request.verify_png_reference),
    'Deferred save requires owned airflow diagnostic and PNG')
  assert(request.airflow_diagnostic == nil or type(request.airflow_diagnostic) == 'boolean',
    'Airflow diagnostic switch must be boolean')
  if request.airflow_diagnostic then
    local sim, car = ac.getSim(), ac.getCar(0)
    assert(request.frames >= 1 and request.frames <= 600 and sim.isPaused
      and not sim.isOnlineRace and car and car.isAIControlled == false
      and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98
      and not request.mmf_metadata and not request.shared_transport
      and not request.camera_motion_probe and not request.allow_ideal_line_probe,
      'Airflow diagnostic requires bounded offline brake-held initial conditions')
  end
  assert(not expertProbe, 'Stationary expert API probe must finish before camera acquisition')
  if request.camera_sensor_layout ~= nil then
    local sim, car = ac.getSim(), ac.getCar(0)
    assert(request.camera_sensor_layout == 'exterior_front_v3'
      and ac.getCarID(0) == 'bmw_z4_gt3' and not request.camera_motion_probe
      and not request.allow_ideal_line_probe and request.mmf_metadata == true
      and request.host_persistence_protocol == 'immutable_ram_v3'
      and request.frames <= 1200 and sim.isPaused and not sim.isOnlineRace
      and car and car.isAIControlled == false and math.abs(car.speedKmh) < .1
      and car.gas < .02 and car.brake > .98,
      'Exterior normal stream requires explicit BMW MMF v3 layout and stationary owned start')
  end
  if request.camera_motion_probe ~= nil then
    local probe, sim, car = request.camera_motion_probe, ac.getSim(), ac.getCar(0)
    assert(type(probe) == 'table' and finite(probe.yaw_left_rad_s)
      and math.abs(probe.yaw_left_rad_s) <= .25
      and probe.static_pre_seconds == .6 and probe.motion_seconds == 1.2
      and (probe.scene == 'full_scene' or probe.scene == 'track_only')
      and request.frames >= 1 and request.frames <= 80
      and request.period_seconds >= .04 and request.period_seconds <= .1
      and request.deadline_seconds <= 10 and not request.async_readback_probe
      and not request.shared_texture_probe and not request.allow_ideal_line_probe
      and sim.isPaused and not sim.isOnlineRace and car and car.isAIControlled == false
      and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98,
      'Camera-motion diagnostic requires bounded paused offline brake-held protocol')
    if probe.front_mount ~= nil then
      assert(probe.front_mount == 'bmw_z4_gt3_bumper_v1'
        and ac.getCarID(0) == 'bmw_z4_gt3' and probe.scene == 'full_scene'
        and probe.yaw_left_rad_s == 0 and request.frames <= 8,
        'Exterior eye requires explicit BMW static full-scene diagnostic of at most8 frames')
    end
  end
  assert(request.allow_ideal_line_probe == nil or type(request.allow_ideal_line_probe) == 'boolean',
    'Ideal-line diagnostic switch must be boolean')
  if request.allow_ideal_line_probe then
    local sim, car = ac.getSim(), ac.getCar(0)
    assert(request.frames <= 8 and sim.isPaused and not sim.isOnlineRace
      and car and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98,
      'Aid-on diagnostic is limited to eight paused stationary frames')
  else
    assert(ac.getSim().idealLineShown == false, 'Verified ideal-line OFF required for normal capture')
  end
  assert(request.async_readback_probe == nil or type(request.async_readback_probe) == 'boolean',
    'Async readback probe switch must be boolean')
  assert(request.shared_texture_probe == nil or type(request.shared_texture_probe) == 'boolean',
    'Shared texture probe switch must be boolean')
  assert(request.shared_transport == nil or type(request.shared_transport) == 'boolean',
    'Shared transport switch must be boolean')
  assert(request.mmf_metadata == nil or type(request.mmf_metadata) == 'boolean',
    'MMF metadata switch must be boolean')
  assert(request.source_timing_v1 == nil or type(request.source_timing_v1) == 'boolean',
    'Source timing switch must be boolean')
  if request.source_timing_v1 then
    assert(request.mmf_metadata and request.host_persistence_protocol == 'immutable_ram_v3'
      and request.frames <= 1200 and request.deadline_seconds <= 60,
      'Source timing requires separate finite MMF v3 diagnostic of at most60 seconds')
  end
  if request.mmf_metadata then
    local sim, car = ac.getSim(), ac.getCar(0)
    local pausedProbe = request.id:match('^mmf%-probe%-%x+$') and #request.id == 42
      and request.frames >= 1 and request.frames <= 8
    local ownedStream = request.id:match('^mmf%-stream%-%x+$') and #request.id == 43
      and request.frames >= 1 and request.frames <= 1200
      and request.host_persistence_protocol == 'immutable_ram_v3'
      and not request.verify_png_reference and not request.camera_motion_probe
    assert(request.shared_transport and (pausedProbe or ownedStream)
      and sim.isPaused and not sim.isOnlineRace and car and car.isAIControlled == false
      and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98,
      'MMF requires eight paused probe frames or bounded v3 stream starting paused')
  end
  if request.shared_transport then
    assert(not request.shared_texture_probe and not request.async_readback_probe
      and request.image_format == 'bmp' and request.period_seconds >= .04,
      'Shared transport requires separate lossless host BMP and40ms period')
  end
  if request.shared_texture_probe then
    assert(request.frames == 1 and not request.async_readback_probe,
      'Shared texture probe holds exactly one paired frame group')
  end
  if request.async_readback_probe or request.shared_texture_probe then
    local sim, car = ac.getSim(), ac.getCar(0)
    assert(request.frames >= 1 and request.frames <= 8 and sim.isPaused and not sim.isOnlineRace
      and car and car.isAIControlled == false and car.speedKmh < .1
      and car.gas < .02 and car.brake > .98,
      'Async readback is limited to eight paused offline brake-held probe groups')
  end
  request.image_format = request.image_format or 'png'
  assert(request.image_format == 'png' or request.image_format == 'bmp',
    'Only explicit lossless PNG/BMP transport allowed')
  assert(request.verify_png_reference == nil or type(request.verify_png_reference) == 'boolean',
    'Paired reference switch must be boolean')
  assert(not request.verify_png_reference or request.image_format == 'bmp' and request.frames <= 8,
    'Paired PNG reference is limited to eight BMP groups')
  assert(request.expected_generation == nil or request.expected_generation == generation,
    'Camera producer generation changed')
  assert(type(request.id) == 'string' and request.id:match('^[a-zA-Z0-9_-]+$')
    and #request.id <= 80, 'Invalid owned request ID')
  assert(type(request.frames) == 'number' and request.frames % 1 == 0
    and request.frames >= 1 and request.frames <= 4000, 'Finite 1..4000 frame budget required')
  assert(type(request.period_seconds) == 'number' and request.period_seconds >= .03
    and request.period_seconds <= 2, 'Finite capture period required')
  assert(type(request.deadline_seconds) == 'number' and request.deadline_seconds >= 1
    and request.deadline_seconds <= 150, 'Finite deadline required')
  assert(type(request.exposure) == 'number' and request.exposure >= .01
    and request.exposure <= 4, 'Explicit bounded exposure required')
  lastID = request.id
  local directory = root .. '/' .. request.id
  assert(not io.fileExists(directory .. '/result.json')
    and not io.fileExists(directory .. '/frame-0000.json'), 'Request directory already used')
  if request.source_timing_v1 then
    assert(not io.fileExists(directory .. '/source-timing-v1.json'), 'Source timing ledger already used')
  end
  io.createDir(directory)
  active = {directory = directory, request = request, camera_status_policy = acceptedStatusPolicy,
    frames = 0, started_clock = now,
    next = now, deadline = now + request.deadline_seconds, views = {}}
  if request.source_timing_v1 then
    sourceTiming = {schema='flybrain.csp.source_timing.v1', request_id=request.id,
      generation=generation, directory=directory, started_clock=os.preciseClock(),
      clock_domain='csp_os_preciseClock_seconds', max_records=SOURCE_TIMING_MAX_RECORDS,
      record_fields={'stage','started_clock','finished_clock','duration_seconds',
        'succeeded','update_index','capture_index','detail'},
      overflow_records=0, update_index=0, records={}, actor_input=false,
      timing_observation_only=true, per_frame_timing_disk_writes=0}
    active.source_timing = sourceTiming
  end
  if request.mmf_metadata then
    assert(type(ac.writeMemoryMappedFile) == 'function'
      and type(ac.readMemoryMappedFile) == 'function', 'Installed MMF API unavailable')
    active.mmfAck = ac.readMemoryMappedFile('FlyBrainMMF_ack_' .. request.id, mmfLayout, false)
    active.mmfFrame = ac.writeMemoryMappedFile('FlyBrainMMF_frame_' .. request.id, mmfLayout, false)
    assert(active.mmfFrame.magic == 0 and active.mmfFrame.sequence == 0,
      'MMF producer map already used; never reset existing contents')
  end
  local trackOnly = request.camera_motion_probe and request.camera_motion_probe.scene == 'track_only'
  local scene = ac.findNodes(trackOnly and 'trackRoot:yes' or 'sceneRoot:yes')
  if request.camera_motion_probe then
    assert(not scene:empty(), 'Diagnostic scene root must exist')
    active.diagnostic_scene_nodes = scene:size()
  end
  for _, name in ipairs({'front', 'left', 'right', 'rear'}) do
    local shot = ac.GeometryShot(scene, vec2(320, 180), 1, false,
      render.AntialiasingMode.YEBIS, nil,
      render.TextureFlags.Shared)
    table.insert(active.views, {id = name, shot = shot})
    shot:setName('FlyBrain_' .. name):setSky(true):setOriginalLighting(true)
      :setTransparentPass(true):setShadersType(render.ShadersType.SimplifiedWithLights)
      :setExposure(request.exposure)
  end
  status('capturing')
  terminalState, terminalReason = 'capturing', nil
end

local function asyncProbe(car, layout, now)
  local owner = active
  owner.pending, owner.pending_count = true, 4
  local index, rows, renderFrame = owner.frames, {}, ac.getSim().frame
  local function finishView(row, err, data)
    if active ~= owner or owner.async_error then
      if data then data:dispose() end
      return
    end
    local ok, problem = pcall(function()
      assert(not err and data, tostring(err or 'Async readback has no data'))
      row.async_callback_started_clock = os.preciseClock()
      local size = data:size()
      assert(size.x == 320 and size.y == 180, 'Async pixel dimensions changed')
      local ffi = require('ffi')
      local buffer = ffi.new('uint8_t[?]', 320 * 180 * 3)
      local color, offset = rgbm(), 0
      for y = 0, 179 do
        for x = 0, 319 do
          data:colorTo(color, x, y)
          assert(color.r >= 0 and color.r <= 1 and color.g >= 0 and color.g <= 1
            and color.b >= 0 and color.b <= 1, 'Async color is not normalized RGBA8888')
          buffer[offset] = math.floor(color.r * 255 + .5)
          buffer[offset + 1] = math.floor(color.g * 255 + .5)
          buffer[offset + 2] = math.floor(color.b * 255 + .5)
          offset = offset + 3
        end
      end
      row.conversion_finished_clock = os.preciseClock()
      assert(io.save(owner.directory .. '/' .. row.file,
        'P6\n320 180\n255\n' .. ffi.string(buffer, offset), true), 'Cannot save async RGB probe')
      row.async_file_finished_clock = os.preciseClock()
      data:dispose()
      data = nil
      owner.pending_count = owner.pending_count - 1
      if owner.pending_count == 0 then
        write(owner.directory .. string.format('/frame-%04d.json', index), {
          schema = 'flybrain.csp.camera.async_probe.frame.v1', request_id = owner.request.id,
          index = index, generation = generation, render_frame = renderFrame,
          script_clock = now, car_id = ac.getCarID(0), track_id = ac.getTrackID(),
          views = rows, actor_stream_created = false, training_admissible = false,
          physical_exposure_sync_verified = false, actuator_messages_sent = 0})
        owner.frames, owner.pending = owner.frames + 1, false
        owner.next = os.preciseClock() + owner.request.period_seconds
        if owner.frames >= owner.request.frames then
          writeCameraTerminal(owner, owner.directory .. '/result.json', {status = 'completed', frames = owner.frames,
            request_id = owner.request.id, async_readback_probe = true,
            actor_stream_created = false, training_admissible = false, actuator_messages_sent = 0})
          closeActive()
          terminalState, terminalReason = 'completed', nil
          status('completed')
        end
      end
    end)
    if not ok then
      if data then pcall(function() data:dispose() end) end
      owner.async_error = tostring(problem)
    end
  end
  for _, view in ipairs(owner.views) do
    local camera = layout[view.id]
    local row = {view_id = view.id, width = 320, height = 180,
      file = string.format('%04d-%s.ppm', index, view.id),
      reference_png = string.format('%04d-%s-reference.png', index, view.id),
      render_started_clock = os.preciseClock()}
    table.insert(rows, row)
    view.shot:update(camera[1], camera[2], car.up, camera[3])
    row.render_call_finished_clock = os.preciseClock()
    -- Same unchanged texture is saved and then downloaded; no later update until all callbacks.
    view.shot:save(owner.directory .. '/' .. row.reference_png, ac.ImageFormat.PNG)
    row.reference_finished_clock = os.preciseClock()
    row.async_requested_clock = os.preciseClock()
    view.shot:accessData(function(err, data) finishView(row, err, data) end)
  end
end

local function capture(now)
  if active.pending then return end
  local preambleTiming = timingStart()
  local car = ac.getCar(0)
  assert(car, 'Player car unavailable')
  local sim = ac.getSim()
  assert(active.request.allow_ideal_line_probe or sim.idealLineShown == false,
    'Ideal-line state changed or unavailable during normal capture')
  local dashConfig = timingCall('capture.INIConfig.carData', ac.INIConfig.carData, 0, 'dash_cam.ini')
  assert(dashConfig.sections.DASH_CAM and dashConfig.sections.DASH_CAM.POS,
    'Native dashboard camera position unavailable')
  local dashLocal = dashConfig:get('DASH_CAM', 'POS', vec3())
  local dashWorld = car.bodyTransform:transformPoint(dashLocal)
  -- Native BMW model +X is driver-left, confirmed by seat position and rendered landmarks.
  local layout = {
    front = {dashWorld, car.look, 56},
    left = {car.position + car.side * 1.15 + car.up * 1.0, car.side, 90},
    right = {car.position - car.side * 1.15 + car.up * 1.0, -car.side, 90},
    rear = {car.position - car.look * 2.4 + car.up * 1.0, -car.look, 90}
  }
  local motionDiagnostic = nil
  local normalExterior = active.request.camera_sensor_layout == 'exterior_front_v3'
  if normalExterior then
    layout.front[1] = car.position + car.look * 2.5 + car.up * 1.0
  end
  if active.request.camera_motion_probe then
    local probe = active.request.camera_motion_probe
    assert(sim.isPaused and not sim.isOnlineRace and car.isAIControlled == false
      and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98,
      'Camera-motion diagnostic lost stationary paused state')
    if probe.front_mount == 'bmw_z4_gt3_bumper_v1' then
      -- Engineering virtual-eye mount, forward/up from the native car reference.
      -- Retain the full scene and other cars; do not hide physical objects globally.
      layout.front[1] = car.position + car.look * 2.5 + car.up * 1.0
    end
    local elapsed = now - active.started_clock
    local moving = math.max(0, math.min(probe.motion_seconds, elapsed - probe.static_pre_seconds))
    local angle = moving * probe.yaw_left_rad_s
    local c, s, up = math.cos(angle), math.sin(angle), car.up
    for _, camera in pairs(layout) do
      local look = camera[2]
      local cross = vec3(up.y*look.z-up.z*look.y, up.z*look.x-up.x*look.z,
                        up.x*look.y-up.y*look.x)
      local dot = up.x*look.x + up.y*look.y + up.z*look.z
      camera[2] = look*c + cross*s + up*(dot*(1-c))
    end
    motionDiagnostic = {schema='flybrain.camera.rotation_probe.v1',
      requested_yaw_left_rad_s=probe.yaw_left_rad_s, yaw_left_angle_rad=angle,
      elapsed_csp_seconds=elapsed, static_pre_seconds=probe.static_pre_seconds,
      motion_seconds=probe.motion_seconds, scene=probe.scene, fixed_camera_centers=true,
      scene_root_nodes=active.diagnostic_scene_nodes,
      front_mount=probe.front_mount or 'native_dashboard_v2',
      physical_body_paused=true, actor_admissible=false}
  end
  if active.request.async_readback_probe then
    asyncProbe(car, layout, now)
    return
  end
  local index = active.frames
  local rows = {}
  timingFinish('capture.preamble_to_first_render', preambleTiming, true)
  for _, view in ipairs(active.views) do
    local camera = layout[view.id]
    local originBefore = optionalVector(function() return ac.getSim().originShift end)
    local physicsBefore = active.request.airflow_diagnostic and car.timestamp or nil
    local frameBefore = active.request.airflow_diagnostic and tostring(ac.getSim().frame) or nil
    local started = os.preciseClock()
    view.shot:update(camera[1], camera[2], car.up, camera[3])
    local rendered = os.preciseClock()
    local physicsAfter = active.request.airflow_diagnostic and car.timestamp or nil
    local frameAfter = active.request.airflow_diagnostic and tostring(ac.getSim().frame) or nil
    if view.id == 'front' and sourceTiming then
      timingFinish('capture.first_render', {owner=sourceTiming, started=started}, true)
    end
    local file = string.format('%04d-%s.%s', index, view.id, active.request.image_format)
    local saved = nil
    if not active.request.shared_transport and not active.request.deferred_save_probe then
      view.shot:save(active.directory .. '/' .. file,
        active.request.image_format == 'bmp' and ac.ImageFormat.BMP or ac.ImageFormat.PNG)
      saved = os.preciseClock()
    end
    local referenceFile = nil
    if active.request.verify_png_reference then
      referenceFile = string.format('%04d-%s-reference.png', index, view.id)
      if not active.request.shared_transport and not active.request.deferred_save_probe then
        view.shot:save(active.directory .. '/' .. referenceFile, ac.ImageFormat.PNG)
      end
    end
    table.insert(rows, {view_id = view.id, file = file, width = 320, height = 180,
      image_format = active.request.image_format, reference_png = referenceFile,
      requested_fov_degrees = camera[3], fov_axis_independently_verified = false,
      color_pipeline = 'CSP YEBIS HDR-to-LDR using current main post-processing settings',
      requested_exposure = active.request.exposure,
      render_started_clock = started, render_call_finished_clock = rendered,
      diagnostic_car_timestamp_before_ms = physicsBefore,
      diagnostic_car_timestamp_after_ms = physicsAfter,
      diagnostic_graphics_frame_before = frameBefore,
      diagnostic_graphics_frame_after = frameAfter,
      save_finished_clock = saved,
      reference_finished_clock = not active.request.shared_transport
        and not active.request.deferred_save_probe and os.preciseClock() or nil,
      shared_texture_handle = view.shot:sharedHandle(true),
      image_writer = active.request.shared_transport and 'host_lossless_bmp' or 'producer',
      view_matrix_diagnostic = tostring(view.shot:viewMatrix()),
      projection_matrix_diagnostic = tostring(view.shot:projectionMatrix()),
      renderer_origin_shift_before_diagnostic = originBefore,
      renderer_origin_contract_diagnostic = 'getSim.originShift_before_after_v1',
      renderer_origin_shift_after_diagnostic = optionalVector(function()
        return ac.getSim().originShift end),
      renderer_only_world_position = vector(camera[1]), renderer_only_world_look = vector(camera[2]),
      renderer_only_world_up = vector(car.up)})
  end
  if active.request.deferred_save_probe then
    for indexInGroup, view in ipairs(active.views) do
      local row = rows[indexInGroup]
      view.shot:save(active.directory .. '/' .. row.file, ac.ImageFormat.PNG)
      row.save_finished_clock = os.preciseClock()
    end
  end
  local metadata = {
    schema = 'flybrain.csp.camera.frame.v1', request_id = lastID, index = index,
    generation = generation,
    ideal_line_shown = optionalBoolean(function() return sim.idealLineShown end),
    aid_probe_only = active.request.allow_ideal_line_probe == true or motionDiagnostic ~= nil,
    deferred_save_probe = active.request.deferred_save_probe == true,
    camera_motion_diagnostic = motionDiagnostic,
    render_frame = sim.frame, script_clock = now, clock_domain = 'csp_os_preciseClock_unaligned',
    car_id = ac.getCarID(0), track_id = ac.getTrackID(), views = rows,
    camera_layout = (normalExterior or motionDiagnostic
      and motionDiagnostic.front_mount == 'bmw_z4_gt3_bumper_v1')
      and 'virtual front eye car.position+look*2.5+up*1.0; FOV56; exterior side/rear'
      or 'native dashboard POS with requested FOV56; model +X driver-left; exterior side/rear',
    native_dash_position_local = vector(dashLocal),
    camera_layout_revision = (normalExterior or motionDiagnostic
      and motionDiagnostic.front_mount == 'bmw_z4_gt3_bumper_v1') and 3 or 2,
    camera_mount_contract = normalExterior and {
      schema='flybrain.ac.camera_mount.v1', layout_id='exterior_front_v3', layout_revision=3,
      front_mount='bmw_z4_gt3_bumper_v1', front_offset_forward_m=2.5,
      front_offset_up_m=1.0, scene='full_scene', car_id='bmw_z4_gt3'} or nil,
    native_dash_camera_sections_diagnostic = dashConfig.sections,
    transport = active.request.shared_transport and 'shared_immutable_v1' or 'producer_file',
    exposure_synchronization_verified = false, actor_stream_created = false,
    training_admissible = false, actuator_messages_sent = 0}
  if active.request.airflow_diagnostic then
    local stationary = sim.isPaused and car.isAIControlled == false
      and math.abs(car.speedKmh) < .1 and car.gas < .02 and car.brake > .98
    local owned = expertProbe and expertProbe.motion
      and expertProbe.camera_request_id == active.request.id
    assert(not sim.isOnlineRace and (stationary or owned),
      'Airflow diagnostic lost paused state or its owned motion probe')
    metadata.airflow_diagnostic = airflowDiagnostic.capture(ac, os.preciseClock, generation)
    metadata.aid_probe_only = true
  end
  if active.camera_status_policy == CAMERA_STATUS_POLICY then
    metadata.camera_status_policy_echo = cameraStatusPolicyEcho(active)
  end
  if active.mmfFrame then
    metadata.metadata_transport = 'mmf_slot_sha256_v1'
    mmfPublish(active.mmfFrame, index, JSON.stringify(metadata))
  else
    write(active.directory .. string.format('/frame-%04d.json', index), metadata)
  end
  active.frames = active.frames + 1
  active.last_capture_clock = os.preciseClock()
  -- Preserve requested cadence across render ticks, including acknowledged shared groups.
  repeat active.next = active.next + active.request.period_seconds until active.next > now
  if active.request.shared_transport then
    active.pending, active.awaiting_ack = true, index
    return -- keep every source texture immutable until matching consumer acknowledgment
  end
  if active.request.shared_texture_probe then
    active.pending = true -- source textures remain unchanged until owned stop or deadline
    status('shared_probe_waiting')
    return
  end
  if active.frames >= active.request.frames then
    writeCameraTerminal(active, active.directory .. '/result.json', {status = 'completed', frames = active.frames,
      request_id = lastID, actuator_messages_sent = 0, actor_stream_created = false,
      exposure_synchronization_verified = false, training_admissible = false})
    closeActive()
    terminalState, terminalReason = 'completed', nil
    status('completed')
  end
end

function script.update(dt)
  if sourceTiming then sourceTiming.update_index = sourceTiming.update_index + 1 end
  local updateTiming = timingStart()
  local now = os.preciseClock()
  local settleTiming = timingStart()
  local settleOK, settleErr = pcall(pollBrakeSettle, now)
  timingFinish('pollBrakeSettle', settleTiming, settleOK)
  if not settleOK then
    sessionError = tostring(settleErr)
    pcall(ac.tryToPause, true)
    brakeSettle = nil
  end
  local probeTiming = timingStart()
  local probeOK, probeErr = pcall(pollExpertProbe, now)
  timingFinish('pollExpertProbe', probeTiming, probeOK)
  if not probeOK then
    sessionError = tostring(probeErr)
    pcall(stopExpertProbe, now, 'probe observer error')
  end
  local sessionTiming = timingStart()
  local sessionOK, sessionErr = pcall(pollSession, now)
  timingFinish('pollSession', sessionTiming, sessionOK)
  if not sessionOK then sessionError = tostring(sessionErr) end
  local ok, err = pcall(function()
    local pingTiming = timingStart()
    local pingText = timingCall('clock_ping.io.load', io.load, root .. '/clock-ping.json')
    if pingText and pingText ~= lastPing then
      local ping = JSON.parse(pingText)
      assert(type(ping.nonce) == 'string', 'Clock nonce required')
      write(root .. '/clock-reply.json', {nonce = ping.nonce, generation = generation,
        producer_clock = os.preciseClock(), window_handle = ac.getSim().windowHandle})
      lastPing = pingText
    end
    timingFinish('clock_ping', pingTiming, true)
    if active then
      if timingCall('stop.io.fileExists', io.fileExists, active.directory .. '/stop') then
        writeCameraTerminal(active, active.directory .. '/result.json', {status = 'stopped', frames = active.frames,
          request_id = lastID, actuator_messages_sent = 0, actor_stream_created = false})
        closeActive()
        terminalState, terminalReason = 'stopped', 'owned consumer stop'
        status('stopped', terminalReason)
        return
      end
      if active.async_error then error(active.async_error) end
      if now > active.deadline then error('Capture deadline exceeded') end
      if active.awaiting_ack ~= nil then
        local text
        if active.mmfAck then text = timingCall('ACK.mmfRead', mmfRead, active.mmfAck, active.awaiting_ack)
        else
          text = timingCall('ACK.io.load', io.load,
            active.directory .. string.format('/ack-%04d.json', active.awaiting_ack))
        end
        if text then
          local ack = timingCall('ACK.JSON.parse', JSON.parse, text)
          assert(ack.request_id == active.request.id and ack.generation == generation
            and ack.capture_id == active.awaiting_ack, 'Shared texture acknowledgment mismatch')
          if active.request.verify_png_reference then
            local referenceStarted = os.preciseClock()
            for _, view in ipairs(active.views) do
              view.shot:save(active.directory .. string.format('/%04d-%s-reference.png',
                active.awaiting_ack, view.id), ac.ImageFormat.PNG)
            end
            write(active.directory .. string.format('/reference-%04d.json', active.awaiting_ack), {
              request_id = active.request.id, generation = generation, capture_id = active.awaiting_ack,
              stage = 'after_host_copy', started_clock = referenceStarted,
              finished_clock = os.preciseClock()})
          end
          active.pending, active.awaiting_ack = false, nil
          if active.frames >= active.request.frames then
            writeCameraTerminal(active, active.directory .. '/result.json', {status = 'completed', frames = active.frames,
              request_id = lastID, shared_transport = true, actuator_messages_sent = 0,
              actor_stream_created = false, training_admissible = false})
            closeActive()
            terminalState, terminalReason = 'completed', nil
            status('completed')
            return
          end
        end
      end
      if now >= active.next then timingCall('capture', capture, now) end
    elseif now >= nextPoll and not heartbeatTransitionBusy() then
      nextPoll = now + .25
      local text = io.load(requestPath)
      if text then
        local request = JSON.parse(text)
        if request.id ~= lastID then begin(request, now) end
      end
    end
    if now >= nextStatus then
      nextStatus = now + 1
      if not (active and active.camera_status_policy == CAMERA_STATUS_POLICY) then
        status(active and 'capturing' or terminalState, terminalReason)
      end
    end
  end)
  if not ok then
    if active then
      writeCameraTerminal(active, active.directory .. '/result.json', {status = 'failed', reason = tostring(err),
        frames = active.frames, actuator_messages_sent = 0})
    end
    closeActive()
    terminalState, terminalReason = 'failed', tostring(err)
    status('failed', err)
  end
  timingFinish('script.update', updateTiming, ok and settleOK and probeOK and sessionOK)
  flushSourceTiming() -- frozen capture ledger waits for observed control quiescence
end

function script.windowMain(dt)
  ui.text('Read-only bounded engine camera probe')
  ui.text(message)
end

ac.onRelease(function()
  if brakeSettle then pcall(ac.tryToPause, true) end
  stopExpertProbe(os.preciseClock(), 'app release')
  closeActive()
  flushSourceTiming() -- best effort only; skip if release cleanup is still unobserved
  if heartbeatTransport then heartbeatTransport:close() end -- keep host lease/fence
end)

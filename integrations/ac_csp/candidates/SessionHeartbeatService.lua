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
    committed_source_clock=self.committed_source_clock}
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
      local original = self.committed_source_clock or self.started_clock
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
    local count, why = self.publisher:publish(state)
    assert(count or why == 'not_due', 'Heartbeat publisher retired: '..tostring(why))
    if count then
      self.published_clocks[count] = state.script_clock
      if self.retire_requested and self.quiescent() then
        self.final_sequence, self.final_source_clock, self.phase = count, state.script_clock, 'final'
        self:ack('final', {final_sequence=count, final_source_clock=state.script_clock})
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
    published_clocks={}, closed=false}, Service)
  local ok, problem = pcall(function()
    self.host_control = ac.readMemoryMappedFile(controlName(plan.owner_id, 'host'), M.CONTROL_LAYOUT, false)
    self.lua_control = ac.writeMemoryMappedFile(controlName(plan.owner_id, 'lua'), M.CONTROL_LAYOUT, false)
    assert(tonumber(self.lua_control.sequence) == 0 and tonumber(self.lua_control.magic) == 0,
      'Host-precreated Lua control map must be pristine; never reset it')
    self.publisher = proto.new({owner_id=plan.owner_id, generation=plan.generation,
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

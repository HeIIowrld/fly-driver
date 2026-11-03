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

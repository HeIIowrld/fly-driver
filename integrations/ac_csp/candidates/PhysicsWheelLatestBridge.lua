-- Experimental cross-Lua latest snapshot. No car physics writes or I/O.
-- Sequence checks detect some torn reads, not CPU memory-order or tick proof.
local M = {}
local function finite(x)
  return type(x) == 'number' and x == x and x ~= math.huge and x ~= -math.huge
end
function M.layout(ac, carIndex)
  assert(type(carIndex)=='number' and carIndex>=0 and carIndex==math.floor(carIndex), 'explicit car index required')
  local S=ac.StructItem
  return {S.key('flybrain.physics.wheel.latest.v1.car'..tostring(carIndex)),
    commit=S.uint32(), generation=S.uint32(), sample_sequence=S.uint32(),
    relative_elapsed=S.double(), wheel0=S.float(), wheel1=S.float(), wheel2=S.float(), wheel3=S.float()}
end
function M.publish(slot, generation, sample)
  assert(slot and type(generation)=='number' and generation>=1 and generation<=2147483647
    and generation==math.floor(generation), 'explicit generation required')
  assert(type(sample)=='table' and type(sample.sequence)=='number' and sample.sequence>=1
    and sample.sequence<=2147483647 and sample.sequence==math.floor(sample.sequence)
    and finite(sample.elapsed) and sample.elapsed>0 and type(sample.wheel_load_native_order)=='table',
    'finite bounded original physics sample required')
  local w=sample.wheel_load_native_order
  for i=1,4 do assert(finite(w[i]) and w[i]>=0, 'four finite nonnegative loads required') end
  local version=slot.commit
  assert(type(version)=='number' and version>=0 and version<4294967294
    and version==math.floor(version) and version%2==0, 'slot must be stable and not exhausted')
  slot.commit=version+1
  slot.generation=generation; slot.sample_sequence=sample.sequence; slot.relative_elapsed=sample.elapsed
  slot.wheel0=w[1]; slot.wheel1=w[2]; slot.wheel2=w[3]; slot.wheel3=w[4]
  slot.commit=version+2
end
function M.read(slot, expectedGeneration)
  assert(slot and type(expectedGeneration)=='number' and expectedGeneration>=1
    and expectedGeneration<=2147483647 and expectedGeneration==math.floor(expectedGeneration),
    'explicit expected generation required')
  local v1=slot.commit
  if type(v1)~='number' or v1<=0 or v1%2~=0 then return nil, 'uncommitted' end
  local record={generation=slot.generation,sequence=slot.sample_sequence,elapsed=slot.relative_elapsed,
    wheel_load_native_order={slot.wheel0,slot.wheel1,slot.wheel2,slot.wheel3},
    atomicity_verified=false,physical_time_verified=false,native_packet_id=nil,actor_admissible=false}
  local v2=slot.commit
  if v1~=v2 or v2%2~=0 then return nil, 'changed' end
  if record.generation~=expectedGeneration then return nil, 'generation_mismatch' end
  if not finite(record.elapsed) or record.elapsed<=0 or type(record.sequence)~='number'
    or record.sequence<1 or record.sequence~=math.floor(record.sequence) then return nil, 'invalid_sample' end
  for i=1,4 do
    local value=record.wheel_load_native_order[i]
    if not finite(value) or value<0 then return nil, 'invalid_sample' end
  end
  return record,nil
end
return M

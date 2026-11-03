"""Build a separate review candidate from the pinned timing Lua; never deploy."""

import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
BASE_SHA256 = 'b6d87d2708a13b37e858bf6cfe87f362540bf645514223e18e2c870fbd66e1f7'
OUTPUT = Path('integrations/ac_csp/candidates/FlyBrainCamera-heartbeat.lua')


def build():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('Production base changed; review a new binding before building')
    source = raw.decode('utf8').replace('\r\n', '\n')
    candidate_root = Path('integrations/ac_csp/candidates')
    proto = (candidate_root/'SessionHeartbeatMMF.lua').read_text(encoding='utf8')
    service = (candidate_root/'SessionHeartbeatService.lua').read_text(encoding='utf8')
    embedded = ('\n-- Candidate-only embedded protocol; exact deployed base bound by builder.\n'
        'local heartbeatProtocol = (function()\n'+proto+'\nend)()\n'
        'local heartbeatService = (function()\n'+service+'\nend)()\n'
        "local heartbeatLeasePath = sessionRoot .. '/status.json.flybrain-heartbeat-owner'\n"
        "local heartbeatArmPath = sessionRoot .. '/heartbeat-transport.json'\n"
        'local heartbeatTransport, heartbeatArmPoll, heartbeatArmOwner = nil, 0, nil\n'
        'local heartbeatPending = nil\n'
        'local heartbeatActiveLeaseText, heartbeatCompletedLease = nil, nil\n'
        '-- A reload cannot assume the old external OS writer was cancelled.\n'
        'local heartbeatOrphanGuard = io.fileExists(heartbeatLeasePath)\n')
    anchor = 'local expertProbe, brakeSettle = nil, nil\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+embedded)
    old_probe = '''local function probeCall(name, callback)
  local ok, err = pcall(callback)
  table.insert(expertProbe.calls, {name = name, returned = ok,
    script_clock = os.preciseClock(),
    error = not ok and tostring(err) or nil})'''
    new_probe = '''local function probeCall(name, callback)
  local started = os.preciseClock() -- callback boundary, not physical engine time
  local ok, err = pcall(callback)
  local finished = os.preciseClock()
  table.insert(expertProbe.calls, {name = name, returned = ok,
    started_clock = started, finished_clock = finished,
    script_clock = os.preciseClock(),
    error = not ok and tostring(err) or nil})'''
    assert source.count(old_probe) == 1
    source = source.replace(old_probe, new_probe)
    glue = r'''
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

'''
    anchor = 'local function pollSession(now)\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, glue+anchor)
    old = '''  flushSessionReceipt()
  if now >= sessionHeartbeat then
    sessionHeartbeat = now + .25
    local timing = timingStart()
    local ok = pcall(write, sessionRoot .. '/status.json', sessionState())
    timingFinish('session_heartbeat', timing, ok)
    if not ok then heartbeatWriteFailures = heartbeatWriteFailures + 1 end
  end
  if pendingSessionReceipt or expertProbe or brakeSettle then return end
'''
    new = '''  flushSessionReceipt()
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
'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '    elseif now >= nextPoll then\n'
    new = '    elseif now >= nextPoll and not heartbeatTransitionBusy() then\n'
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = ('  flushSourceTiming() -- best effort only; skip if release cleanup is still unobserved'
           '\nend)')
    new = ('  flushSourceTiming() -- best effort only; skip if release cleanup is still unobserved'
           '\n  if heartbeatTransport then heartbeatTransport:close() end -- keep host lease/fence'
           '\nend)')
    assert source.count(old) == 1
    result = source.replace(old, new)
    if b'\r\n' in raw:
        result = result.replace('\n', '\r\n')
    return result.encode('utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    content = build()
    OUTPUT.write_bytes(content)
    receipt = dict(base=str(BASE), base_sha256=BASE_SHA256, candidate=str(OUTPUT),
        candidate_sha256=hashlib.sha256(content).hexdigest(), deployed=False,
        changed_sections=['embedded protocol and lifecycle', 'periodic heartbeat sink',
                          'release closes owned protocol references',
                          'probeCall callback start/finish metadata'],
        control_caps_and_original_receipt_protocol_changed=False)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

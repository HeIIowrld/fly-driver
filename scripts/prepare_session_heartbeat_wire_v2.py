"""Prepare NEW wire-clock candidates from frozen c159; never edit/deploy current files."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('integrations/ac_csp/candidates')
# Preserved failed/deployed c159 candidate; primary main was exceptionally
# restored to b6 by root1790 and must never be used as an implicit new baseline.
BASE = ROOT/'FlyBrainCamera-heartbeat.lua'
PINNED = {
    BASE: 'c15902d21fec2b862bfadcaad75cfbdc076c77c0cc14ed43e3a7eb88b66cf4f6',
    ROOT/'SessionHeartbeatMMF.lua':
        '9056dda38a8edd7610d04dee1c9381391ec26c507dc3f95547dd05932356e415',
    ROOT/'SessionHeartbeatService.lua':
        '38e4e576d54d7fd5d161eebb90ca4b74823e503357c2f40b8996aa7ad199a413',
    Path('scripts/qualify_session_heartbeat_service.py'):
        'a952cece7180d98a16431b4fec3a60f45a052a7a38632543e0e0d0b9ee8501ac',
}


def checked(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINNED[path]:
        raise ValueError('Frozen c159 source changed: '+str(path))
    return raw.decode('utf8').replace('\r\n', '\n')


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Exactly one pinned source replacement required: '+old[:50])
    return text.replace(old, new)


def build():
    original_proto, original_service = (checked(ROOT/name) for name in
        ('SessionHeartbeatMMF.lua', 'SessionHeartbeatService.lua'))
    proto = once(original_proto,
        "  local ok, result = pcall(function()\n",
        "  local wireSource\n  local ok, result = pcall(function()\n")
    proto = once(proto, "    local crc, slot = M.crc32(packet), self.slots[count % 2]\n", '''\
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
''')
    proto = once(proto,
        '    self.published_count, self.last_source_clock = count, source\n',
        '    self.published_count, self.last_source_clock = count, source\n'
        '    self.last_wire_source_clock = wireSource\n'
        '    self.last_wire_delta_seconds = wireSource-source\n')
    proto = once(proto, '  return result\nend\n\nfunction M.new(config)',
        '  return result, nil, wireSource\nend\n\nfunction M.new(config)')
    proto = once(proto, "  local now = os.preciseClock()\n  assert(finite(now) and now >= 0,",
        "  assert(type(config.wire_clock_rounding_margin_ns) == 'number'\n"
        "    and config.wire_clock_rounding_margin_ns % 1 == 0\n"
        "    and config.wire_clock_rounding_margin_ns >= 1\n"
        "    and config.wire_clock_rounding_margin_ns <= 1000000,\n"
        "    'Existing cached clock rounding margin required; never widen it')\n"
        "  local now = os.preciseClock()\n  assert(finite(now) and now >= 0,")
    proto = once(proto, '    max_snapshots=config.max_snapshots, started_clock=now,',
        '    wire_clock_rounding_margin_ns=config.wire_clock_rounding_margin_ns,\n'
        '    max_snapshots=config.max_snapshots, started_clock=now,')
    service = once(original_service, '    committed_source_clock=self.committed_source_clock}',
        '    committed_source_clock=self.committed_source_clock,\n'
        "    last_raw_source_clock_text=self.last_raw_source_clock_text,\n"
        "    last_wire_source_clock_text=self.last_wire_source_clock_text,\n"
        '    last_wire_delta_text=self.last_wire_delta_text}')
    service = once(service,
        '        self.committed_sequence, self.committed_source_clock = count, source\n',
        '        self.committed_sequence, self.committed_source_clock = count, source\n'
        '        self.committed_raw_source_clock = self.published_raw_clocks[count]\n')
    service = once(service,
        '      local original = self.committed_source_clock or self.started_clock\n',
        '      local original = self.committed_raw_source_clock or self.started_clock\n')
    service = once(service, '    local count, why = self.publisher:publish(state)\n',
        '    local count, why, wireSource = self.publisher:publish(state)\n')
    service = once(service, '      self.published_clocks[count] = state.script_clock\n',
        '      self.published_clocks[count] = wireSource\n'
        '      self.published_raw_clocks[count] = state.script_clock\n'
        "      self.last_raw_source_clock_text = string.format('%.17g', state.script_clock)\n"
        "      self.last_wire_source_clock_text = string.format('%.17g', wireSource)\n"
        "      self.last_wire_delta_text = string.format('%.17g', wireSource-state.script_clock)\n")
    service = once(service,
        '        self.final_sequence, self.final_source_clock, self.phase = '
        "count, state.script_clock, 'final'\n"
        "        self:ack('final', {final_sequence=count, "
        'final_source_clock=state.script_clock})\n',
        '        self.final_sequence, self.final_source_clock, self.phase = '
        "count, wireSource, 'final'\n"
        "        self:ack('final', {final_sequence=count, final_source_clock=wireSource})\n")
    service = once(service, '    published_clocks={}, closed=false}, Service)',
        '    published_clocks={}, published_raw_clocks={}, closed=false}, Service)')
    service = once(service, '      duration_seconds=plan.deadline_source_clock-now,',
        '      wire_clock_rounding_margin_ns=plan.clock_rounding_margin_ns,\n'
        '      duration_seconds=plan.deadline_source_clock-now,')
    candidate = once(checked(BASE), original_proto, proto)
    candidate = once(candidate, original_service, service)
    runner_path = Path('scripts/qualify_session_heartbeat_service.py')
    runner = once(checked(runner_path),
        "CANDIDATE = Path('integrations/ac_csp/candidates/FlyBrainCamera-heartbeat.lua')",
        "CANDIDATE = Path('integrations/ac_csp/candidates/FlyBrainCamera-heartbeat-wire-v2.lua')")
    runner = once(runner, '        write_json(session_root/\'heartbeat-transport.json\', plan)\n',
        '        plan[\'clock_rounding_margin_ns\'] = context.calibration.rounding_margin_ns\n'
        '        write_json(session_root/\'heartbeat-transport.json\', plan)\n')
    runner = once(runner, '''        def sync_return():
            ack = context.control.receive()
            return ack and ack[1]['phase'] == 'sync_return'
''', '''        def sync_return():
            from flybrain_drive.session_heartbeat_terminal_v2 import observe_sync_return

            return observe_sync_return(context)
''')
    runner = once(runner,
        "                owned_lease_release=getattr(context, 'release_receipt', None),",
        "                terminal_control_ack=getattr(context, 'terminal_control_ack', None),\n"
        "                owned_lease_release=getattr(context, 'release_receipt', None),")
    return {ROOT/'SessionHeartbeatMMF-wire-v2.lua': proto.encode(),
        ROOT/'SessionHeartbeatService-wire-v2.lua': service.encode(),
        ROOT/'FlyBrainCamera-heartbeat-wire-v2.lua': candidate.replace('\n', '\r\n').encode(),
        Path('scripts/qualify_session_heartbeat_service_v2.py'): runner.encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    outputs = build()
    for path, content in outputs.items():
        path.write_bytes(content)
    receipt = dict(base_sha256=PINNED[BASE], frozen_sources_unchanged=all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in PINNED.items()),
        outputs={str(p): hashlib.sha256(v).hexdigest() for p, v in outputs.items()},
        deployed=False, current_main_changed=False, live_actions=0)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()

"""Embed a tested thermal module into an exact, newer installed Lua base.

Does not deploy or start AC. Preserve every existing source/camera witness.
"""
import argparse
import shutil
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

MODULE=Path('integrations/ac_csp/initial_tyre_temperature.lua')


def build(base, module):
    if 'local initialTyres =' in base:raise ValueError('Thermal module already embedded')
    anchors=['local function sessionState()',
             "    capabilities = {source_body_recording_1600_v1 = true,",
             '  local before = sessionState()\n  if request.command']
    for anchor in anchors:
        if base.count(anchor)!=1:raise ValueError('Exact unique deployed-base anchor required: '+anchor)
    base=base.replace(anchors[0], 'local initialTyres = (function()\n'+module+'\nend)()\n\n'+anchors[0])
    base=base.replace(anchors[1], "    tyres_initialization_epoch = initialTyres.lastEpoch(),\n    capabilities = {initial_tyre_temperature_v1 = initialTyres.available(physics), source_body_recording_1600_v1 = true,")
    branch="""  local before = sessionState()
  if request.command == 'set-tyres-temperature-initial' then
    if not initialTyres.validate(request, before, now, physics) then
      write(resultPath, {status='rejected', id=request.id, command=request.command,
        reason='Thermal initialization guards rejected', before=before})
      return
    end
    -- Acceptance must be durable before any SDK setter. Metadata retry never
    -- re-enters apply; one-shot module epoch also blocks new-ID retries.
    write(resultPath, {status='accepted', id=request.id, command=request.command, before=before})
    local receipt = initialTyres.apply(request, sessionState(), os.preciseClock(), physics)
    receipt.id, receipt.command, receipt.before = request.id, request.command, before
    receipt.after = sessionState()
    receipt.ui_actions_sent, receipt.actuator_messages_sent = 0, 0
    pendingSessionReceipt = {path=terminalPath, value=receipt}
    flushSessionReceipt()
    return
  end
  if request.command"""
    return base.replace(anchors[2],branch)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--base-sha256',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.base)
    if sha256(base)!=a.base_sha256:raise ValueError('Pinned deployed base differs')
    candidate=build(base.read_text(encoding='utf-8'),MODULE.read_text(encoding='utf-8'))
    run=begin_run(a.output,dict(arguments=vars(a),base_sha256=sha256(base),module_sha256=sha256(MODULE),
        program_sha256=sha256(__file__),ui_actions_sent=0,actuator_messages_sent=0,deployed=False))
    shutil.copyfile(base,run/'deployed-base.lua');shutil.copyfile(MODULE,run/'thermal-module.lua')
    shutil.copyfile(__file__,run/'executed-builder.py')
    (run/'FlyBrainCamera.lua').write_text(candidate,encoding='utf-8')
    write_json(run/'result.json',dict(status='completed_thermal_source_build',base_sha256=sha256(base),
        candidate_sha256=sha256(run/'FlyBrainCamera.lua'),module_sha256=sha256(MODULE),deployed=False,
        compile_verified=False,temperature_applied_verified=False,
        termination_reason='Exact installed base preserved and embedded candidate generated; tests/deployment/live readback separate'))


if __name__=='__main__':main()

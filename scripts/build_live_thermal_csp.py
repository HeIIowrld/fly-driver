"""Build an explicit stationary-live thermal variant from the exact installed source."""
import argparse,shutil
from pathlib import Path
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256


def build(base,old_module,new_module):
    old='local initialTyres = (function()\n'+old_module+'\nend)()\n\n'
    if base.count(old)!=1:raise ValueError('Exact original embedded thermal module required')
    source=base.replace(old,'local initialTyres = (function()\n'+new_module+'\nend)()\n\n',1)
    cap='initial_tyre_temperature_v1 = initialTyres.available(physics),'
    if source.count(cap)!=1:raise ValueError('Unique thermal capability required')
    if 'initial_tyre_temperature_live_v1 =' not in source:source=source.replace(cap,cap+' initial_tyre_temperature_live_v1 = initialTyres.available(physics),',1)
    if 'native_thermal_wake_v1 =' not in source:source=source.replace(cap,cap+' native_thermal_wake_v1 = initialTyres.wakeAvailable(physics),',1)
    source=source.replace('    tyres_initialization_epoch = initialTyres.lastEpoch(),','    tyres_native_wake_epoch = initialTyres.lastWakeEpoch(),\n    tyres_initialization_epoch = initialTyres.lastEpoch(),',1)
    declaration='local function sessionState()'
    snapshot="""local function thermalSDKSnapshot()
  local ok, value = pcall(function()
    local car, temperatures = ac.getCar(0), {}
    for i=0,3 do temperatures[i+1] = car.wheels[i].tyreCoreTemperature end
    return {source_clock=os.preciseClock(), core_temperature_c=temperatures, actor_input=false}
  end)
  return ok and value or {error=tostring(value), actor_input=false}
end

"""
    assert source.count(declaration)==1
    if 'local function thermalSDKSnapshot()' not in source:source=source.replace(declaration,snapshot+declaration,1)
    call='    local receipt = initialTyres.apply(request, sessionState(), os.preciseClock(), physics)'
    assert source.count(call)==1
    if 'receipt.sdk_before, receipt.sdk_after' not in source:source=source.replace(call,'    local sdkBefore = thermalSDKSnapshot()\n'+call+'\n    receipt.sdk_before, receipt.sdk_after = sdkBefore, thermalSDKSnapshot()',1)
    anchor="  if request.command == 'set-tyres-temperature-initial' then"
    wake="""  if request.command == 'wake-thermal-native-initial' then
    if not initialTyres.validateWake(request,before,now,physics) then
      write(resultPath,{status='rejected',id=request.id,command=request.command,reason='Native thermal wake guards rejected',before=before})
      return
    end
    write(resultPath,{status='accepted',id=request.id,command=request.command,before=before})
    local receipt=initialTyres.prepareNative(request,sessionState(),os.preciseClock(),physics)
    receipt.id,receipt.command,receipt.before=request.id,request.command,before
    receipt.after=sessionState()
    receipt.ui_actions_sent,receipt.actuator_messages_sent=0,0
    pendingSessionReceipt={path=terminalPath,value=receipt}
    flushSessionReceipt()
    return
  end
"""
    assert source.count(anchor)==1
    return source.replace(anchor,wake+anchor,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--base-sha256',required=True);p.add_argument('--old-module',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    base=Path(a.base);assert sha256(base)==a.base_sha256
    module=Path('integrations/ac_csp/initial_tyre_temperature.lua')
    candidate=build(base.read_text(encoding='utf-8'),Path(a.old_module).read_text(encoding='utf-8'),module.read_text(encoding='utf-8'))
    from lupa.luajit21 import LuaRuntime
    lua=LuaRuntime(unpack_returned_tuples=True);ok,error=lua.eval('function(s) local f,e=loadstring(s); return f~=nil,e end')(candidate)
    if not ok:raise ValueError('Full candidate LuaJIT compilation failed: '+str(error))
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),base_sha256=sha256(base),module_sha256=sha256(module),seed=None,deployed=False))
    shutil.copy2(base,run/'installed-base.lua');shutil.copy2(module,run/'thermal-module.lua');shutil.copy2(__file__,run/'executed-builder.py')
    (run/'FlyBrainCamera.lua').write_text(candidate,encoding='utf-8')
    write_json(run/'result.json',dict(status='completed_compiled_live_thermal_candidate',candidate_sha256=sha256(run/'FlyBrainCamera.lua'),compile_verified=True,deployed=False,actor_admitted=False))

if __name__=='__main__':main()

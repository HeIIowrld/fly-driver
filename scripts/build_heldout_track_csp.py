"""Build and compile a source-pinned, two-track CSP restart candidate."""
import argparse
from pathlib import Path

from lupa.luajit21 import LuaRuntime

from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256

EXPECTED = "0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e"
INSTALLED = Path("C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/apps/lua/FlyBrainCamera/FlyBrainCamera.lua")
SWITCH_GUARD = """
local function heldoutTrackDestination(before, request)
  if request.command ~= 'restart-heldout-track' or request.race_ini ~= nil
      or request.expected_track_id ~= before.track_id
      or request.native_ai_off_verified ~= true
      or request.released_controller_verified ~= true then return nil end
  if request.same_track_engine_relaunch ~= nil then
    if request.same_track_engine_relaunch == true and before.track_id == 'magione'
        and request.target_track_id == 'magione' and before.paused == true
        and before.in_menu == false then return 'magione' end
    return nil
  end
  if before.track_id == 'monza' and request.target_track_id == 'magione'
      and before.paused == true and before.in_menu == false then return 'magione' end
  if before.track_id == 'magione' and request.target_track_id == 'monza'
      and (before.paused == true or before.in_menu == true) then return 'monza' end
  return nil
end
local function stationaryEngineShutdownAllowed(before, request)
  return request.command == 'shutdown-engine'
    and (before.track_id == 'monza' or before.track_id == 'magione')
    and request.expected_track_id == before.track_id
    and request.race_ini == nil and request.target_track_id == nil
    and request.native_ai_off_verified == true and request.released_controller_verified == true
    and before.ai_controlled == false and before.camera_active == false
    and (before.paused == true or before.in_menu == true)
end
local function heldoutLifecycleAllowed(before, request)
  if before.track_id ~= 'magione' or request.expected_track_id ~= 'magione'
      or request.race_ini ~= nil or request.target_track_id ~= nil then return false end
  return request.command == 'start' and before.in_menu == true
    or request.command == 'restart' and before.paused == true and before.in_menu == false
    or request.command == 'pause' and before.live == true and before.in_menu == false
      and before.paused == false
    or request.command == 'resume' and before.paused == true and before.in_menu == false
end
local function heldoutGeometryAllowed(before, request)
  return request.command == 'sample-heldout-geometry' and request.expected_track_id == 'magione'
    and request.target_track_id == nil and request.race_ini == nil
    and request.native_ai_off_verified == true and request.released_controller_verified == true
    and before.track_id == 'magione' and (before.paused == true or before.in_menu == true)
end

local function heldoutGeometry()
  assert(ac.hasTrackSpline(), 'Installed track spline required')
  local rows = {}
  for index = 0, 511 do
    local progress = index / 512
    local point = ac.trackProgressToWorldCoordinate(progress)
    local sides = ac.getTrackAISplineSides(progress)
    assert(point and sides and math.abs(point.x) < 1e6 and math.abs(point.y) < 1e6 and math.abs(point.z) < 1e6
      and sides.x > 0 and sides.x < 100 and sides.y > 0 and sides.y < 100,
      'Finite installed track spline point and positive sides required')
    rows[index+1] = {progress=progress, native_xyz={point.x,point.y,point.z},
      left_m=sides.x, right_m=sides.y}
  end
  return rows
end
"""


def replace_one(text, original, replacement):
    if text.count(original) != 1:
        raise ValueError("Installed source insertion anchor is missing or ambiguous")
    return text.replace(original,replacement,1)


def build(text):
    anchor="local function sessionState()"
    text=replace_one(text,anchor,SWITCH_GUARD+"\n"+anchor)
    cap="      restart_engine = type(ac.restartAssettoCorsa) == 'function',"
    text=replace_one(text,cap,cap+"\n      restart_heldout_track_v1 = type(ac.restartAssettoCorsa) == 'function',"
        +"\n      heldout_track_lifecycle_v1 = true,"
        +"\n      stationary_engine_shutdown_v1 = type(ac.shutdownAssettoCorsa) == 'function',"
        +"\n      heldout_magione_engine_relaunch_v1 = type(ac.restartAssettoCorsa) == 'function', "
        +"\n      heldout_track_geometry_v1 = ac.hasTrackSpline ~= nil and ac.getTrackAISplineSides ~= nil and ac.trackProgressToWorldCoordinate ~= nil,")
    declaration="  local isEngineRestart = request.command == 'restart-engine'"
    text=replace_one(text,declaration,declaration+"\n  local heldoutDestination = heldoutTrackDestination(before, request)"
        +"\n  local isHeldoutGeometry = heldoutGeometryAllowed(before, request)"
        +"\n  local isHeldoutLifecycle = heldoutLifecycleAllowed(before, request)"
        +"\n  local isEngineShutdown = stationaryEngineShutdownAllowed(before, request)")
    track="""    and before.car_id == 'bmw_z4_gt3' and before.track_id == 'monza'
    and (isHandoff or before.speed_kmh"""
    text=replace_one(text,track,"""    and before.car_id == 'bmw_z4_gt3' and (before.track_id == 'monza' or heldoutDestination ~= nil
      or isHeldoutGeometry or isHeldoutLifecycle or isEngineShutdown)
    and (isHandoff or before.speed_kmh""")
    text=replace_one(text,"      or isEngineRestart and before.paused",
                     "      or isHeldoutGeometry\n"
                     "      or isEngineShutdown and type(ac.shutdownAssettoCorsa) == 'function'\n"
                     "        and finite(request.old_engine_pid) and request.old_engine_pid > 0\n"
                     "        and request.old_engine_pid % 1 == 0\n"
                     "      or (isEngineRestart and before.paused or heldoutDestination ~= nil)")
    text=replace_one(text,"  if isEngineRestart then", """  if isHeldoutGeometry then
    local ok, rows = pcall(heldoutGeometry)
    write(terminalPath, {status=ok and 'heldout_geometry_sampled' or 'heldout_geometry_rejected',
      id=request.id, command=request.command, before=before,
      geometry_schema='flybrain.csp.heldout_geometry.v1', actor_input=false,
      rows=ok and rows or nil, reason=not ok and tostring(rows) or nil})
    return
  end
  if isEngineRestart or heldoutDestination ~= nil or isEngineShutdown then""")
    text=replace_one(text,"status='engine_restart_intent_committed'",
        "status=isEngineShutdown and 'engine_shutdown_intent_committed' or 'engine_restart_intent_committed'")
    text=replace_one(text,"      command=request.command, old_engine_pid=request.old_engine_pid, before=before,",
        "      command=request.command, old_engine_pid=request.old_engine_pid, before=before,\n      target_track_id=heldoutDestination,")
    text=replace_one(text,"    local ok, err = pcall(function() ac.restartAssettoCorsa() end)",
        """    local ok, err = pcall(function()
      if isEngineShutdown then
        ac.shutdownAssettoCorsa()
      elseif heldoutDestination ~= nil then
        ac.restartAssettoCorsa('[RACE]\\nTRACK=' .. heldoutDestination .. '\\nCONFIG_TRACK=\\n')
      else
        ac.restartAssettoCorsa()
      end
    end)""")
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--base",type=Path,default=INSTALLED)
    args=parser.parse_args()
    if sha256(args.base)!=EXPECTED:
        raise ValueError("Exact original installed CSP source required")
    candidate=build(args.base.read_text(encoding="utf-8"))
    lua=LuaRuntime(unpack_returned_tuples=True)
    ok,reason=lua.eval("function(s) local f,e=loadstring(s); return f~=nil,e end")(candidate)
    if not ok:
        raise ValueError(f"Full LuaJIT compilation failed: {reason}")
    lua.execute(SWITCH_GUARD.replace("local function","function"))
    guard=lua.globals().heldoutTrackDestination
    def destination(origin,target,*,paused=True,in_menu=False,**changes):
        request=dict(command="restart-heldout-track",expected_track_id=origin,target_track_id=target,
                     native_ai_off_verified=True,released_controller_verified=True)
        request.update(changes)
        return guard(lua.table(track_id=origin,paused=paused,in_menu=in_menu),lua.table_from(request))
    assert destination("monza","magione")=="magione"
    assert destination("magione","magione",same_track_engine_relaunch=True)=="magione"
    assert destination("magione","magione") is None
    assert destination("magione","magione",paused=False,in_menu=True,same_track_engine_relaunch=True) is None
    assert destination("monza","magione",same_track_engine_relaunch=True) is None
    assert destination("magione","monza")=="monza"
    assert destination("magione","monza",paused=False,in_menu=True)=="monza"
    assert destination("magione","monza",paused=False,in_menu=False) is None
    assert destination("monza","magione",paused=False,in_menu=True) is None
    assert destination("monza","spa") is None
    assert destination("monza","magione",race_ini="unreviewed") is None
    assert destination("monza","magione",expected_track_id="magione") is None
    assert destination("monza","magione",released_controller_verified=False) is None
    assert destination("monza","magione",native_ai_off_verified=False) is None
    lifecycle=lua.globals().heldoutLifecycleAllowed
    menu=lua.table(track_id="magione",in_menu=True,paused=False,live=False)
    paused=lua.table(track_id="magione",in_menu=False,paused=True,live=False)
    live=lua.table(track_id="magione",in_menu=False,paused=False,live=True)
    def lifecycle_allowed(state, command, **changes):
        request=dict(command=command,expected_track_id="magione")
        request.update(changes)
        return lifecycle(state,lua.table_from(request))
    assert lifecycle_allowed(menu,"start")
    assert lifecycle_allowed(paused,"resume")
    assert lifecycle_allowed(live,"pause")
    assert not lifecycle_allowed(live,"start")
    assert lifecycle_allowed(paused,"restart")
    assert not lifecycle_allowed(live,"restart")
    assert not lifecycle_allowed(paused,"set-tyres-temperature-initial")
    assert not lifecycle_allowed(paused,"resume",expected_track_id="monza")
    assert not lifecycle_allowed(paused,"resume",race_ini="unreviewed")
    assert not lifecycle_allowed(paused,"resume",target_track_id="monza")
    shutdown_guard=lua.globals().stationaryEngineShutdownAllowed
    shutdown=dict(command="shutdown-engine", expected_track_id="magione",
                  native_ai_off_verified=True, released_controller_verified=True)
    stationary=dict(track_id="magione", paused=False, in_menu=True,
                    ai_controlled=False, camera_active=False)
    assert shutdown_guard(lua.table_from(stationary),lua.table_from(shutdown))
    assert shutdown_guard(lua.table_from(dict(stationary, track_id="monza", paused=True, in_menu=False)),
                          lua.table_from(dict(shutdown, expected_track_id="monza")))
    assert not shutdown_guard(lua.table_from(dict(stationary, in_menu=False)),lua.table_from(shutdown))
    assert not shutdown_guard(lua.table_from(dict(stationary, camera_active=True)),lua.table_from(shutdown))
    assert not shutdown_guard(lua.table_from(dict(stationary, ai_controlled=True)),lua.table_from(shutdown))
    assert not shutdown_guard(lua.table_from(stationary),lua.table_from(dict(shutdown, target_track_id="monza")))
    assert not shutdown_guard(lua.table_from(stationary),lua.table_from(dict(shutdown, released_controller_verified=False)))
    assert not shutdown_guard(lua.table_from(stationary),lua.table_from(dict(shutdown, expected_track_id="monza")))
    geometry_guard=lua.globals().heldoutGeometryAllowed
    valid=dict(command="sample-heldout-geometry",expected_track_id="magione",
        native_ai_off_verified=True,released_controller_verified=True)
    assert geometry_guard(lua.table(track_id="magione",paused=False,in_menu=True),lua.table_from(valid))
    assert not geometry_guard(lua.table(track_id="monza",paused=True),lua.table_from(valid))
    assert not geometry_guard(lua.table(track_id="magione",paused=False,in_menu=False),lua.table_from(valid))
    assert not geometry_guard(lua.table(track_id="magione",paused=True),lua.table_from(dict(valid,race_ini="other")))
    lua.execute("""ac={hasTrackSpline=function() return true end,
      trackProgressToWorldCoordinate=function(v) return {x=v,y=1,z=-v} end,
      getTrackAISplineSides=function() return {x=3,y=4} end}""")
    sample=lua.globals().heldoutGeometry()
    assert len(sample)==512 and sample[512]["progress"]==511/512 and sample[1]["left_m"]==3
    out=begin_run(args.output,dict(program_sha256=sha256(__file__),base_sha256=EXPECTED,seed=None,
        actor_input=False,deployed=False))
    original=out/"installed-base.lua"
    original.write_bytes(args.base.read_bytes())
    if sha256(original)!=EXPECTED:
        raise ValueError("Original installed CSP changed during collection")
    path=out/"FlyBrainCamera.lua"
    path.write_bytes(candidate.encode("utf-8"))
    write_json(out/"result.json",dict(status="completed_compiled_heldout_csp_candidate",
        candidate_sha256=sha256(path),guard_cases=36,geometry_sample_cases=1,deployed=False,live_restart_verified=False,
        vehicle_control_sent=False))

if __name__=="__main__":
    main()

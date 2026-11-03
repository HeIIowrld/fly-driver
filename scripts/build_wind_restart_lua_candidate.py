"""Build an isolated bounded wind-merge lifecycle candidate; never deploy it."""

import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def replace_exact(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Expected exact unique original lifecycle source required')
    return source.replace(old, new, 1)


def build_candidate(source, diagnostic_source=None):
    source = replace_exact(source, "  local isEngineRestart = request.command == 'restart-engine'",
        """  local isWindRestart = request.command == 'restart-engine-wind-diagnostic'
  local isEngineRestart = request.command == 'restart-engine' or isWindRestart
  local windValid = not isWindRestart or (
    (request.wind_direction_deg == 0 or request.wind_direction_deg == 90
      or request.wind_direction_deg == 180 or request.wind_direction_deg == 270)
    and (request.wind_speed_kmh == 36 or request.wind_speed_kmh == 0
      and request.wind_direction_deg == 0)
    and request.native_ai_off_verified == true
    and type(request.original_race_sha256) == 'string'
    and #request.original_race_sha256 == 64
    and request.original_race_sha256:match('^[0-9a-f]+$') ~= nil)""")
    source = replace_exact(source,
        "    and type(request.expires_clock) == 'number' and request.expires_clock >= now\n",
        "    and windValid\n"
        "    and type(request.expires_clock) == 'number' and request.expires_clock >= now\n")
    source = replace_exact(source,
        "      command=request.command, old_engine_pid=request.old_engine_pid, before=before,\n",
        """      command=request.command, old_engine_pid=request.old_engine_pid, before=before,
      wind_direction_deg=isWindRestart and request.wind_direction_deg or nil,
      wind_speed_kmh=isWindRestart and request.wind_speed_kmh or nil,
      original_race_sha256=isWindRestart and request.original_race_sha256 or nil,
""")
    source = replace_exact(source,
        "    local ok, err = pcall(function() ac.restartAssettoCorsa() end)",
        """    local merge = nil
    if isWindRestart then
      merge = string.format('[WIND]\\nDIRECTION_DEG=%d\\nSPEED_KMH_MIN=%d\\nSPEED_KMH_MAX=%d\\n',
        request.wind_direction_deg, request.wind_speed_kmh, request.wind_speed_kmh)
    end
    local ok, err = pcall(function() ac.restartAssettoCorsa(merge) end)""")
    if diagnostic_source is not None:
        source = replace_exact(source,
            "local sessionRoot = ac.getFolder(ac.FolderID.AppDataLocal) .. '/flybrain/csp-session'",
            "local windDiagnostic = (function()\n" + diagnostic_source + "\nend)()\n"
            "local sessionRoot = ac.getFolder(ac.FolderID.AppDataLocal) .. '/flybrain/csp-session'")
        source = replace_exact(source,
            "  local isWindRestart = request.command == 'restart-engine-wind-diagnostic'",
            "  local isWindObserve = request.command == 'wind-observe-diagnostic'\n"
            "  local isWindRestart = request.command == 'restart-engine-wind-diagnostic'")
        source = replace_exact(source,
            "      or isDisableLine and (before.paused or request.allow_live_stationary == true",
            "      or isWindObserve and before.paused and not before.camera_active\n"
            "        and request.race_ini == nil\n"
            "      or isDisableLine and (before.paused or request.allow_live_stationary == true")
        source = replace_exact(source, "  if isEngineRestart then\n    -- Durable intent",
            """  if isWindObserve then
    local ok, observed = pcall(windDiagnostic.capture, ac, os.preciseClock, generation)
    write(terminalPath, {status=ok and 'wind_observed_diagnostic' or 'wind_observation_failed',
      id=request.id, command=request.command, before=before,
      observation=ok and observed or nil, error=not ok and tostring(observed) or nil,
      actor_admissible=false, ui_actions_sent=0, actuator_messages_sent=0})
    return
  end
  if isEngineRestart then
    -- Durable intent""")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original = Path('integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua')
    diagnostic = Path('integrations/ac_csp/candidates/RelativeAirflowDiagnostic.lua')
    source = build_candidate(original.read_text(encoding='utf8'), diagnostic.read_text(
        encoding='utf8'))
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'base_sha256': sha256(original), 'diagnostic_sha256': sha256(diagnostic),
        'scope': __doc__, 'deployment': False})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    candidate = run/'FlyBrainCamera-wind-restart-diagnostic.lua'
    candidate.write_text(source, encoding='utf8', newline='\n')
    result = {'status': 'completed_wind_restart_source_candidate',
              'base_sha256': sha256(original), 'candidate_sha256': sha256(candidate),
              'candidate': str(candidate), 'installed': False, 'simulator_restart': False,
              'native_flag_host_attestation_only': True, 'race_restore_implemented': False,
              'physical_wind_qualified': False, 'actor_deployed': False}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

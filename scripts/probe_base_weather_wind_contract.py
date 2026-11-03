"""Run installed base WeatherFX controller against synthetic input; no simulator changes."""

import argparse
import json
from pathlib import Path

from flybrain_drive.evidence import begin_run, write_json
from flybrain_drive.graph import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from lupa.luajit21 import LuaRuntime

    source = Path(args.controller)
    source_bytes = source.read_bytes()
    source_hash = sha256(source)
    cases = [(0, 0), (0, 36), (90, 36), (180, 36), (270, 36)]
    observations = []
    for direction, speed in cases:
        lua = LuaRuntime(unpack_returned_tuples=True)
        lua.execute('''
          package.preload['shared/sim/weather']=function()
            return {setRainIntensity=function(c, initial) c.rainIntensity=0 end,
                    debugAware=function() return false end}
          end
          refbool=function(value) return {value=value} end
          script={}; applied={}
          ac={getInputWeatherType=function() return 3 end,
              ConditionsSet=function() return {} end,
              getInputTemperatures=function() return {ambient=21,road=27} end,
              getInputWind=function() return {direction=windDirection,
                       speedFrom=windSpeed,speedTo=windSpeed} end,
              getInputHumidity=function() return .4 end,
              getInputPressure=function() return 101325 end,
              getInputTrackState=function() return {sessionStart=95} end,
              getSim=function() return {timestamp=1000} end,
              setConditionsSet2=function(c)
                table.insert(applied, {direction=c.wind.direction,
                  speedFrom=c.wind.speedFrom,speedTo=c.wind.speedTo,
                  ambient=c.temperatures.ambient,road=c.temperatures.road,
                  currentType=c.currentType,humidity=c.humidity,pressure=c.pressure})
              end}
        ''')
        lua.globals().windDirection = direction
        lua.globals().windSpeed = speed
        lua.execute(source_bytes.decode('utf8'))
        lua.execute('script.update(0.1)')
        actual = lua.globals().applied
        if len(actual) != 1:
            raise ValueError('Static synthetic input should be applied exactly once')
        row = {key: actual[1][key] for key in ('direction', 'speedFrom', 'speedTo',
            'ambient', 'road', 'currentType', 'humidity', 'pressure')}
        if row != {'direction': direction, 'speedFrom': speed, 'speedTo': speed,
                   'ambient': 21, 'road': 27, 'currentType': 3, 'humidity': .4,
                   'pressure': 101325}:
            raise ValueError('Installed controller output differs from synthetic input')
        observations.append(row)
    if sha256(source) != source_hash:
        raise ValueError('Installed controller changed during read-only probe')
    run = begin_run(args.output, {'program_sha256': sha256(__file__),
        'controller_sha256': source_hash, 'cases': cases, 'scope': __doc__})
    (run/'executed-program.py').write_bytes(Path(__file__).read_bytes())
    result = {'status': 'completed_synthetic_installed_weather_controller_probe',
        'cases': observations, 'installed_controller_modified': False,
        'live_controller_identity_verified': False, 'game_applied_weather_verified': False,
        'physics_wind_source_verified': False, 'simulator_restarted': False,
        'actor_deployed': False, 'optimizer_updates': 0}
    write_json(run/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

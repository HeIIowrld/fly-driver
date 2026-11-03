"""Validate diagnostic CSP clock pairs without inferring physical exposure time."""
import math


def summarize_clock_pairs(frame):
    views=frame.get('views') if isinstance(frame,dict) else None
    if not isinstance(views,list) or len(views)!=4 or {v.get('view_id') for v in views}!={'front','left','right','rear'}:
        raise ValueError('exactly four named views required')
    if frame.get('training_admissible') is not False or frame.get('actor_stream_created') is not False:
        raise ValueError('diagnostic data must be fit- and actor-forbidden')
    summaries=[]
    for v in views:
        first,last=v.get('diagnostic_clock_pair_before'),v.get('diagnostic_clock_pair_after')
        if not isinstance(first,dict) or not isinstance(last,dict):raise ValueError('both clock pairs required')
        keys=('source_clock_before','source_clock_after','sim_time_ms','car_timestamp_ms')
        values=[p.get(k) for p in (first,last) for k in keys]+[v.get('render_started_clock'),v.get('render_call_finished_clock')]
        if not all(type(x) in (float,int) and math.isfinite(x) and x>=0 for x in values):raise ValueError('finite nonnegative clock values required')
        if not (first['source_clock_before']<=first['source_clock_after']<=v['render_started_clock']<=v['render_call_finished_clock']<=last['source_clock_before']<=last['source_clock_after']):
            raise ValueError('clock read and render call intervals must be ordered')
        summaries.append({'view_id':v['view_id'],'before_car_minus_sim_ms':first['car_timestamp_ms']-first['sim_time_ms'],
            'after_car_minus_sim_ms':last['car_timestamp_ms']-last['sim_time_ms'],
            'before_read_duration_s':first['source_clock_after']-first['source_clock_before'],
            'after_read_duration_s':last['source_clock_after']-last['source_clock_before'],
            'sim_clock_reversed_during_render':last['sim_time_ms']<first['sim_time_ms'],
            'car_clock_reversed_during_render':last['car_timestamp_ms']<first['car_timestamp_ms']})
    return {'views':summaries,'physical_exposure_sync_verified':False,'training_admissible':False,
        'interpretation':'Diagnostic readings only: simulator time may be cached, render call boundaries do not establish physical exposure or a host clock mapping.'}

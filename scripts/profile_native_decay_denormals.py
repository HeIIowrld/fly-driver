"""Synthetic zero-drive CPU diagnostic on frozen native flight CSR, not road evidence."""
import argparse,gc,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from flybrain_drive.audited_native_flight_candidate import load_audited_native_flight_candidate
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.native_flight_decision_runtime import prepare_fixed_cpu_backend
from flybrain_drive.sac_unified_student import read

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate-plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(1)
    (model,*_),receipt=load_audited_native_flight_candidate(read(a.candidate_plan)['native'])
    warm=prepare_fixed_cpu_backend(model);circuit=model.motor.backbone.circuit
    run=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),candidate=receipt,seed=None,
        synthetic_initial_state='all ones',synthetic_drive='all zeros',steps=1200,optimizer_updates=0,vehicle_control_sent=False,
        scope='Numerical synthetic CSR diagnostic; not measured physiological activity, actual live hidden states or real road control'))
    shutil.copyfile(__file__,run/'executed-entrypoint.py')
    passes=[]
    before=gc.isenabled();gc.disable()
    try:
        for flush in [False,True]:
            assert torch.set_flush_denormal(flush)
            state=torch.ones(1,circuit.size);drive=torch.zeros_like(state);rows=[];durations=[]
            for i in range(1200):
                started=time.perf_counter();_,state=circuit._advance_tensor(drive,state);elapsed=time.perf_counter()-started
                durations.append(elapsed)
                if (i+1)%100==0:
                    bits=state.view(torch.int32)
                    count=int((((bits & 0x7f800000)==0)&((bits & 0x007fffff)!=0)).sum())
                    rows.append(dict(step=i+1,subnormal_state_count=count,phase_seconds_percentiles=np.percentile(durations[-100:],[50,95,100]).tolist()))
            passes.append(dict(flush_denormal=flush,rows=rows,final_state=state.numpy().copy()))
        difference=float(abs(passes[0]['final_state']-passes[1]['final_state']).max())
        for v in passes:v.pop('final_state')
        write_json(run/'result.json',dict(status='completed_synthetic_native_decay_diagnostic',passes=passes,final_state_max_difference=difference,
            actor_qualified=False,vehicle_control_sent=False,live_subnormal_state_proven=False,
            termination_reason='Synthetic fixed zero-drive CSR timing only; actual live state and full candidate parity required before any numerical execution change'))
    finally:
        torch.set_flush_denormal(False)
        gc.enable() if before else gc.disable()

if __name__=='__main__':main()

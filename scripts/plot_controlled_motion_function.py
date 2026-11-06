"""Standalone plot of synthetic held-out-texture functional ordering margins."""
import argparse,os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
import numpy as np
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True);p.add_argument('--functional',required=True);p.add_argument('--output',required=True);a=p.parse_args();roots=[Path(a.parent),Path(a.functional)];groups=[]
    for root in roots:
        r=read(root/'result.json');assert r['status']=='completed_controlled_motion_factorial_response' and sha256(root/'groups.json')==r['groups_sha256'];groups.append(read(root/'groups.json'))
    speeds=sorted({x['speed_mps'] for x in groups[0]});temperatures=sorted({x['temperature_c'] for x in groups[0]});fig,axes=plt.subplots(len(speeds),len(temperatures),figsize=(10,5.3),sharex=True,sharey=True,squeeze=False);maximum=max(abs(g['branches']['fine']['hypothesized_bend_margin'])*1000 for v in groups for g in v)
    for row,speed in enumerate(speeds):
        for col,temperature in enumerate(temperatures):
            ax=axes[row,col]
            for source,label,color in zip(groups,['Parent motion port','Functional fit'],['#3267ad','#d96c24']):
                selected=sorted([g for g in source if g['speed_mps']==speed and g['temperature_c']==temperature],key=lambda g:g['lateral_m']);assert len(selected)==3;ax.plot([g['lateral_m'] for g in selected],[1000*g['branches']['fine']['hypothesized_bend_margin'] for g in selected],marker='o',label=label,color=color)
            ax.axhline(.1,color='#666666',linestyle='--',linewidth=.8);ax.axhline(0,color='#bbbbbb',linewidth=.6);ax.set_title(f'{speed:g} m/s; body temperature {temperature:g} C',fontsize=10);ax.set_xticks([-2,0,2]);ax.grid(alpha=.2);ax.set_ylim(-maximum*1.2,maximum*1.2)
            if row==len(speeds)-1:ax.set_xlabel('Lateral position (m)')
    axes[0,0].legend(fontsize=8,loc='best');fig.supylabel('Signed bend margin (canonical x1000)',fontsize=11);fig.suptitle('Controlled synthetic bend response: reserved texture 781105',fontsize=13);fig.text(.5,.015,'Dashed: predeclared positive-order threshold. Synthetic body/scene hypotheses; not physical driving or tyre adaptation.',ha='center',fontsize=9);fig.tight_layout(rect=[.02,.04,1,.95]);out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),parent_result_sha256=sha256(roots[0]/'result.json'),functional_result_sha256=sha256(roots[1]/'result.json'),vehicle_control_sent=False));fig.savefig(out/'heldout-bend-margins.png',dpi=160);plt.close(fig);write_json(out/'result.json',dict(status='completed_controlled_motion_function_plot',image_sha256=sha256(out/'heldout-bend-margins.png'),groups_per_model=len(groups[0]),vehicle_control_sent=False))


if __name__=='__main__':main()

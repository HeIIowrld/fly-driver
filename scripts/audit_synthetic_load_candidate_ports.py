"""Offline adapter-only diagnostic, never grants model or vehicle qualification."""
import itertools,json,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from flybrain_drive.flight_body_feedback_adapter import CenteredSpeedBodyFeedbackAdapter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args();path=Path(a.candidate);basefile=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt');cand=torch.load(path,map_location='cpu',weights_only=True);orig=torch.load(basefile,map_location='cpu',weights_only=True)
 if cand.get('schema')!='flybrain.synthetic.load_adapter.v1' or cand.get('actor_qualified') is not False or cand.get('synthetic_only') is not True or cand.get('base_sha256')!=sha(basefile):raise ValueError('Exact synthetic candidate and original base required')
 w=cand['model']['projection.weight'];old=orig['model']['projection.weight'];assert torch.equal(w[:,:2],old[:,:2]) and torch.equal(w[:,6:],old[:,6:])
 c=CenteredSpeedBodyFeedbackAdapter(w.shape[0]);c.load_state_dict(cand['model']);b=CenteredSpeedBodyFeedbackAdapter(w.shape[0]);b.load_state_dict(orig['model']);c.eval();b.eval()
 body=torch.zeros((1,21));body[:,4]=0.;body[:,16]=3.;body[:,17:21]=3/.29;body[:,8:12]=.25;valid=torch.ones((1,21),dtype=torch.bool)
 with torch.no_grad():
  neutral=float((c(body,valid)-b(body,valid)).abs().max());cases={}
  for name,v in {'left':[.30,.20,.30,.20],'right':[.20,.30,.20,.30],'front':[.30,.30,.20,.20],'rear':[.20,.20,.30,.30]}.items():
   x=body.clone();x[:,8:12]=torch.tensor(v);cases[name]=(c(x,valid)-b(x,valid)).flatten().numpy()
  missing=[]
  for pattern in itertools.product((False,True),repeat=4):
   mask=valid.clone();mask[:,8:12]=torch.tensor(pattern);delta=float((c(body,mask)-b(body,mask)).abs().max());missing.append({'valid':list(pattern),'difference_from_base_max':delta})
  left=cases['left'];right=cases['right'];front=cases['front'];rear=cases['rear']
  report={'status':'completed_adapter_only_audit','candidate_sha256':sha(path),'base_sha256':sha(basefile),'non_load_weight_exact':True,'equal_load_max_port_delta':neutral,'load_column_row_sum_max':float(w[:,2:6].sum(dim=1).abs().max()),'left_right_port_antisymmetry_max':float(np.max(np.abs(left+right))),'front_rear_port_antisymmetry_max':float(np.max(np.abs(front+rear))),'partial_missing_max_port_delta':max(x['difference_from_base_max'] for x in missing if 0<sum(x['valid'])<4),'all_missing_port_delta':next(x['difference_from_base_max'] for x in missing if not any(x['valid'])),'mask_cases':missing,'port_perturbations':{k:float(np.max(np.abs(v))) for k,v in cases.items()},'actor_qualified':False,'interpretation':'Direct adapter output only; partial missing can change the learned load signal even when measurements are unavailable, and no real calibration or driving validity is asserted.'}
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

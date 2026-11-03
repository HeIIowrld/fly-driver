"""Offline probe of model-generated synthetic histories; never an actor."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

SOURCE=Path('D:/CodexResearchRuns/flybrain-dynamic-history-20261102-01')
EXPECTED={'scenes.npz':'63832ceaa911007cc1217d42ee34bc5cfd9827f0a45eb9c9e6b3120832ffb9b7','trajectory.npz':'2e4e5898c0f30fc7e179eddc97e6be401645976582b84c4d2494ded97e241123'}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 plan={'seed_train':883501,'seed_test':883502,'modes':['baseline','candidate'],'classes':[-1,0,1],'windows_m':{'before':[9,11.5],'after':[13,16]},'features':['retina_centered','motion','turn_intent'],'classifier':'train-only standardized ridge multiclass, alpha=10, intercept unregularized','sensor_boundary':'scenes and intent only; x and bend for offline grouping/labels only','unit':'per-decision; per-episode results reported, correlated samples','scope':'synthetic diagnostic; not an actor, biological or AC validation','input_sha256':EXPECTED,'program_sha256':sha(__file__)}
 (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
 for name,expected in EXPECTED.items():
  if sha(SOURCE/name)!=expected:raise ValueError('input hash mismatch '+name)
 with np.load(SOURCE/'scenes.npz',allow_pickle=False) as ss,np.load(SOURCE/'trajectory.npz',allow_pickle=False) as ts:
  data={}
  for mode in plan['modes']:
   for seed in (883501,883502):
    for bend in plan['classes']:
     key=f'{mode}_{seed}_{bend}';s=ss[key].astype(np.float64);t=ts[key]
     if s.shape!=(81,576) or t.shape!=(81,16) or not np.isfinite(s).all() or not np.isfinite(t).all():raise ValueError('invalid history '+key)
     s=s.reshape(81,4,9,16);ret=s[:,:,0,:];ret=ret-ret.mean(axis=-1,keepdims=True)
     data[key]={'x':t[:,1], 'retina_centered':ret.reshape(81,-1),'motion':s[:,:,1:,:].reshape(81,-1),'turn_intent':t[:,6:7]}
  report={}
  for win,(lo,hi) in plan['windows_m'].items():
   report[win]={}
   for feature in plan['features']:
    def batch(seed):
     xs=[];ys=[];groups=[]
     for mode in plan['modes']:
      for bend in plan['classes']:
       key=f'{mode}_{seed}_{bend}';v=data[key];mask=(v['x']>=lo)&(v['x']<hi);x=v[feature][mask]
       if len(x)==0:raise ValueError('empty interval '+key)
       xs.append(x);ys.extend([bend]*len(x));groups.extend([key]*len(x))
     return np.concatenate(xs),np.asarray(ys),np.asarray(groups)
    X,y,_=batch(883501);Q,z,groups=batch(883502)
    mean=X.mean(0);scale=X.std(0);scale=np.maximum(scale,1e-6);X=(X-mean)/scale;Q=(Q-mean)/scale
    # Center one-hot targets to fit a free intercept, then ridge in dual form.
    Y=np.eye(3)[y+1];ym=Y.mean(0);W=np.linalg.solve(X@X.T+10*np.eye(len(X)),Y-ym);scores=Q@X.T@W+ym;pred=np.argmax(scores,axis=1)-1
    confusion=[[int(np.sum((z==truth)&(pred==guess))) for guess in plan['classes']] for truth in plan['classes']]
    episodes={k:{'frames':int(np.sum(groups==k)),'accuracy':float(np.mean(pred[groups==k]==z[groups==k]))} for k in sorted(set(groups))}
    report[win][feature]={'train_frames':len(X),'test_frames':len(Q),'heldout_accuracy':float(np.mean(pred==z)),'confusion_rows_truth_columns_prediction':confusion,'episodes':episodes}
 result={'status':'completed_offline_synthetic_probe','plan_sha256':sha(out/'plan.json'),'results':report,'actor_qualified':False,'physical_sync_verified':False,'optimizer_updates':0,'vehicle_control_sent':False}
 (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

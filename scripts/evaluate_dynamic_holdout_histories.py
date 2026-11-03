"""Evaluate untouched-texture synthetic histories using a predeclared protocol."""
import hashlib,json
from pathlib import Path
import numpy as np
OLD=Path('D:/CodexResearchRuns/flybrain-dynamic-history-20261102-01')
NEW=Path('D:/CodexResearchRuns/flybrain-dynamic-holdout-20261102-01')
PLAN=Path('D:/CodexResearchRuns/flybrain-dynamic-holdout-eval-plan-20261102-01/plan.json')
OUT=Path('D:/CodexResearchRuns/flybrain-dynamic-holdout-eval-20261102-01')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(root):
 r=json.loads((root/'result.json').read_text(encoding='utf-8'))
 if r.get('status')!='completed_synthetic_dynamic_history_collection' or r.get('actor_qualified') is not False:raise ValueError('incomplete or qualified source')
 for fname,key in [('scenes.npz','scenes_sha256'),('trajectory.npz','output_sha256')]:
  if sha(root/fname)!=r[key]:raise ValueError('source mismatch '+fname)
 return r

def main():
 plan=json.loads(PLAN.read_text(encoding='utf-8')); rold=load(OLD);rnew=load(NEW)
 if rold['seeds']!=plan['train_seeds'] or rnew['seeds']!=plan['test_seeds']:raise ValueError('seed mismatch')
 OUT.mkdir(parents=True,exist_ok=False)
 sources={}
 for root in [OLD,NEW]:
  sources[str(root)]={n:sha(root/n) for n in ['scenes.npz','trajectory.npz','result.json']}
 data={}
 for root,seeds in [(OLD,plan['train_seeds']),(NEW,plan['test_seeds'])]:
  with np.load(root/'scenes.npz',allow_pickle=False) as ss,np.load(root/'trajectory.npz',allow_pickle=False) as ts:
   for seed in seeds:
    for mode in plan['modes']:
     for bend in plan['classes']:
      k=f'{mode}_{seed}_{bend}';s=ss[k].astype(float);t=ts[k].astype(float)
      if s.shape!=(81,576) or t.shape!=(81,16) or not np.isfinite(s).all() or not np.isfinite(t).all():raise ValueError('bad history '+k)
      s=s.reshape(81,4,9,16);r=s[:,:,0,:];data[k]={'retina_centered':(r-r.mean(-1,keepdims=True)).reshape(81,-1),'motion':s[:,:,1:,:].reshape(81,-1),'x':t[:,1]}
 results={};checks=[]
 for window,(lo,hi) in plan['windows_m'].items():
  results[window]={}
  for feature in plan['features']:
   def batch(seeds):
    xs=[];ys=[];gs=[]
    for seed in seeds:
     for mode in plan['modes']:
      for bend in plan['classes']:
       k=f'{mode}_{seed}_{bend}';v=data[k];mask=(v['x']>=lo)&(v['x']<hi);x=v[feature][mask]
       if not len(x):raise ValueError('empty episode')
       xs.append(x);ys.extend([bend]*len(x));gs.extend([k]*len(x))
    return np.concatenate(xs),np.array(ys),np.array(gs)
   train,y,_=batch(plan['train_seeds']);test,z,groups=batch(plan['test_seeds']);mu=train.mean(0);sd=train.std(0)
   scale={'global_rms':np.full_like(sd,max(1e-6,float(np.sqrt(np.mean(sd**2))))),'per_feature_sd':np.maximum(sd,1e-6)}
   results[window][feature]={}
   for name,d in scale.items():
    X=(train-mu)/d;Q=(test-mu)/d;Y=np.eye(3)[y+1];ym=Y.mean(0);w=np.linalg.solve(X@X.T+10*np.eye(len(X)),Y-ym);pred=np.argmax(Q@X.T@w+ym,axis=1)-1
    eps={k:{'frames':int(sum(groups==k)),'accuracy':float(np.mean(pred[groups==k]==z[groups==k]))} for k in sorted(set(groups))}
    results[window][feature][name]={'accuracy':float(np.mean(pred==z)),'episodes':eps,'confusion':[[int(np.sum((z==a)&(pred==b))) for b in plan['classes']] for a in plan['classes']]}
    if name=='global_rms':checks.extend([v['accuracy']>=.8 for v in eps.values()])
 trajectories={k:{'final_center_error':v['final_center_error'],'termination':v['termination'],'steps':v['steps']} for k,v in rnew['modes'].items()}
 report={'status':'completed_synthetic_holdout_evaluation','plan_sha256':sha(PLAN),'program_sha256':sha(__file__),'sources':sources,'screen_passed':all(checks),'individual_conditions_passed':sum(checks),'individual_conditions_total':len(checks),'results':results,'trajectories':trajectories,'actor_qualified':False,'vehicle_control_sent':False,'limits':plan['limits']}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
 print('screen',report['screen_passed'],sum(checks),'/',len(checks))
 for window in results:
  for feature in results[window]:
   for name,r in results[window][feature].items():print(window,feature,name,'accuracy',round(r['accuracy'],3),'worst_episode',min(x['accuracy'] for x in r['episodes'].values()))
 for k,v in trajectories.items():print(k,round(v['final_center_error'],4))
if __name__=='__main__':main()

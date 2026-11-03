"""Post-hoc paired scaling sensitivity for previously recorded synthetic histories.
This is a decoder diagnostic; it does not change actor inputs or weights.
"""
import json, hashlib
from pathlib import Path
import numpy as np

ROOT=Path('D:/CodexResearchRuns/flybrain-dynamic-history-20261102-01')
EXPECTED={'scenes.npz':'63832ceaa911007cc1217d42ee34bc5cfd9827f0a45eb9c9e6b3120832ffb9b7','trajectory.npz':'2e4e5898c0f30fc7e179eddc97e6be401645976582b84c4d2494ded97e241123'}
OUT=Path('D:/CodexResearchRuns/flybrain-dynamic-scale-diagnostic-20261102-01')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 plan={'status':'posthoc_design_before_running','note':'Previous holdout has been inspected, hence exploratory, not independent model selection','features':['retina_centered','motion'],'windows_m':{'before':[9,11.5],'after':[13,16]},'train_texture':883501,'heldout_texture':883502,'modes':['baseline','candidate'],'classes':[-1,0,1],'scaling':['feature_sd_floor_1e-6','feature_sd_floor_0.1_median_nonzero_sd','single_global_rms'],'ridge_alpha':10,'sensor_boundary':'only pooled image response; position and bend solely for selection and labels','source_sha256':EXPECTED,'program_sha256':sha(__file__)}
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
 for f,h in EXPECTED.items():
  if sha(ROOT/f)!=h:raise ValueError('source hash mismatch')
 results={}
 with np.load(ROOT/'scenes.npz',allow_pickle=False) as scene,np.load(ROOT/'trajectory.npz',allow_pickle=False) as traj:
  for window,(lo,hi) in plan['windows_m'].items():
   results[window]={}
   for feature in plan['features']:
    def batch(seed):
     xs=[];ys=[];groups=[]
     for mode in plan['modes']:
      for b in plan['classes']:
       k=f'{mode}_{seed}_{b}';v=scene[k].reshape(81,4,9,16);t=traj[k];mask=(t[:,1]>=lo)&(t[:,1]<hi)
       q=v[:,:,0,:]-v[:,:,0,:].mean(-1,keepdims=True) if feature=='retina_centered' else v[:,:,1:,:]
       xs.append(q.reshape(81,-1)[mask]);ys.extend([b]*int(mask.sum()));groups.extend([k]*int(mask.sum()))
     return np.concatenate(xs).astype('float64'),np.array(ys),np.array(groups)
    train,y,_=batch(883501);test,z,groups=batch(883502);mean=train.mean(0);sd=train.std(0);positive=sd[sd>1e-12]
    if len(positive)==0:raise ValueError('no feature variation')
    scales={'feature_sd_floor_1e-6':np.maximum(sd,1e-6),'feature_sd_floor_0.1_median_nonzero_sd':np.maximum(sd,max(1e-6,.1*np.median(positive))),'single_global_rms':np.full_like(sd,max(1e-6,float(np.sqrt(np.mean(sd**2)))))}
    results[window][feature]={}
    for name,scale in scales.items():
     X=(train-mean)/scale;Q=(test-mean)/scale;target=np.eye(3)[y+1];intercept=target.mean(0)
     w=np.linalg.solve(X@X.T+10*np.eye(len(X)),target-intercept)
     result={}
     for part,A,B in [('train',X,y),('holdout',Q,z)]:
      pred=np.argmax(A@X.T@w+intercept,axis=1)-1
      result[part]={'accuracy':float(np.mean(pred==B)),'class_accuracy':[float(np.mean(pred[B==c]==c)) for c in (-1,0,1)],'standardized_rms':float(np.sqrt(np.mean(A*A)))}
      if part=='holdout':result[part]['episodes']={k:float(np.mean(pred[groups==k]==B[groups==k])) for k in sorted(set(groups))}
     result['scale_min']=float(scale.min());result['scale_max']=float(scale.max());results[window][feature][name]=result
 report={'status':'completed_posthoc_scaling_sensitivity','plan_sha256':sha(OUT/'plan.json'),'results':results,'actor_qualified':False,'vehicle_control_sent':False,'optimizer_updates':0}
 (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf8')
 for window in results:
  for feature in results[window]:
   for name,r in results[window][feature].items(): print(window,feature,name,'train',round(r['train']['accuracy'],3),'heldout',round(r['holdout']['accuracy'],3),'classes',r['holdout']['class_accuracy'],'rms',round(r['holdout']['standardized_rms'],3))
if __name__=='__main__':main()

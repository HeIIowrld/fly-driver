"""Post-hoc train/holdout diagnosis for the declared synthetic probe."""
import json,hashlib
from pathlib import Path
import numpy as np
p=Path('D:/CodexResearchRuns/flybrain-dynamic-history-20261102-01'); out=Path('D:/CodexResearchRuns/flybrain-dynamic-visual-probe-20261102-01')
result={}
with np.load(p/'scenes.npz') as s,np.load(p/'trajectory.npz') as t:
 for label,lo,hi in [('before',9,11.5),('after',13,16)]:
  result[label]={}
  for feature in ['retina_centered','motion','turn_intent']:
   blocks=[]
   for seed in (883501,883502):
    x=[];y=[]
    for mode in ('baseline','candidate'):
     for bend in (-1,0,1):
      k=f'{mode}_{seed}_{bend}';v=s[k].reshape(81,4,9,16);mask=(t[k][:,1]>=lo)&(t[k][:,1]<hi)
      q=v[:,:,0,:]-v[:,:,0,:].mean(-1,keepdims=True) if feature=='retina_centered' else (v[:,:,1:,:] if feature=='motion' else t[k][:,6:7])
      x.append(q.reshape(81,-1)[mask]);y.extend([bend]*int(mask.sum()))
    blocks.append((np.concatenate(x).astype('float64'),np.array(y)))
   (X,y),(Q,z)=blocks;mu=X.mean(0);sd=np.maximum(X.std(0),1e-6);X=(X-mu)/sd;Q=(Q-mu)/sd;Y=np.eye(3)[y+1];ym=Y.mean(0);W=np.linalg.solve(X@X.T+10*np.eye(len(X)),Y-ym)
   record={}
   for name,A,B in [('train',X,y),('heldout',Q,z)]:
    pred=np.argmax(A@X.T@W+ym,axis=1)-1
    record[name]={'accuracy':float(np.mean(pred==B)),'accuracy_by_class':[float(np.mean(pred[B==c]==c)) for c in (-1,0,1)],'standardized_feature_rms':float(np.sqrt(np.mean(A*A)))}
   result[label][feature]=record
report={'status':'completed_posthoc_exploratory','program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_plan_sha256':hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),'note':'RMS compares feature distribution to train-only mean and scale; not a generalization bound or causal proof. Trajectory samples correlated.','results':result}
(out/'posthoc-train-shift.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))

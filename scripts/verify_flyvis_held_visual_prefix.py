"""Compare speculative prefix against original held-input FlyVis on archived RGB."""
import argparse,hashlib,io,json,os,time,zipfile
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.flyvis_camera import FlyvisCameraProjector
from flybrain_drive.flyvis_compat import install_datamate_cache_compat
from flybrain_drive.flight_stream import held_input_batches
from flybrain_drive.graph import sha256
from flybrain_drive.held_visual_prefix import HeldVisualPrefix


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run=begin_run(a.output,dict(program_sha256=sha256(__file__),prefix_sha256=sha256('src/flybrain_drive/held_visual_prefix.py'),actor=False,optimizer_updates=0))
 result={'status':'failed','vehicle_control_sent':False,'actor_deployed':False}
 try:
  cfg=json.loads(Path('configs/flight-stream-batched-v1-20261029.json').read_text());camera=json.loads(Path(cfg['camera_config']).read_text());modeldir=Path(cfg['author_model_dir']);cache=Path(cfg['cache_root'])
  if sha256(cfg['camera_config'])!=cfg['camera_config_sha256'] or sha256(modeldir/'best_chkpt')!=cfg['author_weights_sha256']:raise ValueError('Exact visual sources required')
  archive=Path('C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-multirate-early-context/vision');bundle=json.loads((archive/'bundle.json').read_text());manifest=json.loads((archive/'archive.json').read_text());zipfile_path=archive/'camera.zip'
  if sha256(zipfile_path)!=bundle['sha256'] or sha256(archive/'archive.json')!=bundle['archive_manifest_sha256']:raise ValueError('Archive identity changed')
  expected={x['file']:x for x in manifest};rgb=[]
  with zipfile.ZipFile(zipfile_path) as z:
   for name in ('front','left','right','rear'):
    f='0026-'+name+'.png';blob=z.read(f)
    if hashlib.sha256(blob).hexdigest()!=expected[f]['sha256']:raise ValueError('Original image changed')
    rgb.append(np.asarray(Image.open(io.BytesIO(blob)).convert('RGB')).copy())
  os.environ['FLYVIS_ROOT_DIR']=str(cache.resolve());os.environ['MPLBACKEND']='Agg';install_datamate_cache_compat(cache)
  import flyvis
  torch.set_num_threads(1);torch.manual_seed(863009)
  with torch.no_grad(),torch.device('cuda'):
   network=flyvis.Network(**yaml.safe_load((modeldir/'_meta.yaml').read_text())['config']['network']).to('cuda');network.load_state_dict(torch.load(modeldir/'best_chkpt',map_location='cpu',weights_only=True)['network'],strict=True);network.eval().requires_grad_(False)
   nodes=network.connectome.nodes;receptors=network.stimulus.input_index[0];projector=FlyvisCameraProjector(camera,nodes.u[:][receptors],nodes.v[:][receptors]);held=torch.as_tensor(np.stack([projector.sample(x) for x in rgb]),device='cuda')
   initial=network.steady_state(1.,.005,batch_size=4,value=.5);durations=[.002,.005,.0237,.06916,.08,.0923,.15];rows=[]
   prefix=HeldVisualPrefix(network,initial,held,steps=16);torch.cuda.synchronize()
   # Prefix is single-use in production; reuse a freshly built one per comparison.
   for elapsed in durations:
    prefix=HeldVisualPrefix(network,initial,held,steps=16)
    candidate=prefix.commit(elapsed);baseline=initial
    for dt,n in held_input_batches(elapsed):baseline=network.simulate(held[:,None,None,:].expand(-1,n,-1,-1),dt=dt,initial_state=baseline,as_states=True)[-1]
    torch.cuda.synchronize();differences={}
    for section in ('nodes','edges'):
     aa=getattr(candidate,section);bb=getattr(baseline,section)
     if set(aa.keys())!=set(bb.keys()):raise ValueError('Full dynamic state keys changed')
     for key in aa.keys():
      if torch.is_tensor(aa[key]):differences[section+'.'+key]=float((aa[key]-bb[key]).abs().max().item())
      elif aa[key]!=bb[key]:raise ValueError('Non-tensor state differs')
    if not differences or max(differences.values())>2e-6:raise ValueError('Dynamic state parity failed: '+str(differences))
    rows.append({'elapsed':elapsed,'state_max_abs_differences':differences})
  result.update(status='completed_archived_rgb_prefix_parity',durations=durations,comparisons=rows,archive_sha256=sha256(zipfile_path),checkpoint_sha256=sha256(modeldir/'best_chkpt'),interpretation='One stationary archived four-view input and one initial state; offline computational parity, not live camera, physiology or driving')
 except Exception as e:result.update(error=type(e).__name__,reason=str(e))
 finally:write_json(run/'result.json',result)
 print(json.dumps(result,allow_nan=False))
 if result['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()

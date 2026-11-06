"""Original front-frame contact sheet for the diagnostic receiver/donor pairs."""
import argparse,io,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import begin_run,write_json
from flybrain_drive.graph import sha256
from flybrain_drive.sac_unified_student import read
from scripts.train_qualified_sac_unified import load_episode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.diagnostic);r=read(root/'result.json');m=read(root/'manifest.json')['config'];actor=Path(m['arguments']['actor_data']);d,c,provenance=load_episode(actor);cfg=read(actor/'manifest.json')['config'];recording=Path(cfg['arguments']['recording']);iv=read(Path(m['arguments']['interventions'])/'result.json');qr=read(Path(cfg['arguments']['qualification'])/'result.json')
    assert r['status']=='completed_precorner_road_pose_diagnostic' and provenance['arrays_sha256']==m['actor_arrays_sha256'] and sha256(root/'diagnostic-road-poses.npz')==r['arrays_sha256']
    with np.load(root/'diagnostic-road-poses.npz') as z:brackets=z['pose_brackets'].copy();event_map=z['event_to_camera'].copy()
    out=begin_run(a.output,dict(arguments=vars(a),program_sha256=sha256(__file__),diagnostic_result_sha256=sha256(root/'result.json'),camera_bundle_sha256=qr['camera_bundle_sha256'],actor_input=False,vehicle_control_sent=False,scope='Unmodified original front RGB frames copied side by side, with diagnostic source-bracket mean offset/speed annotations. Not pose-corrected images, neural replay or counterfactual driving evidence.'));sheet=Image.new('RGB',(640,204*len(iv['windows'])),(22,22,22));draw=ImageDraw.Draw(sheet);records=[]
    with CameraArchive(recording/'camera/vision') as archive:
        assert archive.bundle_sha256==qr['camera_bundle_sha256'];rows=[json.loads(line) for line in archive.read('stream.jsonl').decode('utf8').splitlines()]
        for n,w in enumerate(iv['windows']):
            indices=np.asarray(w['pre_indices']);i=int(indices[np.argmin(abs(d['decision_time'][indices]-(w['receiver']['time']-.45)))]);stamp=w['receiver']['donor_time']+d['decision_time'][i]-w['receiver']['time'];j=int(np.searchsorted(d['decision_time'],stamp,side='right')-1)
            for col,(label,event) in enumerate([('receiver',i),('opposite donor',j)]):
                camera=int(d['camera_index'][event]);row=rows[camera];frame=archive.json(Path(row['file']).name);view=next(v for v in frame['views'] if v['view_id']=='front');blob=archive.read(view['file']);image=Image.open(io.BytesIO(blob)).convert('RGB');assert image.size==(320,180);sheet.paste(image,(col*320,n*204+24));pose=brackets[event_map[event]].mean(0);draw.text((col*320+4,n*204+4),f'{n+1} {label} offset {pose[0]:+.2f}m v {pose[2]:.1f}m/s',fill=(255,255,255));records.append(dict(window=n,role=label,actor_event=event,camera_index=camera,member=view['file'],member_sha256=__import__('hashlib').sha256(blob).hexdigest(),source_bracket_mean_offset_m=float(pose[0]),source_bracket_mean_speed_mps=float(pose[2])))
    sheet.save(out/'original-pairs.png');write_json(out/'result.json',dict(status='completed_original_precorner_pose_contact_sheet',image_sha256=sha256(out/'original-pairs.png'),original_frames=len(records),frames=records,actor_input=False,vehicle_control_sent=False,synthetic_or_pose_corrected_pixels=False))


if __name__=='__main__':main()

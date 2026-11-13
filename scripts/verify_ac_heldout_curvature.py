"""Fit a pixel/motion probe on archived AC video, with whole map corners held out."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from flybrain_drive.ac_camera_archive import CameraArchive
from flybrain_drive.evidence import write_json
from flybrain_drive.graph import sha256


HORIZONS = np.array([5., 10., 20.])
BUFFER_M = 90.
TEST_M = 32.
VIEWS = ('front', 'left', 'right', 'rear')


def road_model(track):
    xy = np.genfromtxt(track, names=True, delimiter=',')
    xy = np.unique(np.c_[xy['pos_x'], xy['pos_y']], axis=0, return_index=True)
    xy = xy[0][np.argsort(xy[1])]
    xy = np.vstack([xy, xy[0]])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    spline = CubicSpline(arc, xy, axis=0, bc_type='periodic')
    sampled = np.arange(0, arc[-1], 1.)
    tree = cKDTree(spline(sampled))
    speed = spline(sampled, 1)
    curvature = np.cross(np.c_[speed, np.zeros(len(speed))],
                         np.c_[spline(sampled, 2), np.zeros(len(speed))])[:, 2]
    curvature /= np.linalg.norm(speed, axis=1)**3
    # Pick both signed extrema from the map before inspecting training or test predictions.
    center = sampled[int(np.argmax(curvature))]
    away = np.abs((sampled-center+arc[-1]/2) % arc[-1]-arc[-1]/2) > 2*BUFFER_M
    other = sampled[np.argmin(np.where(away, curvature, np.inf))]
    return spline, sampled, tree, arc[-1], np.array([center, other])


def read_examples(camera, road):
    spline, sampled, tree, length, centers = road
    with CameraArchive(camera) as archive:
        stream = [json.loads(line) for line in archive.read('stream.jsonl').splitlines()]
        images, truth, groups, ids = [], [], [], []
        for i in range(2, len(stream), 2):
            row, old = stream[i], stream[i-2]
            if (row['ideal_line_shown'] or row['aid_probe_only'] or old['ideal_line_shown']
                    or old['aid_probe_only'] or old['generation'] != row['generation']):
                continue
            meta = archive.json(Path(row['file']).name)
            front = next(v for v in meta['views'] if v['view_id']=='front')
            body = front['source_body_before_render']
            if not body['physics_available'] or body['speed_m_s'] < 8:
                continue
            # Map position and orientation are supervision only. AC SDK x=native z, y=native x.
            point = np.asarray(body['renderer_only_position'])[[2,0]]
            dist, index = tree.query(point)
            origin = sampled[index]
            tangent = spline(origin, 1)
            look = np.asarray(body['renderer_only_look'])[[2,0]]
            if dist > 7 or tangent @ look < .7:
                continue
            separation = np.abs((origin-centers+length/2) % length-length/2)
            group = int(np.argmin(separation))
            if separation[group] > BUFFER_M:
                group = -1
            elif separation[group] > TEST_M:
                continue
            local = spline(origin)
            displacement = spline((origin+HORIZONS)%length)-local
            direction = tangent/np.linalg.norm(tangent)
            lateral = direction[0]*displacement[:,1]-direction[1]*displacement[:,0]
            target = 2*lateral/np.sum(displacement**2,axis=1)
            t0 = {v['view_id']:v for v in old['views']}
            t1 = {v['view_id']:v for v in row['views']}
            if not all(.10 < t1[k]['render_started_lower']-t0[k]['render_started_lower'] < .35 for k in VIEWS):
                continue

            def pixels(view):
                raw = archive.read(Path(view['file']).name)
                gray = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError('Archived image decode failed')
                return cv2.resize(gray,(20,12),interpolation=cv2.INTER_AREA).ravel()/255.

            now = np.concatenate([pixels(t1[k]) for k in VIEWS])
            past = np.concatenate([pixels(t0[k]) for k in VIEWS])
            images.append(np.r_[now,now-past])
            truth.append(target)
            groups.append(group)
            ids.append(row['capture_id'])
        return np.array(images),np.array(truth),np.array(groups),np.array(ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--track', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.train.resolve()==args.test.resolve():
        raise ValueError('Distinct recordings required')
    road=road_model(args.track)
    x,y,g,_=read_examples(args.train,road)
    a,b,h,ids=read_examples(args.test,road)
    x,y=x[g==-1],y[g==-1]
    if len(x)<100 or any(np.count_nonzero(h==j)<10 for j in (0,1)):
        raise ValueError(f'Insufficient separated observations: train={len(x)}, test={np.unique(h,return_counts=True)}')
    predictions={}
    models={}
    for key,width in [('pixels',len(x[0])//2),('pixels_and_motion',len(x[0]))]:
        train=x[:,:width]
        mean=train.mean(0)
        scale=np.maximum(train.std(0),.02)
        train=(train-mean)/scale
        target_mean=y.mean(0)
        weight=train.T@np.linalg.solve(train@train.T+len(x)*.15*np.eye(len(x)),y-target_mean)
        predictions[key]=((a[:,:width]-mean)/scale)@weight+target_mean
        models.update({key+'_mean':mean,key+'_scale':scale,key+'_weight':weight,key+'_bias':target_mean})
    predictions['train_mean']=np.broadcast_to(y.mean(0),b.shape)
    args.output.mkdir(parents=True,exist_ok=False)
    metrics={}
    for j in (0,1):
        mask=h==j
        significant=np.abs(b[mask])>=.002
        metrics[str(j)]={key:dict(
            rmse_per_m=np.sqrt(np.mean((p[mask]-b[mask])**2,axis=0)).tolist(),
            signed_accuracy=np.divide(((np.sign(p[mask])==np.sign(b[mask]))&significant).sum(0),
                                     significant.sum(0).clip(1)).tolist(),
            significant=significant.sum(0).tolist(),
        ) for key,p in predictions.items()}
    archive_hash={k:sha256(p/'bundle.json') for k,p in [('train',args.train),('test',args.test)]}
    np.savez_compressed(args.output/'predictions.npz',truth=b[h>=0],groups=h[h>=0],capture_ids=ids[h>=0],
                        **{k:v[h>=0] for k,v in predictions.items()},**models)
    write_json(args.output/'result.json',dict(status='completed',program_sha256=sha256(__file__),
        archive_descriptor_sha256=archive_hash,track_sha256=sha256(args.track),
        held_out_arc_m=road[-1].tolist(),horizons_m=HORIZONS.tolist(),buffer_m=BUFFER_M,
        train_rows=len(x),held_out_rows=[int((h==i).sum()) for i in (0,1)],metrics=metrics,
        outputs_sha256=sha256(args.output/'predictions.npz'),actor_inputs='four-view pixels and temporal pixel differences',
        scope='offline same-track corner holdout; pixel baseline, no circuit or closed-loop claim'))


if __name__=='__main__':
    main()

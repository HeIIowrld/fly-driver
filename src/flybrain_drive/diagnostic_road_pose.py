"""Evaluation-only road coordinates; never an actor observation adapter."""
import numpy as np


def wrap_angle(value):
    return np.arctan2(np.sin(value),np.cos(value))


def project_road_pose(points,yaw,centerline):
    """Exact nearest-segment projection of a closed ordered centerline.

    Offset is positive to the left of its ordered tangent. Heading is vehicle
    yaw minus local road tangent, in radians. No racing-line target is used.
    """
    points=np.asarray(points,float);yaw=np.asarray(yaw,float);line=np.asarray(centerline,float)
    if points.ndim!=2 or points.shape[1]!=2 or yaw.shape!=(len(points),) or line.ndim!=2 or line.shape[1]!=2 or len(line)<3 or any(not np.isfinite(x).all() for x in (points,yaw,line)):raise ValueError('Complete finite diagnostic points/yaw/closed road required')
    if np.linalg.norm(line[-1]-line[0])<1e-8:line=line[:-1]
    vectors=np.roll(line,-1,axis=0)-line;length2=np.sum(vectors*vectors,axis=1)
    if len(line)<3 or np.min(length2)<1e-12:raise ValueError('Nondegenerate closed road required')
    answers=[]
    for begin in range(0,len(points),128):
        p=points[begin:begin+128];relative=p[:,None,:]-line[None];fraction=np.clip(np.sum(relative*vectors[None],axis=-1)/length2,0,1);nearest=line[None]+fraction[:,:,None]*vectors[None];distance2=np.sum((p[:,None]-nearest)**2,axis=-1);segment=np.argmin(distance2,axis=1);j=np.arange(len(p));v=vectors[segment];length=np.sqrt(length2[segment]);difference=p-nearest[j,segment];offset=(v[:,0]*difference[:,1]-v[:,1]*difference[:,0])/length;angle=wrap_angle(yaw[begin:begin+len(p)]-np.arctan2(v[:,1],v[:,0]));answers.append(np.column_stack([offset,angle,np.sqrt(distance2[j,segment]),segment,fraction[j,segment]]))
    return np.concatenate(answers) if answers else np.empty((0,5))


def paired_pose_bounds(receiver,donor):
    """Worst observed endpoint difference, not a physical interpolation proof."""
    receiver,donor=np.asarray(receiver,float),np.asarray(donor,float)
    if receiver.shape!=donor.shape or receiver.ndim!=3 or receiver.shape[1:]!=(2,3) or not np.isfinite(receiver).all() or not np.isfinite(donor).all():raise ValueError('Aligned finite source brackets [time,2,offset/yaw/speed] required')
    delta=receiver[:,:,None,:]-donor[:,None,:,:]
    delta[:,:,:,1]=wrap_angle(delta[:,:,:,1])
    return np.max(abs(delta),axis=(1,2))

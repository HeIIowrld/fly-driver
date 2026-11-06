"""Analytic synthetic planar motion and explicit known/missing body channels.

X-forward/Y-left/Z-up. This is a declared kinematic stimulus, not an AC sensor
registration, tyre plant, measured fly physiology, or world heading actor input.
"""
import math
import numpy as np

_GAUSS_X,_GAUSS_W=np.polynomial.legendre.leggauss(16)

def _require(target,onset,duration):
    if not np.isfinite([target,onset,duration]).all() or abs(target)>.5 or onset<0 or not .05<=duration<=2:raise ValueError('Finite bounded synthetic yaw profile required')

def yaw_profile(time,target,*,onset,duration):
    _require(target,onset,duration)
    if not np.isfinite(time).all():raise ValueError('Finite source-relative sample time required')
    elapsed=np.maximum(np.asarray(time,float)-onset,0.);limited=np.minimum(elapsed,duration);phase=np.pi*limited/duration;rate=np.where(elapsed<duration,target*np.sin(phase),0.);angle=target*duration/np.pi*(1-np.cos(phase));return rate,angle

def planar_pose(time,speed,target,*,onset,duration,lateral=0.,yaw0=0.):
    _require(target,onset,duration)
    if not np.isfinite([time,speed,lateral,yaw0]).all() or not 0<=time<=3 or not 0<=speed<=10 or abs(lateral)>10 or abs(yaw0)>1:raise ValueError('Bounded finite synthetic trajectory required')
    boundaries=sorted(set([0.,float(time),*([onset] if 0<onset<time else []),*([onset+duration] if 0<onset+duration<time else [])]));x=y=0.
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        half=(end-begin)/2;t=(begin+end)/2+half*_GAUSS_X;_,angle=yaw_profile(t,target,onset=onset,duration=duration);x+=speed*half*float(_GAUSS_W@np.cos(angle+yaw0));y+=speed*half*float(_GAUSS_W@np.sin(angle+yaw0))
    rate,angle=yaw_profile(time,target,onset=onset,duration=duration);return np.asarray([x,lateral+y,yaw0+float(angle),float(rate)])

def synthetic_turn_body(time,speed,target,temperature,*,onset,duration):
    if not np.isfinite([speed,temperature]).all() or not 0<=speed<=10 or not 0<=temperature<=150:raise ValueError('Declared synthetic body values required')
    rate,_=yaw_profile(time,target,onset=onset,duration=duration);rate=float(rate);body=np.zeros(21);body[4]=rate;body[12:16]=temperature;body[16]=speed;valid=np.zeros(21,dtype=bool);valid[[0,1,4,12,13,14,15,16]]=True
    extension=np.zeros(19);extension[1]=speed*rate;extension[11:15]=temperature;extension_valid=np.zeros(19,dtype=bool);extension_valid[[0,1,11,12,13,14]]=True
    return body,valid,extension,extension_valid

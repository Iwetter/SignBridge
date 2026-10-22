"""Versioned, class-independent geometry derived from saved normalized xyz.

The raw Dataset v1 vectors remain unchanged. Training, evaluation and inference
use this same map. Global image orientation is retained in the first 63 values.
"""
import numpy as np

RAW = 'coordinates-v1'
GEOMETRIC = 'coordinates-geometry-v1'
FEATURE_SETS = (RAW, GEOMETRIC)
CHAINS = ((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))
TIPS = (4,8,12,16,20)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-6)


def _derive(points):
    values=[];names=[]
    def scalar(name,value):
        names.append(name);values.append(np.asarray(value)[...,None])
    def vector(name,value):
        for j,axis in enumerate('xyz'):scalar(name+'_'+axis,value[...,j])
    for f,chain in enumerate(CHAINS):
        bones=[points[...,b,:]-points[...,a,:] for a,b in zip(chain,chain[1:])]
        for j in range(3):scalar(f'finger{f}_joint{j}_cos',np.sum(_unit(bones[j])*_unit(bones[j+1]),axis=-1))
        length=sum(np.linalg.norm(b,axis=-1) for b in bones)
        scalar(f'finger{f}_extension',np.linalg.norm(points[...,chain[-1],:]-points[...,chain[0],:],axis=-1)/np.maximum(length,1e-6))
        vector(f'finger{f}_direction',_unit(points[...,chain[-1],:]-points[...,chain[1],:]))
    for i,a in enumerate(TIPS):
        for b in TIPS[i+1:]:scalar(f'tip_{a}_{b}_distance',np.linalg.norm(points[...,a,:]-points[...,b,:],axis=-1))
    for j in range(5,21):scalar(f'thumb4_landmark{j}_distance',np.linalg.norm(points[...,4,:]-points[...,j,:],axis=-1))
    for a,b in ((4,6),(4,10),(4,14),(8,12),(6,10)):
        vector(f'relative_{a}_{b}',points[...,a,:]-points[...,b,:])
    forward=_unit(points[...,[5,9,13,17],:].mean(axis=-2)-points[...,0,:])
    lateral=_unit(points[...,5,:]-points[...,17,:])
    vector('palm_forward',forward);vector('palm_lateral',lateral)
    vector('palm_normal',_unit(np.cross(lateral,forward)))
    # Signed separation/crossing quantities, never a class-specific override.
    for a,b in ((8,12),(6,10),(5,9)):
        scalar(f'separation_{a}_{b}_lateral',np.sum((points[...,a,:]-points[...,b,:])*lateral,axis=-1))
    a,b,c,d=[points[...,i,:2] for i in (6,8,10,12)]
    def cross2(u,v):return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    for j,(u,v) in enumerate(((b-a,c-a),(b-a,d-a),(d-c,a-c),(d-c,b-c))):
        scalar(f'index_middle_crossing_{j}',cross2(u,v))
    return np.concatenate(values,axis=-1),names


def feature_names(feature_set=RAW):
    if feature_set not in FEATURE_SETS:raise ValueError(f'Unsupported feature set: {feature_set}')
    names=[f'landmark{i}_{axis}' for i in range(21) for axis in 'xyz']
    return names+(_derive(np.zeros((21,3),np.float32))[1] if feature_set==GEOMETRIC else [])


def model_features(features,feature_set=RAW):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim not in (1,2) or x.shape[-1]!=63 or not np.isfinite(x).all() or np.any(np.abs(x)>4.000001):
        raise ValueError('Feature preprocessing requires finite normalized Dataset v1 vectors (...,63) in [-4,4].')
    if feature_set==RAW:return x
    if feature_set!=GEOMETRIC:raise ValueError(f'Unsupported feature set: {feature_set}')
    extra,_=_derive(x.reshape(*x.shape[:-1],21,3))
    return np.concatenate((x,extra),axis=-1).astype(np.float32)

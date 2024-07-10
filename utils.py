r"""
    Utils for the project.
"""

import re
import json
import enum
import torch
import torch.nn as nn
import random
import numpy as np
from glob import glob
from typing import Dict

import articulate as art
from config import General, Paths


_smpl_to_rbdl = [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 3, 4, 5, 12, 13, 14, 21, 22, 23, 30, 31, 32, 6, 7, 8,
                 15, 16, 17, 24, 25, 26, 36, 37, 38, 45, 46, 47, 51, 52, 53, 57, 58, 59, 63, 64, 65, 39, 40, 41,
                 48, 49, 50, 54, 55, 56, 60, 61, 62, 66, 67, 68, 33, 34, 35, 42, 43, 44]
_rbdl_to_smpl = [0, 1, 2, 12, 13, 14, 24, 25, 26, 3, 4, 5, 15, 16, 17, 27, 28, 29, 6, 7, 8, 18, 19, 20, 30, 31, 32,
                 9, 10, 11, 21, 22, 23, 63, 64, 65, 33, 34, 35, 48, 49, 50, 66, 67, 68, 36, 37, 38, 51, 52, 53, 39,
                 40, 41, 54, 55, 56, 42, 43, 44, 57, 58, 59, 45, 46, 47, 60, 61, 62]
_rbdl_to_bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 33, 34, 35,
                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 63, 64, 65, 66, 67, 68]
smpl_to_rbdl_data = _smpl_to_rbdl

# 24 all joint, convert smpl[24] to rbdl[24]
idx_smpl_to_rbdl = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23, 12, 15]
# 24 all joint, convert rbdl[24] to smpl[24]
idx_rbdl_to_smpl = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 22, 12, 17, 23, 13, 18, 14, 19, 15, 20, 16, 21]
# 16 MotionAug joints to 24 SMPL joints
idx_motionaug_to_smpl = [0, 3, 9, 12, 2, 5, 8, 17, 19, 21, 1, 4, 7, 16, 18, 20]

class Body(enum.Enum):
    r"""
    Prefix L = left; Prefix R = right.
    """
    ROOT = 2        # 0
    PELVIS = 2      # 0
    SPINE = 2       # 0
    LHIP = 5        # 1
    RHIP = 17       # 2
    SPINE1 = 29     # 3
    LKNEE = 8       # 4
    RKNEE = 20      # 5
    SPINE2 = 32     # 6
    LANKLE = 11     # 7
    RANKLE = 23     # 8
    SPINE3 = 35     # 9
    LFOOT = 14      # 10
    RFOOT = 26      # 11
    NECK = 68       # 12
    LCLAVICLE = 38  # 13
    RCLAVICLE = 53  # 14
    HEAD = 71       # 15
    LSHOULDER = 41  # 16
    RSHOULDER = 56  # 17
    LELBOW = 44     # 18
    RELBOW = 59     # 19
    LWRIST = 47     # 20    
    RWRIST = 62     # 21
    LHAND = 50      # 22
    RHAND = 65      # 23
    
    
def init_rand_seed(seed: int):
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_and_concat(glb_acc, glb_ori, acc_scale:float=General.acc_scale):
    ''' Normalization part in data preprocessing.
    args:
        acc_scale: transpose = 30, pip = 1.
    '''
    glb_acc = glb_acc.view(-1, 6, 3)
    glb_ori = glb_ori.view(-1, 6, 3, 3)
    # TODO: What is acc scale?
    # bmm: batch matrix-matrix product, see pytorch doc.
    acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_ori[:, -1]) / acc_scale
    ori = torch.cat((glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]), glb_ori[:, 5:]), dim=1)
    data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
    return data


def syn_acc(v, smooth_n=4):
    ''' Synthesize accelerations from vertex positions.
        Could be faster.
    '''
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def syn_acc2(v, smooth_n=4):
    ''' Synthesize accelerations from vertex positions.
        Use reflection padding strategy.
    ''' 
    acc = (v[2*smooth_n:] + v[:-2*smooth_n] - 2*v[smooth_n:-smooth_n]) * 3600 / (smooth_n * smooth_n)
    prepend = 2*acc[:1] - acc[range(smooth_n,0,-1)]
    append = 2*acc[-1:] - acc[range(-2,-smooth_n-2,-1)]
    acc = torch.cat([prepend, acc, append], dim=0)
    return acc


def syn_vel(p, smooth_n=2):
    ''' Synthesize velocity from joint positions.
        Use reflection padding strategy.
    ''' 
    vel = (p[2*smooth_n:] - p[:-2*smooth_n]) * 60 / (2 * smooth_n)
    prepend = 2*vel[:1] - vel[range(smooth_n,0,-1)]
    append = 2*vel[-1:] - vel[range(-2,-smooth_n-2,-1)]
    vel = torch.cat([prepend, vel, append], dim=0)
    return vel


def print_title(s):
    print('============ %s ============' % s)


def smpl_to_rbdl(poses, trans):
    r"""
    Convert smpl poses and translations to robot configuration q. (numpy, batch)

    :param poses: Array that can reshape to [n, 24, 3, 3].
    :param trans: Array that can reshape to [n, 3].
    :return: Ndarray in shape [n, 75] (3 root position + 72 joint rotation).
    """
    poses = np.array(poses).reshape(-1, 24, 3, 3)   # [N, 24, 3, 3]
    trans = np.array(trans).reshape(-1, 3)          # [N, 3]
    euler_poses = art.math.rotation_matrix_to_euler_angle_np(poses[:, 1:], 'XYZ').reshape(-1, 69)   # joints, [N, 69]
    euler_glbrots = art.math.rotation_matrix_to_euler_angle_np(poses[:, :1], 'xyz').reshape(-1, 3)  # root, [N, 3]
    euler_glbrots = art.math.euler_convert_np(euler_glbrots[:, [2, 1, 0]], 'xyz', 'zyx')            # root, [N, 3]
    qs = np.concatenate((trans, euler_glbrots, euler_poses[:, _smpl_to_rbdl]), axis=1)
    qs[:, 3:] = art.math.normalize_angle(qs[:, 3:])
    return qs   # [N, 75]


def rbdl_to_smpl(qs):
    r"""
    Convert robot configuration q to smpl poses and translations. (numpy, batch)

    :param qs: Ndarray that can reshape to [n, 75] (3 root position + 72 joint rotation).
    :return: Poses ndarray in shape [n, 24, 3, 3] and translation ndarray in shape [n, 3].
    """
    qs = qs.reshape(-1, 75)
    trans, euler_glbrots, euler_poses = qs[:, :3], qs[:, 3:6], qs[:, 6:][:, _rbdl_to_smpl]
    euler_glbrots = art.math.euler_convert_np(euler_glbrots, 'zyx', 'xyz')[:, [2, 1, 0]]
    glbrots = art.math.euler_angle_to_rotation_matrix_np(euler_glbrots, 'xyz').reshape(-1, 1, 3, 3)
    poses = art.math.euler_angle_to_rotation_matrix_np(euler_poses, 'XYZ').reshape(-1, 23, 3, 3)
    poses = np.concatenate((glbrots, poses), axis=1)
    return poses, trans


def read_debug_param_values_from_json(file_path: str):
    r"""
    Read debug parameter values from a json file.

    :return: A dict for all debug parameters.
    """
    with open(file_path, 'r') as f:
        result = {param['name']: param['value'] for param in json.load(f)}
    return result


def calc_model_parameters(model: nn.Module):
    params = model.parameters()
    total = 0
    for item in params:
        print(item.shape, np.prod(item.shape))
        total += np.prod(item.shape)
    print(f'### total: {total // 1000} k')
    
    
def stat_amass():
    ''' Count data length in each AMASS subdataset.
    '''
    res = {}
    total_cnt = 0
    for dataset_name in General.amass_data:
        print(f'### Loading {dataset_name} ...')
        cnt = 0
        data_paths = glob(f'{Paths.amass_dir}/{dataset_name}/data_*.pt')
        for data_path in data_paths:
            data = torch.load(data_path)
            cnt += np.sum([item.shape[0] for item in data['pose']])
        res[dataset_name] = cnt
        total_cnt += cnt
    for name in General.amass_data:
        print(f'{name} length: {res[name]} ({100*res[name]/total_cnt:.3f} %)')
    print(f'AMASS total length: {total_cnt}')


def sample_data(dir:str, data_scale:float) -> Dict[int, np.ndarray]:
    ''' Determin how to sample the data, given an augmented data dir, and the data scale.
    args:
        dir: str, dir of the augmented dataset.
        data_scale: float, between 0 and 1, how much data to sample.
    returns:
        Dict[int, np.ndarray]: data_id -> idxs.
    '''
    total_cnt = 0
    cnt_dict: Dict[int, int] = dict()   # data_id -> cnt
    data_paths = glob(f'{dir}/data_*_aug_0.pt')
    for data_path in data_paths:
        search_res = re.search(r'data_(\d+)_', data_path)
        assert search_res is not None
        data_id = int(search_res.group(1))
        data = torch.load(data_path)
        cnt = len(data['tran'])
        cnt_dict[data_id] = cnt
        total_cnt += cnt
    idxs = np.sort(np.random.choice(total_cnt, size=int(data_scale*total_cnt), replace=False))
    res = dict()
    base = 0
    for data_id, cnt in sorted(cnt_dict.items(), key=lambda x:x[0]):
        res[data_id] = idxs[(base <= idxs) & (idxs < base+cnt)] - base
        base += cnt
    return res
        
        
if __name__ == '__main__':
    pass

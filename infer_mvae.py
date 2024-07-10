import torch
import torch.nn as nn
import torch.utils.data

import articulate as art
from articulate.vae import *
from config import Paths, JointSet, TrainMVAE
from train_mvae import get_mvae_frames


BODY_MODEL = art.ParametricModel(Paths.smpl_file)
  
    
def infer_mvae(model: nn.Module, pose:torch.Tensor, tran:torch.Tensor,
    joint:torch.Tensor, n_aug:int=10, n_sample:int=10):
    ''' Wrapper for MVAE input and output.
    return:
        MVAE model output.
    '''
    frames = get_mvae_frames(pose, tran, joint).to(TrainMVAE.device)
    N = frames.shape[0]
    
    # get value restrictions
    pos_range = 0.15     # must > 0
    vel_bias = 0.2       # must > 0
    vel_range = 2.0      # must > 1
    root_pos_t = frames[:, :3]
    joint_pos_t = frames[:, 12:69]
    root_vel_t = frames[:, 3:6]
    joint_vel_t = frames[:, 69:126]
    
    # calculate pos and vel lower/upper bounds
    root_pos_lb, root_pos_ub = root_pos_t - pos_range, root_pos_t + pos_range
    joint_pos_lb, joint_pos_ub = joint_pos_t - pos_range, joint_pos_t + pos_range
    root_vel_lb, root_vel_ub = root_vel_t * (1/vel_range), root_vel_t * vel_range
    mask = root_vel_t < 0
    root_vel_lb[mask], root_vel_ub[mask] = root_vel_ub[mask], root_vel_lb[mask]
    root_vel_lb -= vel_bias; root_vel_ub += vel_bias
    joint_vel_lb, joint_vel_ub = joint_vel_t * (1/vel_range), joint_vel_t * vel_range
    mask = joint_vel_t < 0
    joint_vel_lb[mask], joint_vel_ub[mask] = joint_vel_ub[mask], joint_vel_lb[mask]
    joint_vel_lb -= vel_bias; joint_vel_ub += vel_bias
    
    # generate new pose from mu and logvar
    output = torch.empty(n_aug, *frames.shape).to(TrainMVAE.device)
    output[:, 0, :] = frames[None, 0, :]
    with torch.no_grad():
        for i in range(N-1):        # iterate over timestamps
            x = frames[i+1:i+2, :].repeat(n_aug * n_sample, 1)
            c = output[:, i, :].repeat(n_sample, 1)
            out = model(x, c)[0]
            
            # restrict positions
            root_pos = out[:, :3]
            root_pos[:] = root_pos.clip(min=root_pos_lb[i+1:i+2,:], max=root_pos_ub[i+1:i+2,:])
            joint_pos = out[:, 12:69]
            joint_pos[:] = joint_pos.clip(min=joint_pos_lb[i+1:i+2,:], max=joint_pos_ub[i+1:i+2,:])
            # restrict velocities
            root_vel = out[:, 3:6]
            root_vel[:] = root_vel.clip(min=root_vel_lb[i+1:i+2,:], max=root_vel_ub[i+1:i+2,:])
            joint_vel = out[:, 69:126]
            joint_vel[:] = joint_vel.clip(min=joint_vel_lb[i+1:i+2,:], max=joint_vel_ub[i+1:i+2,:])            
            # normalize tensor
            root_rot = out[:, 6:12]
            root_rot[:] = art.math.refine_r6d(root_rot)
            joint_rot = out[:, 126:240].view(-1, 19, 6)
            joint_rot[:] = art.math.refine_r6d(joint_rot.contiguous()).view(-1, 19, 6)
            
            out = out.view(n_sample, n_aug, -1)
            min_idx = torch.argmin(torch.sum(torch.square(out[:, :, -JointSet.n_aug*6:]
                - frames[None, i+1:i+2, -JointSet.n_aug*6:]), dim=2), dim=0)
            output[:, i+1, :] = out[min_idx, range(n_aug), :]
    output = output.to(torch.device('cpu'))

    # reconstruct pose_aug from output
    pose_aug = torch.cat([torch.zeros(N,1,3), pose[:,1:,:]], dim=1)
    pose_aug = BODY_MODEL.forward_kinematics_R(art.math.axis_angle_to_rotation_matrix(pose_aug).view(-1,24,3,3))
    pose_aug = pose_aug[None,...].repeat(n_aug,1,1,1,1)
    pose_aug[:,:,JointSet.aug,:,:] = art.math.r6d_to_rotation_matrix(
        output[:, :, -JointSet.n_aug*6:].contiguous()).view(n_aug, N, JointSet.n_aug, 3, 3)
    pose_aug = BODY_MODEL.inverse_kinematics_R(pose_aug.view(-1,24,3,3)).view(n_aug,N,24,3,3)
    pose_aug = art.math.rotation_matrix_to_axis_angle(pose_aug).view(n_aug,N,24,3)
    pose_aug[:,:,0,:] = pose[None,:,0,:]
    return pose_aug 

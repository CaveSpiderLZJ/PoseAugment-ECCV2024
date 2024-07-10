''' Train PoseNet.
'''

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Callable

import utils
from config import General, Paths, JointSet, TrainTransPose
import articulate as art
from articulate.rnn import RNNDataset, RNNLossWrapper
from transpose_net import TransPoseNet
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(Paths.smpl_file)


def train(net, train_dataloader, vald_dataloader=None, save_dir='weights', loss_fn=torch.nn.MSELoss(),
          eval_fn=None, optimizer=None, n_epoch=500, lr=TrainTransPose.lr1, n_val_steps=-1,
          clip_grad_norm=0.0, save_log=True, eval_metric_names=None):
    r"""
    Train a net.

    Notes
    -----
    - The current weights, best weights, train info, optimizer states, and tensorboard logs
      (if `save_log` is True) will be saved into `save_dir` at the end of each validation.
    - When `vald_dataloader` is None, there is no validation and the best weight will be automatically updated
      whenever train loss decreases. Otherwise, it will be updated when validation loss decreases.
    - `*_dataloader` args are used as `for i, (data, label) in enumerate(dataloader)` and `len(dataloader)`.

    Args
    -----
    :param net: Network to train.
    :param train_dataloader: Train dataloader, enumerable and has __len__. It loads (train_data, train_label) pairs.
    :param vald_dataloader: Validation dataloader, enumerable and has __len__. It loads (vald_data, vald_label) pairs.
    :param save_dir: Directory for the saved model, weights, best weights, train information, etc.
    :param loss_fn: Loss function. Call like loss_fn(model(data), label). It should return one-element loss tensor.
    :param eval_fn: Eval function for validation. If None, use loss_fn for validation. It should return a tensor.
        The very first element is used as the major validation loss (to save the weights).
    :param optimizer: Optimizer. If None, Adam is used by default and optimize net.parameters().
    :param n_epoch: Total number of training epochs. One epoch loads the entire dataset once.
    :param n_val_steps: Number of training iterations between two consecutive validations.
        If negative, validations will be done once every epoch.
    :param clip_grad_norm: 0 for no clip. Otherwise, clip the 2-norm of the gradient of net.parameters().
    :param save_log: If True, train-validation-loss curves will be plotted using tensorboard (saved in save_dir/log).
    :param eval_metric_names: A list of strings. Names of the returned values of eval_fn (used in tensorboard).
    """
    if optimizer is None: optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if eval_fn is None: eval_fn = loss_fn
    if eval_fn == loss_fn and eval_metric_names is None: eval_metric_names = ['loss']
    if not os.path.isdir(save_dir): os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, 'best.pt')
    min_val_loss = 1e10
    n_iter_per_eopch = len(train_dataloader)
    print(f'### num iter per epoch: {n_iter_per_eopch}')
    n_train_steps = n_epoch * n_iter_per_eopch
    # number of training steps to perform validation, at least validate once every epoch
    n_val_steps = n_val_steps if n_val_steps > 0 else n_iter_per_eopch
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        start_factor=1.0, end_factor=0.1, total_iters=n_train_steps)
    writter = SummaryWriter(save_dir) if save_log else None
    net.train()

    total_it = 0
    tic = time.perf_counter()
    for epoch in range(n_epoch):
        train_loss = []
        for i, (d, l) in enumerate(train_dataloader):
            loss = loss_fn(net(d)[0], l)
            loss.backward()
            if clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            total_it += 1

            if total_it % n_val_steps == 0:  # validation
                net.eval()
                with torch.no_grad():
                    val_loss = np.mean([eval_fn(net(d)[0], l).cpu().item() for d, l in vald_dataloader])
                timestamp = time.perf_counter() - tic
                print(f'### timestamp: {timestamp:.3f}, epoch: {epoch}/{n_epoch}, iter: {i+1}/{n_iter_per_eopch}, '
                    + f'train_loss: {np.mean(train_loss):.4e}, val_loss: {val_loss:.4e}')

                if save_log:
                    writter.add_scalar('Train Loss', np.mean(train_loss), total_it)
                    writter.add_scalar('Val Loss', val_loss, total_it)
                if total_it >= (n_train_steps // 2) and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(net.state_dict(), best_model_path)
                    print('### best model is saved.')
                net.train()

    if save_log: writter.close()
        
        
def get_train_dataset(get_data_in_out:Callable, sampled_data:dict, aug_scale:int=1, aug_type:str='vae', params:dict=None):          
    all_data, all_label = [], []          
    for dir, id2idx in sampled_data.items():
        for data_id, idxs in id2idx.items():
            if aug_type == 'vae':
                data_paths = [f'{dir}/data_{data_id}_aug_{aug_id}.pt' for aug_id in range(aug_scale)]
            else: data_paths = [f'{dir}/data_{data_id}_aug_0.pt']
            for data_path in data_paths:
                data = torch.load(data_path)
                acc, rot, joint, pose = (data[key] for key in ('vacc', 'vrot', 'joint', 'pose'))
                for idx in idxs:
                    data_in, data_out = get_data_in_out(pose[idx], joint[idx], acc[idx], rot[idx])
                    all_data.append(data_in)
                    all_label.append(data_out)
                    if aug_type == 'jitter':
                        for _ in range(aug_scale - 1):                        
                            all_data.append(art.math.jitter(data_in, std=params['jitter_std']))
                            all_label.append(data_out)
                    elif aug_type == 'mag_warp':
                        for _ in range(aug_scale - 1):
                            augmented = art.math.magnitude_warp(data_in, dim=0,
                                n_knots=params['mag_warp_n'], std=params['mag_warp_std'])
                            all_data.append(augmented)
                            all_label.append(data_out)
                    elif aug_type == 'time_warp':
                        for _ in range(aug_scale - 1):
                            time_warp_params = art.math.time_warp_params(
                                n_knots=params['time_warp_n'], std=params['time_warp_std'])
                            all_data.append(art.math.time_warp(data_in, dim=0, params=time_warp_params))
                            all_label.append(art.math.time_warp(data_out, dim=0, params=time_warp_params))
                    elif aug_type == 'combined':
                        for _ in range(aug_scale - 1):
                            time_warp_params = art.math.time_warp_params(
                                n_knots=params['time_warp_n'], std=params['time_warp_std'])
                            augmented = art.math.magnitude_warp(data_in, dim=0,
                                n_knots=params['mag_warp_n'], std=params['mag_warp_std'])
                            augmented = art.math.time_warp(augmented, dim=0, params=time_warp_params)
                            augmented = art.math.jitter(augmented, std=params['jitter_std'])
                            all_data.append(augmented)
                            all_label.append(art.math.time_warp(data_out, dim=0, params=time_warp_params))
                    else: assert aug_type == 'vae'
    return RNNDataset(all_data, all_label, device=device)


def get_test_dataset(get_data_in_out:Callable):
    data = torch.load(f'{Paths.dipimu_dir}/test.pt')
    pose_all, joint_all, acc_all, rot_all = (data[key] for key in ('pose', 'joint', 'vacc', 'vrot'))
    all_data, all_label = [], []
    for pose, joint, acc, rot in zip(pose_all, joint_all, acc_all, rot_all):
        data_in, data_out = get_data_in_out(pose, joint, acc, rot)
        all_data.append(data_in)
        all_label.append(data_out)
    return RNNDataset(all_data, all_label, device=device)


def train_pose_s1(model_name:str, sampled_data:dict, aug_scale:int=1,
    aug_type:str='vae', params:dict=None, lr=TrainTransPose.lr1, base_epoch:int=TrainTransPose.base_epoch):
    
    
    def get_data_in_out(pose, joint, acc, rot):
        root_rot = art.math.axis_angle_to_rotation_matrix(pose[:,0])
        leaf_joint = (joint[:, JointSet.leaf] - joint[:, :1]).bmm(root_rot).flatten(1)
        imu = utils.normalize_and_concat(acc, rot, TrainTransPose.acc_scale)              
        data_in, data_out = imu[1:-1], leaf_joint[1:-1]
        return data_in, data_out
    

    print(f'### Training Pose-S1 ...')
    save_dir = f'{Paths.transpose_dir}/{model_name}/pose_s1'
    net = TransPoseNet(is_train=True).pose_s1.to(device)
    
    # get dataloader
    train_dataset = get_train_dataset(get_data_in_out, sampled_data, aug_scale, aug_type, params)
    train_dataloader = DataLoader(train_dataset,
        batch_size=TrainTransPose.batch_size1, shuffle=True, collate_fn=RNNDataset.collate_fn)
    test_dataloader = DataLoader(get_test_dataset(get_data_in_out),
        batch_size=TrainTransPose.test_batch_size, collate_fn=RNNDataset.collate_fn)
        
    # loss function
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_dist_eval_fn = RNNLossWrapper(art.PositionErrorEvaluator())
    
    n_epoch = int(base_epoch / aug_scale)
    train(net, train_dataloader, test_dataloader, save_dir, loss_fn=rnn_mse_loss_fn,
        eval_fn=rnn_dist_eval_fn, n_epoch=n_epoch, lr=lr, n_val_steps=TrainTransPose.n_val_steps1,
        clip_grad_norm=1, eval_metric_names=['5 joint error (m)'])
    
    with torch.no_grad():
        net.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt')))
        net.eval()
        err = [rnn_dist_eval_fn(net(d)[0], l).item() for d, l in test_dataloader]
        err = sum(err)/len(err)
        print(f'[DIP-IMU] Pose-S1 5 joint error: {err:.6f} m')
        # save training result
        res_dict = {'model_name': model_name, 'module': 'Pose-S1', 'err': err}
        json.dump(res_dict, open(f'{save_dir}/result.json', 'w'), indent=4)
        
        
def train_pose_s2(model_name:str, sampled_data:dict, aug_scale:int=1,
    aug_type:str='vae', params:dict=None, lr=TrainTransPose.lr2, base_epoch:int=TrainTransPose.base_epoch):
    
    
    def get_data_in_out(pose, joint, acc, rot):
        root_rot = art.math.axis_angle_to_rotation_matrix(pose[:,0])
        all_joint = (joint - joint[:, :1]).bmm(root_rot)
        leaf_joint = all_joint[:, JointSet.leaf, :].flatten(1)
        # add Gaussian noise
        leaf_joint = art.math.jitter(leaf_joint, std=0.04)
        full_joint = all_joint[:, JointSet.full, :].flatten(1)
        imu = utils.normalize_and_concat(acc, rot, TrainTransPose.acc_scale)
        data_in, data_out = torch.cat((leaf_joint[1:-1], imu[1:-1]), dim=1), full_joint[1:-1]
        return data_in, data_out
    
    
    print(f'### Training Pose-S2 ...')
    save_dir = f'{Paths.transpose_dir}/{model_name}/pose_s2'
    net = TransPoseNet(is_train=True).pose_s2.to(device)
    
    # get dataloader
    train_dataset = get_train_dataset(get_data_in_out, sampled_data, aug_scale, aug_type, params)
    train_dataloader = DataLoader(train_dataset,
        batch_size=TrainTransPose.batch_size2, shuffle=True, collate_fn=RNNDataset.collate_fn)
    test_dataloader = DataLoader(get_test_dataset(get_data_in_out),
        batch_size=TrainTransPose.test_batch_size, collate_fn=RNNDataset.collate_fn)
    
    # loss function
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_dist_eval_fn = RNNLossWrapper(art.PositionErrorEvaluator())
    
    n_epoch = int(base_epoch / aug_scale)
    train(net, train_dataloader, test_dataloader, save_dir, loss_fn=rnn_mse_loss_fn,
        eval_fn=rnn_dist_eval_fn, n_epoch=n_epoch, lr=lr, n_val_steps=TrainTransPose.n_val_steps2,
        clip_grad_norm=1, eval_metric_names=['23 joint error (m)'])
    
    with torch.no_grad():
        net.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt')))
        net.eval()
        err = [rnn_dist_eval_fn(net(d)[0], l).item() for d, l in test_dataloader]
        err = sum(err)/len(err)
        print(f'[DIP-IMU] Pose-S2 23 joint error: {err:.6f} m')
        # save training result
        res_dict = {'model_name': model_name, 'module': 'Pose-S2', 'err': err}
        json.dump(res_dict, open(f'{save_dir}/result.json', 'w'), indent=4)
        
        
def train_pose_s3(model_name:str, sampled_data:dict, aug_scale:int=1,
    aug_type:str='vae', params:dict=None, lr=TrainTransPose.lr3, base_epoch:int=TrainTransPose.base_epoch):
    
    
    def get_data_in_out(pose, joint, acc, rot):
        p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        full_joint = (joint[:, JointSet.full] - joint[:, :1]).bmm(p[:, 0]).flatten(1)
        # add Gaussian noise
        full_joint = art.math.jitter(full_joint, std=0.025)
        imu = utils.normalize_and_concat(acc, rot, TrainTransPose.acc_scale)
        p[:, 0] = torch.eye(3)
        global_p = body_model.forward_kinematics_R(p)
        p6d = art.math.rotation_matrix_to_r6d(global_p[:, JointSet.reduced]).view(-1, JointSet.n_reduced * 6)
        data_in, data_out = torch.cat((full_joint[1:-1], imu[1:-1]), dim=1), p6d[1:-1]
        return data_in, data_out

    
    print(f'### Training Pose-S3 ...')
    save_dir = f'{Paths.transpose_dir}/{model_name}/pose_s3'
    net = TransPoseNet(is_train=True).pose_s3.to(device)
    
    # get dataloader
    train_dataset = get_train_dataset(get_data_in_out, sampled_data, aug_scale, aug_type, params)
    train_dataloader = DataLoader(train_dataset,
        batch_size=TrainTransPose.batch_size3, shuffle=True, collate_fn=RNNDataset.collate_fn)
    test_dataloader = DataLoader(get_test_dataset(get_data_in_out),
        batch_size=TrainTransPose.test_batch_size, collate_fn=RNNDataset.collate_fn)
    
    # loss function
    rnn_mse_loss_fn = RNNLossWrapper(torch.nn.MSELoss())
    rnn_rot_eval_fn = RNNLossWrapper(art.RotationErrorEvaluator(art.math.RotationRepresentation.R6D))
    
    n_epoch = int(base_epoch / aug_scale)
    train(net, train_dataloader, test_dataloader, save_dir, loss_fn=rnn_mse_loss_fn,
        eval_fn=rnn_rot_eval_fn, n_epoch=n_epoch, lr=lr, n_val_steps=TrainTransPose.n_val_steps3,
        clip_grad_norm=1, eval_metric_names=['global rotation error (deg)'])
    
    with torch.no_grad():
        net.load_state_dict(torch.load(os.path.join(save_dir, 'best.pt')))
        net.eval()
        err = [rnn_rot_eval_fn(net(d)[0], l).item() for d, l in test_dataloader]
        err = sum(err)/len(err)
        print(f'[DIP-IMU] Pose-S3 global rotation error: {err:.4f} deg')
        # save training result
        res_dict = {'model_name': model_name, 'module': 'Pose-S3', 'err': err}
        json.dump(res_dict, open(f'{save_dir}/result.json', 'w'), indent=4)
            
            
def test_mdm():
     # baseline params 
    params = {'jitter_std': 0.002, 'mag_warp_n': 6, 'mag_warp_std': 0.001,
        'time_warp_n': 6, 'time_warp_std': 0.001}    
    
    # NOTE: you can customize the test_id
    test_id = 0
    
    # NOTE: you can customize the augmented datasets
    for dataset_name in ['mdm', 'mdm_PoseAugment']:
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.mdm_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
    
        base_epoch = int(1000 / General.mdm_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
    
    # test Jitter
    if True:    # NOTE: you can turn it off
        dataset_name = 'mdm'
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.mdm_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
        
        base_epoch = int(1000 / General.mdm_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_jitter_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            
            
def test_mdm_m2m():
     # baseline params 
    params = {'jitter_std': 0.002, 'mag_warp_n': 6, 'mag_warp_std': 0.001,
        'time_warp_n': 6, 'time_warp_std': 0.001}    
    
    # NOTE: you can customize the test_id
    test_id = 1
    
    # NOTE: you can customize the augmented datasets
    for dataset_name in ['mdm_m2m', 'mdm_m2m_PoseAugment']:
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.mdm_m2m_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
    
        base_epoch = int(1000 / General.mdm_m2m_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
    
    # test Jitter
    if True:    # NOTE: you can turn it off
        dataset_name = 'mdm_m2m'
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.mdm_m2m_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
        
        base_epoch = int(1000 / General.mdm_m2m_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_jitter_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            
            
def test_actor():
     # baseline params 
    params = {'jitter_std': 0.002, 'mag_warp_n': 6, 'mag_warp_std': 0.001,
        'time_warp_n': 6, 'time_warp_std': 0.001}    
    
    # NOTE: you can customize the test_id
    test_id = 2
    
    # NOTE: you can customize the augmented datasets
    for dataset_name in ['actor', 'actor_PoseAugment']:
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.actor_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
    
        base_epoch = int(1000 / General.actor_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            
    # test Jitter
    if True:    # NOTE: you can turn it off
        dataset_name = 'actor'
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.actor_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
        
        base_epoch = int(1000 / General.actor_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_jitter_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            

def test_motionaug():
     # baseline params 
    params = {'jitter_std': 0.002, 'mag_warp_n': 6, 'mag_warp_std': 0.001,
        'time_warp_n': 6, 'time_warp_std': 0.001}    
    
    # NOTE: you can customize the test_id
    test_id = 3
    
    # NOTE: you can customize the augmented datasets
    for dataset_name in ['motionaug', 'motionaug_PoseAugment']:
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.motionaug_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
    
        base_epoch = int(1000 / General.motionaug_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'vae', params, base_epoch=base_epoch)
            
    # test Jitter
    if True:    # you can turn it off
        dataset_name = 'motionaug'
        # sample training data
        utils.init_rand_seed(TrainTransPose.seed)
        sampled_data = dict()
        dir = f'{Paths.motionaug_aug_dir}/{dataset_name}'
        sampled_data[dir] = utils.sample_data(dir, data_scale=1.0)
        
        base_epoch = int(1000 / General.motionaug_scale)
        for aug_scale in range(1, 6):
            model_name = f'{test_id}_{dataset_name}_jitter_aug{aug_scale}_epoch{base_epoch}'
            utils.init_rand_seed(TrainTransPose.seed)
            train_pose_s1(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s2(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)
            train_pose_s3(model_name, sampled_data, aug_scale, 'jitter', params, base_epoch=base_epoch)


if __name__ == '__main__':
    utils.init_rand_seed(TrainTransPose.seed)
    # NOTE: choose one to run
    test_mdm()
    # test_mdm_m2m()
    # test_actor()
    # test_motionaug()

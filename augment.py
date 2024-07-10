''' Deal with pose data augmentation.
'''

import os
import re
import tqdm
import pickle
import torch
import torch.nn as nn
import numpy as np
from glob import glob

import utils
import articulate as art
from articulate.vae import *
from config import General, Paths, TrainMVAE
from infer_mvae import infer_mvae
from dynamics import VAEOptimizer


body_model = art.ParametricModel(Paths.smpl_file)
    

def augment_data(model: nn.Module, data_path: str, save_dir: str):
    ''' Use VAE model to augment data, and save to proper locations.
    '''
    # preparations
    data_name = data_path.split('/')[-1]    # data_0.pt
    data_name = data_name[:-3]  # data_0
    data = torch.load(data_path)
    
    # augment pose data
    n_aug = 4
    n_sample = 2    # 2, ablation
    n_split = 200  
    n_min_length = 50
    optimizer = VAEOptimizer()
    pose_all, shape_all, tran_all, joint_all, vrot_all, vacc_all = \
        [data[key] for key in ('pose', 'shape', 'tran', 'joint', 'vrot', 'vacc')]
    pose_aug_all = {'pose': [], 'tran': [], 'joint': [], 'vrot': [], 'vacc': []}
    
    for pose, tran, joint in tqdm.tqdm(list(zip(pose_all, tran_all, joint_all))):
        for p, t, j in zip(pose.split(n_split), tran.split(n_split), joint.split(n_split)):
            N = p.shape[0]
            if N < n_min_length: continue
            augmented = {'pose': [], 'tran': [], 'joint': [], 'vrot': [], 'vacc': []}
                        
            # save original data
            rot_t, joint_t, vert_t = body_model.forward_kinematics(
                art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), tran=t, calc_mesh=True)
            augmented['pose'].append(p)     # [N, 24, 3]
            augmented['tran'].append(t)     # [N, 3]
            augmented['joint'].append(j)    # [N, 24, 3]
            augmented['vrot'].append(rot_t[:, General.ji_mask])
            augmented['vacc'].append(utils.syn_acc2(vert_t[:, General.vi_mask], smooth_n=4))
            
            # augment data by MVAE with sample-based control
            pose_aug = infer_mvae(model, p, t, j, n_aug, n_sample)
            rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                art.math.axis_angle_to_rotation_matrix(pose_aug).view(-1,24,3,3),
                tran=t.repeat(n_aug,1), calc_mesh=True)           
            rot_aug = rot_aug.view(n_aug, N, 24, 3, 3)
            joint_aug = joint_aug.view(n_aug, N, 24, 3)
            vert_aug = vert_aug.view(n_aug, N, -1, 3)
            
            # optimize pose_aug by VAEOptimizer
            if True:    # NOTE: you can set it to False to disable physical optimization
                pose_aug_mat = art.math.axis_angle_to_rotation_matrix(
                    pose_aug).view(n_aug,-1,24,3,3)
                joint_vel = torch.stack([utils.syn_vel(item, smooth_n=2) for item in joint_aug])
                for i in range(n_aug):
                    optimizer.reset_states()
                    pose_opt = []
                    for p_, v_ in zip(pose_aug_mat[i], joint_vel[i]):
                        p_opt, _ = optimizer.optimize_frame_sparse(p_, v_)
                        pose_opt.append(p_opt)
                    pose_opt = torch.stack(pose_opt)
                    pose_aug_mat[i] = pose_opt
                rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                    pose_aug_mat.view(-1, 24, 3, 3), tran=t.repeat(n_aug,1), calc_mesh=True)
                pose_aug = art.math.rotation_matrix_to_axis_angle(pose_aug_mat).view(n_aug, -1, 24, 3)
                rot_aug = rot_aug.view(n_aug, N, 24, 3, 3)
                joint_aug = joint_aug.view(n_aug, N, 24, 3)
                vert_aug = vert_aug.view(n_aug, N, -1, 3)
            
            augmented['pose'].extend(pose_aug)
            augmented['tran'].extend([t] * n_aug)
            augmented['joint'].extend(joint_aug)
            augmented['vrot'].extend(rot_aug[:, :, General.ji_mask])
            augmented['vacc'].extend([utils.syn_acc2(v[:, General.vi_mask], smooth_n=4) for v in vert_aug])
                
            # save augmented
            for key in pose_aug_all:
                pose_aug_all[key].append(augmented[key])
            
    # save data
    for idx in range(n_aug + 1):
        out = {key: [] for key in pose_aug_all}
        for key, item_list_all in pose_aug_all.items():
            for item_list in item_list_all:
                out[key].append(item_list[idx].clone())
        torch.save(out, f'{save_dir}/{data_name}_aug_{idx}.pt')
        
        
def augment_amass():
    model = MVAE().to(TrainMVAE.device)
    model_path = f'{Paths.model_dir}/mvae_weights.pt'
    model.load_state_dict(torch.load(model_path, map_location=TrainMVAE.device))
    model.eval()
    
    # augment AMASS data
    for dataset_name in General.amass:
        data_paths = glob(f'{Paths.amass_dir}/{dataset_name}/data_*.pt')
        save_dir = f'{Paths.amass_aug_dir}/{dataset_name}'
        os.makedirs(save_dir, exist_ok=True)
        for data_path in data_paths:
            print(f'Augmenting {data_path} ...')
            try:
                augment_data(model, data_path, save_dir)
            except KeyboardInterrupt as e:
                print(e)
                exit()
            except Exception as e:
                print(e)
                continue


def preprocess_mdm():
    ''' Before calling this method, you should have humanml test data
            generated by MDM-T2M and MDM-M2M, with repetition == 5.
        Note that MDM-M2M is a motion-to-motion variant of the original MDM,
            please refer to the paper for more details.
        The generated data should be placed at 'data/dataset_work/mdm' or 'data/dataset_work/mdm_m2m'.
        This function aligns the format of data generated by MDM with data generated by PoseAugment.
        The preprocessed data will be placed at 'data/dataset_aug/mdm/mdm' or `data/dataset_aug/mdm_m2m/mdm_m2m`. 
    '''
    # output augmented data, fps = 60
    # pose: list of [200, 24, 3]
    # tran: list of [200, 3]
    # joint: list of [200, 24, 3]
    # vrot: list of [200, 6, 3, 3]
    # vacc: list of [200, 6, 3]
    
    # NOTE: set mode to t2m or m2m to preprocess MDM-T2M or MDM-M2M data
    mode = 't2m' # in {t2m, m2m}
    assert mode in {'t2m', 'm2m'}
    
    src_fps, target_fps = 20, 60
    len_valid = 120     # valid pose length of MDM data
    len_out = 200       # output window length
    
    if mode == 't2m': save_dir = f'{Paths.mdm_aug_dir}/mdm'
    else: save_dir = f'{Paths.mdm_m2m_aug_dir}/mdm_m2m'
    os.makedirs(save_dir, exist_ok=True)
    
    if mode == 't2m':
        paths = glob(f'{Paths.mdm_dir}/results_*.pkl')
    else: paths = glob(f'{Paths.mdm_m2m_dir}/results_*.pkl')
    paths = list(sorted(paths, key=lambda x:int(re.search(r'\d+',x.split('/')[-1])[0])))

    for path in tqdm.tqdm(paths):
        data_idx = int(re.search(r'\d+', path.split('/')[-1])[0])
        data = pickle.load(open(path, 'rb'))
        n_sample, n_repetition = data['num_samples'], data['num_repetitions']
        assert n_repetition == 5
        trans, poses = data['tran'], data['pose']
        
        for rep in range(n_repetition):
            out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
            for i in range(rep * n_sample, (rep+1) * n_sample):
                tran, pose = trans[i], poses[i]
                # upsampling
                tran = torch.from_numpy(art.math.resample(tran[:len_valid], axis=0, ratio=(target_fps/src_fps))).to(torch.float32)
                pose = torch.from_numpy(art.math.resample(pose[:len_valid], axis=0, ratio=(target_fps/src_fps))).to(torch.float32)
                pose_mat = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                # forward kinematics
                rot, joint, vert = body_model.forward_kinematics(pose=pose_mat, tran=tran, calc_mesh=True)
                vrot = rot[:, General.ji_mask]
                vacc = utils.syn_acc2(vert[:, General.vi_mask], smooth_n=1)
                # check nan
                if torch.isnan(pose).any() or torch.isnan(tran).any() or torch.isnan(joint).any() \
                    or torch.isnan(vrot).any() or torch.isnan(vacc).any():
                    print(f'nan in {path}, {i}')
                    pose = torch.zeros_like(pose)
                    tran = torch.zeros_like(tran)
                    joint = torch.zeros_like(joint)
                    vrot = torch.zeros_like(vrot)
                    vacc = torch.zeros_like(vacc)
                
                out_pose.extend([pose[:len_out], pose[-len_out:]])
                out_tran.extend([tran[:len_out], tran[-len_out:]])
                out_joint.extend([joint[:len_out], joint[-len_out:]])
                out_vrot.extend([vrot[:len_out], vrot[-len_out:]])
                out_vacc.extend([vacc[:len_out], vacc[-len_out:]])
                
            save_data = {'pose': out_pose, 'tran': out_tran,
                'joint': out_joint, 'vrot': out_vrot, 'vacc': out_vacc}
            save_path = f'{save_dir}/data_{data_idx}_aug_{rep}.pt'
            torch.save(save_data, save_path)
            
            
def augment_mdm():
    ''' Use one repetition of MDM-T2M or MDM-M2M data, augment them using PoseAugment.
        The augmented data will be placed at 'data/dataset_aug/mdm/mdm_PoseAugment'
            or 'data/dataset_aug/mdm_m2m/mdm_m2m_PoseAugment'.
    '''
    # NOTE: set mode to t2m or m2m to augment MDM-T2M or MDM-M2M data
    mode = 't2m' # in {t2m, m2m}
    assert mode in {'t2m', 'm2m'}
    
    model = MVAE().to(TrainMVAE.device)
    model_path = f'{Paths.model_dir}/mvae_weights.pt'
    model.load_state_dict(torch.load(model_path, map_location=TrainMVAE.device))
    model.eval()
    optimizer = VAEOptimizer()
    
    src_fps, target_fps = 20, 60
    len_valid = 120     # valid pose length of MDM data
    len_out = 200       # output window length
    n_aug = 4
    
    if mode == 't2m':
        save_dir = f'{Paths.mdm_aug_dir}/mdm_PoseAugment'
    else: save_dir = f'{Paths.mdm_m2m_aug_dir}/mdm_m2m_PoseAugment'
    os.makedirs(save_dir, exist_ok=True)
    
    if mode == 't2m':
        paths = glob(f'{Paths.mdm_dir}/results_*.pkl')
    else: paths = glob(f'{Paths.mdm_m2m_dir}/results_*.pkl')
    paths = list(sorted(paths, key=lambda x:int(re.search(r'\d+',x.split('/')[-1])[0])))
    
    for path in paths:
        data_idx = int(re.search(r'\d+', path.split('/')[-1])[0])
        print(f'### Augmenting {path} ...')
        data = pickle.load(open(path, 'rb'))
        n_sample, n_repetition = data['num_samples'], data['num_repetitions']
        trans, poses = data['tran'], data['pose']
        
        out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
        for idx in tqdm.trange(n_sample):     # only use one repetition
            tran, pose = trans[idx], poses[idx]
            # upsampling
            tran2 = torch.from_numpy(art.math.resample(tran[:len_valid], axis=0, ratio=(target_fps/src_fps))).to(torch.float32)
            pose2 = torch.from_numpy(art.math.resample(pose[:len_valid], axis=0, ratio=(target_fps/src_fps))).to(torch.float32)
            
            # split
            for tran, pose in zip([tran2[:len_out], tran2[-len_out:]], [pose2[:len_out], pose2[-len_out:]]):
                pose_mat = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                # forward kinematics
                rot, joint, vert = body_model.forward_kinematics(pose=pose_mat, tran=tran, calc_mesh=True)
                
                # augment by PoseAugment
                # augment data by MVAE with sample-based control
                pose_aug = infer_mvae(model, pose, tran, joint, n_aug, n_sample=2)
                rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                    art.math.axis_angle_to_rotation_matrix(pose_aug).view(-1,24,3,3),
                    tran=tran.repeat(n_aug,1), calc_mesh=True)
                rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
                joint_aug = joint_aug.view(n_aug, -1, 24, 3)
                vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
                
                # optimize pose_aug by VAEOptimizer
                if True:    # NOTE: you can set it to False to disable physical optimization
                    pose_aug_mat = art.math.axis_angle_to_rotation_matrix(
                        pose_aug).view(n_aug,-1,24,3,3)
                    joint_vel = torch.stack([utils.syn_vel(item, smooth_n=2) for item in joint_aug])
                    for i in range(n_aug):
                        optimizer.reset_states()
                        pose_opt = []
                        try:
                            for p_, v_ in zip(pose_aug_mat[i], joint_vel[i]):
                                p_opt, _ = optimizer.optimize_frame_sparse(p_, v_)
                                pose_opt.append(p_opt)
                        except TypeError as e:  # encountered an error, ignore the optimization
                            print(e)
                            continue
                        else:   # optimization succeeded, replace the original data
                            pose_opt = torch.stack(pose_opt)
                            pose_aug_mat[i] = pose_opt
                    rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                        pose_aug_mat.view(-1, 24, 3, 3), tran=tran.repeat(n_aug,1), calc_mesh=True)
                    pose_aug = art.math.rotation_matrix_to_axis_angle(pose_aug_mat).view(n_aug, -1, 24, 3)
                    rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
                    joint_aug = joint_aug.view(n_aug, -1, 24, 3)
                    vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
                    
                # concat with the original data
                pose_aug = torch.concat([pose[None,...], pose_aug], dim=0)
                tran_aug = torch.stack([tran] * (n_aug+1))
                joint_aug = torch.concat([joint[None,...], joint_aug], dim=0)
                rot_aug = torch.concat([rot[None,...], rot_aug], dim=0)
                vert_aug = torch.concat([vert[None,...], vert_aug], dim=0)
                vrot_aug = rot_aug[:, :, General.ji_mask, :, :]
                vacc_aug = torch.stack([utils.syn_acc2(v[:, General.vi_mask], smooth_n=1) for v in vert_aug])
            
                # save data
                out_pose.append(pose_aug)
                out_tran.append(tran_aug)
                out_joint.append(joint_aug)
                out_vrot.append(vrot_aug)
                out_vacc.append(vacc_aug)

        # save data  
        for aug in range(n_aug + 1):      
            save_data = {'pose': [item[aug,...] for item in out_pose],
                'tran': [item[aug,...] for item in out_tran],
                'joint': [item[aug,...] for item in out_joint],
                'vrot': [item[aug,...] for item in out_vrot],
                'vacc': [item[aug,...] for item in out_vacc]}
            save_path = f'{save_dir}/data_{data_idx}_aug_{aug}.pt'
            torch.save(save_data, save_path)


def preprocess_actor():
    ''' Preprocess the pose data generated by ACTOR into the same data structure as PoseAugment.
        Before calling this method, you should have HumanAct12 data generated by ACTOR.
        The generated actor data should be placed at 'data/dataset_work/actor'.
        The preprocessed data will be placed at 'data/dataset_aug/actor/actor'.
    '''
    # load data
    path = f'{Paths.actor_dir}/generation_smpl.npy'
    data = np.load(path)    # [1000, 12, 24, 3, 70]

    # upsampling
    src_fps = 20
    target_fps = 60
    data = art.math.resample(data, axis=-1, ratio=target_fps/src_fps)
    data = data[:, :, :, :, :200].transpose(0, 1, 4, 2, 3)   # [1000, 12, 200, 24, 3]
    
    # reorganize data, 12 classes, aug 5
    n_aug = 5
    n_sample, n_class = data.shape[:2]
    n_sample_per_aug = n_sample // n_aug
    
    save_dir = f'{Paths.actor_aug_dir}/actor'
    os.makedirs(save_dir, exist_ok=True)
    
    # pose: list of [200, 24, 3]
    # tran: list of [200, 3]
    # joint: list of [200, 24, 3]
    # vrot: list of [200, 6, 3, 3]
    # vacc: list of [200, 6, 3]
    for class_idx in tqdm.trange(n_class):
        for aug_idx in range(n_aug):
            out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
            
            for sample_idx in range(n_sample_per_aug*aug_idx, n_sample_per_aug*(aug_idx+1)):
                pose = torch.from_numpy(data[sample_idx, class_idx]).to(torch.float32)
                rot, joint, vert = body_model.forward_kinematics(
                    pose=art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3), calc_mesh=True)
                
                out_pose.append(pose)
                out_tran.append(torch.zeros((pose.shape[0], 3), dtype=torch.float32))
                out_joint.append(joint)
                out_vrot.append(rot[:, General.ji_mask])
                out_vacc.append(utils.syn_acc2(vert[:, General.vi_mask], smooth_n=1))
            
            save_data = {'pose': out_pose, 'tran': out_tran,
                'joint': out_joint, 'vrot': out_vrot, 'vacc': out_vacc}
            save_path = f'{save_dir}/data_{class_idx}_aug_{aug_idx}.pt'
            torch.save(save_data, save_path)
            
            
def augment_actor():
    ''' Use 1/5 of actor data, augment them using PoseAugment.
        The augmented data will be placed at 'data/dataset_aug/actor/actor_PoseAugment'.
    '''
    # load model
    model = MVAE().to(TrainMVAE.device)
    model_path = f'{Paths.model_dir}/mvae_weights.pt'
    model.load_state_dict(torch.load(model_path, map_location=TrainMVAE.device))
    model.eval()
    optimizer = VAEOptimizer()
    
    # load data
    path = f'{Paths.actor_dir}/generation_smpl.npy'
    data = np.load(path)    # [1000, 12, 24, 3, 70]

    # upsampling
    src_fps = 20
    target_fps = 60
    data = art.math.resample(data, axis=-1, ratio=target_fps/src_fps)
    data = data[:, :, :, :, :200].transpose(0, 1, 4, 2, 3)   # [1000, 12, 200, 24, 3]
    
    # reorganize data, 12 classes, aug 5
    n_aug = 4
    n_sample, n_class = data.shape[:2]
    n_sample_per_aug = n_sample // (n_aug + 1)
    
    save_dir = f'{Paths.actor_aug_dir}/actor_PoseAugment'
    os.makedirs(save_dir, exist_ok=True)
    
    # pose: list of [200, 24, 3]
    # tran: list of [200, 3]
    # joint: list of [200, 24, 3]
    # vrot: list of [200, 6, 3, 3]
    # vacc: list of [200, 6, 3]
    for class_idx in range(n_class):
        print(f'### augmenting class {class_idx}')
        out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
        
        for sample_idx in tqdm.trange(n_sample_per_aug):    # only use 1/5 data
            pose = torch.from_numpy(data[sample_idx, class_idx]).to(torch.float32)
            rot, joint, vert = body_model.forward_kinematics(
                pose=art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3), calc_mesh=True)
            tran = torch.zeros((pose.shape[0], 3), dtype=torch.float32)
            
            # augment poses
            pose_aug = infer_mvae(model, pose, tran, joint, n_aug, n_sample=2)
            rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                art.math.axis_angle_to_rotation_matrix(pose_aug).view(-1,24,3,3),
                tran=tran.repeat(n_aug,1), calc_mesh=True)
            rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
            joint_aug = joint_aug.view(n_aug, -1, 24, 3)
            vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
            
            # optimize pose_aug by VAEOptimizer
            if True:    # NOTE: you can set it to False to disable physical optimization
                pose_aug_mat = art.math.axis_angle_to_rotation_matrix(
                    pose_aug).view(n_aug,-1,24,3,3)
                joint_vel = torch.stack([utils.syn_vel(item, smooth_n=2) for item in joint_aug])
                for i in range(n_aug):
                    optimizer.reset_states()
                    pose_opt = []
                    try:
                        for p_, v_ in zip(pose_aug_mat[i], joint_vel[i]):
                            p_opt, _ = optimizer.optimize_frame_sparse(p_, v_)
                            pose_opt.append(p_opt)
                    except TypeError as e:  # encountered an error, ignore the optimization
                        print(e)
                        continue
                    else:   # optimization succeeded, replace the original data
                        pose_opt = torch.stack(pose_opt)
                        pose_aug_mat[i] = pose_opt
                rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                    pose_aug_mat.view(-1, 24, 3, 3), tran=tran.repeat(n_aug,1), calc_mesh=True)
                pose_aug = art.math.rotation_matrix_to_axis_angle(pose_aug_mat).view(n_aug, -1, 24, 3)
                rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
                joint_aug = joint_aug.view(n_aug, -1, 24, 3)
                vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
                
            # concat with the original data
            pose_aug = torch.concat([pose[None,...], pose_aug], dim=0)
            tran_aug = torch.stack([tran] * (n_aug + 1))
            joint_aug = torch.concat([joint[None,...], joint_aug], dim=0)
            rot_aug = torch.concat([rot[None,...], rot_aug], dim=0)
            vert_aug = torch.concat([vert[None,...], vert_aug], dim=0)
            vrot_aug = rot_aug[:, :, General.ji_mask, :, :]
            vacc_aug = torch.stack([utils.syn_acc2(v[:, General.vi_mask], smooth_n=1) for v in vert_aug])
            
            # save data
            out_pose.append(pose_aug)
            out_tran.append(tran_aug)
            out_joint.append(joint_aug)
            out_vrot.append(vrot_aug)
            out_vacc.append(vacc_aug)
        
        # save data  
        for aug in range(n_aug + 1):
            save_data = {'pose': [item[aug,...] for item in out_pose],
                'tran': [item[aug,...] for item in out_tran],
                'joint': [item[aug,...] for item in out_joint],
                'vrot': [item[aug,...] for item in out_vrot],
                'vacc': [item[aug,...] for item in out_vacc]}
            save_path = f'{save_dir}/data_{class_idx}_aug_{aug}.pt'
            torch.save(save_data, save_path)
                
            
def preprocess_motionaug():
    ''' Preprocess the pose data generated by motionaug into the same data structure as PoseAugment.
        Before calling this method, you should download pose data generated by MotionAug from
            'https://github.com/meaten/MotionAug-CVPR2022', in the section 'Augmentations'
            (just download the 8 dataset_VAE_phys_{action_class}_offset_NN.npz datasets, no need to reproduce MotionAug).
        The downloaded .npz files should be placed at 'data/dataset_work/motionaug' before calling this function.
        The preprocessed data will be placed at 'data/dataset_aug/motionaug/motionaug'.
    '''
    src_fps = 120
    target_fps = 60
    n_window = 100  # in 60 FPS
    n_min_length = 50
    n_aug = 4
    save_dir = f'{Paths.motionaug_aug_dir}/motionaug'
    os.makedirs(save_dir, exist_ok=True)
    
    action_classes = ('kick', 'punch', 'walk', 'jog', 'sneak', 'grab', 'deposit', 'throw')
    for class_idx, action_class in enumerate(action_classes):
        data = np.load(f'{Paths.motionaug_dir}/dataset_VAE_phys_{action_class}_offset_NN.npz', allow_pickle=True)
        motions = data['motions']
        
        out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
        for motion in motions:
            if motion.shape[0] < n_min_length * (src_fps / target_fps): continue
            motion = art.math.resample(motion, axis=0, ratio=target_fps/src_fps)
            for motion_clip in torch.from_numpy(motion).float().split(n_window):
                if motion_clip.shape[0] < n_min_length: continue
                motion_clip = motion_clip.reshape(-1, 17, 3)
                tran = motion_clip[:, 0, :] * 0.05
                pose = torch.zeros((motion_clip.shape[0], 24, 3), dtype=torch.float32)
                for i in range(16): # MotionAug joint idxs
                    if i == 4:      # RightUpLeg
                        q = art.math.axis_angle_to_rotation_matrix(torch.tensor([0, 0, -25*np.pi/180]))
                        rot_mat = art.math.axis_angle_to_rotation_matrix(motion_clip[:,i+1,:])
                        pose[:, utils.idx_motionaug_to_smpl[i], :] = art.math.rotation_matrix_to_axis_angle(q @ rot_mat)
                    elif i == 10:   # LeftUpLeg
                        q = art.math.axis_angle_to_rotation_matrix(torch.tensor([0, 0, 25*np.pi/180]))
                        rot_mat = art.math.axis_angle_to_rotation_matrix(motion_clip[:,i+1,:])
                        pose[:, utils.idx_motionaug_to_smpl[i], :] = art.math.rotation_matrix_to_axis_angle(q @ rot_mat)
                    elif i in (9, 15, 6, 12):   # skip RightHand, LeftHand, RightFoot, LeftFoot
                        continue
                    else: pose[:, utils.idx_motionaug_to_smpl[i], :] = motion_clip[:, i+1, :]
                rot, joint, vert = body_model.forward_kinematics(
                    pose=art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3),
                    tran=tran, calc_mesh=True)
                out_pose.append(pose)
                out_tran.append(tran)
                out_joint.append(joint)
                out_vrot.append(rot[:, General.ji_mask])
                out_vacc.append(utils.syn_acc2(vert[:, General.vi_mask], smooth_n=1))
                
        # save data
        n_per_aug = len(out_pose) // (n_aug + 1)
        for aug_idx in range(n_aug + 1):
            save_data = {'pose': out_pose[aug_idx*n_per_aug:(aug_idx+1)*n_per_aug],
                'tran': out_tran[aug_idx*n_per_aug:(aug_idx+1)*n_per_aug],
                'joint': out_joint[aug_idx*n_per_aug:(aug_idx+1)*n_per_aug],
                'vrot': out_vrot[aug_idx*n_per_aug:(aug_idx+1)*n_per_aug],
                'vacc': out_vacc[aug_idx*n_per_aug:(aug_idx+1)*n_per_aug]}
            save_path = f'{save_dir}/data_{class_idx}_aug_{aug_idx}.pt'
            torch.save(save_data, save_path)
            print(f'### data saved to {save_path}')
            

def augment_motionaug():
    ''' Use 1/5 of MotionAug data, augment them using PoseAugment.
        The augmented data will be placed at 'data/dataset_aug/motionaug/motionaug_PoseAugment'.
    '''
    # # load model
    model = MVAE().to(TrainMVAE.device)
    model_path = f'{Paths.model_dir}/mvae_weights.pt'
    model.load_state_dict(torch.load(model_path, map_location=TrainMVAE.device))
    model.eval()
    optimizer = VAEOptimizer()
    
    # config
    n_aug = 4
    # directly used the preprocessed data, different with other methods.
    data_dir = f'{Paths.motionaug_aug_dir}/motionaug'
    assert os.path.exists(data_dir), 'Error: source data does not exist. Please preprocess motionaug data first'
    save_dir = f'{Paths.motionaug_aug_dir}/motionaug_PoseAugment'
    os.makedirs(save_dir, exist_ok=True)
    
    for class_idx in range(8):
        path = f'{data_dir}/data_{class_idx}_aug_0.pt'
        print(f'Augmenting class {class_idx} ...')
        out_pose, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], []
        
        data = torch.load(path)
        poses, trans = data['pose'], data['tran']
        for idx in tqdm.trange(len(poses)):
            pose, tran = poses[idx], trans[idx]        
            rot, joint, vert = body_model.forward_kinematics(
                pose=art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3),
                tran=tran, calc_mesh=True)
            
            # augment poses
            pose_aug = infer_mvae(model, pose, tran, joint, n_aug, n_sample=2)
            rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                art.math.axis_angle_to_rotation_matrix(pose_aug).view(-1,24,3,3),
                tran=tran.repeat(n_aug,1), calc_mesh=True)
            rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
            joint_aug = joint_aug.view(n_aug, -1, 24, 3)
            vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
            
            # optimize pose_aug by VAEOptimizer
            if True:    # NOTE: you can set it to False to disable physical optimization
                pose_aug_mat = art.math.axis_angle_to_rotation_matrix(
                    pose_aug).view(n_aug,-1,24,3,3)
                joint_vel = torch.stack([utils.syn_vel(item, smooth_n=2) for item in joint_aug])
                for i in range(n_aug):
                    optimizer.reset_states()
                    pose_opt = []
                    try:
                        for p_, v_ in zip(pose_aug_mat[i], joint_vel[i]):
                            p_opt, _ = optimizer.optimize_frame_sparse(p_, v_)
                            pose_opt.append(p_opt)
                    except Exception as e:  # encountered an error, ignore the optimization
                        print(e)
                        continue
                    else:   # optimization succeeded, replace the original data
                        pose_opt = torch.stack(pose_opt)
                        pose_aug_mat[i] = pose_opt
                rot_aug, joint_aug, vert_aug = body_model.forward_kinematics(
                    pose_aug_mat.view(-1, 24, 3, 3), tran=tran.repeat(n_aug,1), calc_mesh=True)
                pose_aug = art.math.rotation_matrix_to_axis_angle(pose_aug_mat).view(n_aug, -1, 24, 3)
                rot_aug = rot_aug.view(n_aug, -1, 24, 3, 3)
                joint_aug = joint_aug.view(n_aug, -1, 24, 3)
                vert_aug = vert_aug.view(n_aug, -1, 6890, 3)
                
            # concat with the original data
            pose_aug = torch.concat([pose[None,...], pose_aug], dim=0)
            tran_aug = torch.stack([tran] * (n_aug + 1))
            joint_aug = torch.concat([joint[None,...], joint_aug], dim=0)
            rot_aug = torch.concat([rot[None,...], rot_aug], dim=0)
            vert_aug = torch.concat([vert[None,...], vert_aug], dim=0)
            vrot_aug = rot_aug[:, :, General.ji_mask, :, :]
            vacc_aug = torch.stack([utils.syn_acc2(v[:, General.vi_mask], smooth_n=1) for v in vert_aug])
            
            # save data
            out_pose.append(pose_aug)
            out_tran.append(tran_aug)
            out_joint.append(joint_aug)
            out_vrot.append(vrot_aug)
            out_vacc.append(vacc_aug)
            
        # save data  
        for aug in range(n_aug + 1):
            save_data = {'pose': [item[aug,...] for item in out_pose],
                'tran': [item[aug,...] for item in out_tran],
                'joint': [item[aug,...] for item in out_joint],
                'vrot': [item[aug,...] for item in out_vrot],
                'vacc': [item[aug,...] for item in out_vacc]}
            save_path = f'{save_dir}/data_{class_idx}_aug_{aug}.pt'
            torch.save(save_data, save_path)
            

if __name__ == '__main__':
    utils.init_rand_seed(3407)
    # NOTE: select one to run
    # for each method, you should always run preprocess_xxx before augment_xxx
    augment_amass()   # demo of how to use PoseAugment
    # preprocess_mdm()  # reproduction of baseline methods, same for the following lines
    # augment_mdm()
    # preprocess_actor()
    # augment_actor()
    # preprocess_motionaug()
    # augment_motionaug()
    
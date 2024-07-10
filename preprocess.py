''' Preprocess DIP-IMU dataset.
    Synthesize AMASS dataset.
    Please refer to the `Paths` in `config.py` and set the path of each dataset correctly.
'''

import articulate as art
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob

import utils
from config import Paths, General


body_model = art.ParametricModel(Paths.smpl_file)


def process_amass():
    n_max = 50
    for ds_name in ['ACCAD', 'CMU', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje']:
        print('\r### Reading', ds_name)
        data_pose, data_trans, data_beta, length = [], [], [], []
        npz_paths = glob(f'{Paths.raw_amass_dir}/{ds_name}/{ds_name}/*/*_poses.npz')
        for npz_path in tqdm(npz_paths):
            try: cdata = np.load(npz_path)
            except: continue
            # cdata.keys(): 'trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses'
            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else:   # resample data
                try:
                    resampled_poses = art.math.resample(cdata['poses'], axis=0, ratio=60/framerate)
                    resampled_trans = art.math.resample(cdata['trans'], axis=0, ratio=60/framerate)
                except: continue
                data_pose.extend(resampled_poses.astype(np.float32))
                data_trans.extend(resampled_trans.astype(np.float32))
                data_beta.append(cdata['betas'][:10])
                length.append(resampled_poses.shape[0])
                continue
            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])
        
        if len(data_pose) == 0:
            print(f'AMASS dataset {ds_name} not found.')
            continue

        length = torch.tensor(length, dtype=torch.int)
        shape = torch.tensor(np.asarray(data_beta, np.float32))
        tran = torch.tensor(np.asarray(data_trans, np.float32))
        pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
        # in AMASS raw data set, 22 is the left hand index 1, 23 is the left hand index 2,
        # 37 is the right hand index 1, this step convert 23 to the right hand joint.
        # we will only use :24 joints as the smpl joint model.
        pose[:, 23] = pose[:, 37]     # right hand
        pose = pose[:, :24].clone()   # only use body

        # align AMASS global fame with DIP
        amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
        tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
            amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))
    

        print('Synthesizing IMU accelerations and orientations')
        b = 0
        out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
        for i, l in tqdm(list(enumerate(length))):
            # discard short sequences
            if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
            p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
            # grot: [N, 24, 3, 3], global joint rotation matrices.
            # joint: [N, 24, 3], global joint positions, with global translation.
            # vert: [N, 6890, 3], global mesh vertex positions, with global translation.
            grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
            out_pose.append(pose[b:b + l].clone())  # N, 24, 3
            out_tran.append(tran[b:b + l].clone())  # N, 3
            out_shape.append(shape[i].clone())  # 10
            out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3   
            # vert[:, vi_mask].shape: [N, 6, 3]
            # order: left wrist, right wrist, left knee, right knee, head, pelvis
            out_vacc.append(utils.syn_acc2(vert[:, General.vi_mask]))  # N, 6, 3
            out_vrot.append(grot[:, General.ji_mask])  # N, 6, 3, 3
            b += l
        
        print(f'out_pose: len = {len(out_pose)}, shape = {out_pose[0].shape}')
        print(f'out_shape: len = {len(out_shape)}, shape = {out_shape[0].shape}')
        print(f'out_tran: len = {len(out_tran)}, shape = {out_tran[0].shape}')
        print(f'out_joint: len = {len(out_joint)}, shape = {out_joint[0].shape}')
        print(f'out_vrot: len = {len(out_vrot)}, shape = {out_vrot[0].shape}')
        print(f'out_vacc: len = {len(out_vacc)}, shape {out_vacc[0].shape}')

        print(f'### Saving {ds_name}')
        save_dir = f'{Paths.amass_dir}/{ds_name}'
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        while idx * n_max < len(out_pose):
            data = {
                'pose': out_pose[idx*n_max:(idx+1)*n_max],
                'shape': out_shape[idx*n_max:(idx+1)*n_max],
                'tran': out_tran[idx*n_max:(idx+1)*n_max],
                'joint': out_joint[idx*n_max:(idx+1)*n_max],
                'vrot': out_vrot[idx*n_max:(idx+1)*n_max],
                'vacc': out_vacc[idx*n_max:(idx+1)*n_max]
            }
            torch.save(data, f'{save_dir}/data_{idx}.pt')
            idx += 1
        print(f'Synthetic AMASS dataset {ds_name} is saved at {save_dir}')


def process_dipimu_test():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(Paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(Paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                _, joint = body_model.forward_kinematics(p)
                out_vacc.append(acc.clone())
                out_vrot.append(ori.clone())
                out_pose.append(pose.clone().view(-1, 24, 3))
                out_tran.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
                out_joint.append(joint)
                out_shape.append(torch.zeros(10))  # 10
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    print(f'### dipimu test length: {len(out_vacc)}')
    os.makedirs(Paths.dipimu_dir, exist_ok=True)
    out_data = {'pose': out_pose, 'shape': out_shape, 'tran': out_tran,
        'joint': out_joint, 'vrot': out_vrot, 'vacc': out_vacc}
    torch.save(out_data, f'{Paths.dipimu_dir}/test.pt')
    print('Preprocessed DIP-IMU dataset is saved at', Paths.dipimu_dir)
    
    
def process_dipimu_train():
    imu_mask = [7, 8, 11, 12, 0, 2]
    train_split = ['s_0%d' % i for i in range(1, 8)]
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []

    for subject_name in train_split:
        for motion_name in os.listdir(os.path.join(Paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(Paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                p = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                _, joint = body_model.forward_kinematics(p)
                out_pose.append(pose.view(-1, 24, 3).contiguous().clone())  # N, 24, 3
                out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
                out_tran.append(torch.zeros(pose.shape[0], 3))  # N, 3
                out_shape.append(torch.zeros(10))  # 10
                out_vacc.append(acc.contiguous().clone())  # N, 6, 3
                out_vrot.append(ori.contiguous().clone())  # N, 6, 3, 3
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(Paths.dipimu_dir, exist_ok=True)
    out_data = {'pose': out_pose, 'shape': out_shape, 'tran': out_tran,
        'joint': out_joint, 'vrot': out_vrot, 'vacc': out_vacc}
    print(f'### dipimu train length: {len(out_pose)}')
    torch.save(out_data, f'{Paths.dipimu_dir}/data_0.pt')
    print('Preprocessed DIP-IMU train dataset is saved at', Paths.dipimu_dir)


if __name__ == '__main__':
    # NOTE: select one of the following to run
    process_amass()
    # process_dipimu_train()
    # process_dipimu_test()   

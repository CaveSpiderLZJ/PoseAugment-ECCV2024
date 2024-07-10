import os
import torch
from visualize.joints2smpl.src import config
import smplx
import h5py
from visualize.joints2smpl.src.smplify import SMPLify3D
import utils.rotation_conversions as geometry
import argparse
import re
import tqdm
import pickle
import numpy as np
from glob import glob


class Joint2SMPL:


    def __init__(self, num_frames, device_id, cuda=True):
        self.device = torch.device("cuda:" + str(device_id) if cuda else "cpu")
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 150
        self.fix_foot = False
        print(config.SMPL_MODEL_DIR)
        smplmodel = smplx.create(config.SMPL_MODEL_DIR, model_type="smpl",
            gender="neutral", ext="pkl", batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = config.SMPL_MEAN_FILE
        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel, batch_size=self.batch_size,
            joints_category=self.joint_category, num_iters=self.num_smplify_iters, device=self.device)


    def npy2smpl(self, npy_path):
        out_path = npy_path.replace('.npy', '_rot.npy')
        motions = np.load(npy_path, allow_pickle=True)[None][0]
        # print_batch('', motions)
        n_samples = motions['motion'].shape[0]
        all_thetas = []
        for sample_i in tqdm.tqdm(range(n_samples)):
            thetas, _ = self.joint2smpl(motions['motion'][sample_i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            all_thetas.append(thetas.cpu().numpy())
        motions['motion'] = np.concatenate(all_thetas, axis=0)
        print('motions', motions['motion'].shape)

        print(f'Saving [{out_path}]')
        np.save(out_path, motions)
        exit()



    def joint2smpl(self, input_joints, init_params=None):
        _smplify = self.smplify # if init_params is None else self.smplify_fast
        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(self.batch_size, self.num_joints, 3).to(self.device)

        # run the whole seqs
        num_seqs = input_joints.shape[0]
        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()

        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, \
        new_opt_cam_t, new_opt_joint_loss = _smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
            # seq_ind=idx
        )

        thetas = new_opt_pose.reshape(self.batch_size, 24, 3)
        root_loc = torch.tensor(keypoints_3d[:, 0])  # [bs, 3]
        
        return root_loc.clone().detach(), thetas.clone().detach()


def joint2smpl():
    # NOTE: you should customize the data_dir and save_dir accordingly
    data_dir = f'data/mdm/'     # or 'data/mdm_m2m'
    save_dir = f'data/mdm_smpl' # or 'data/mdm_m2m_smpl'
    os.makedirs(save_dir, exist_ok=True)
    
    paths = glob(f'{data_dir}/results_*.npy')
    paths = list(sorted(paths, key=lambda x:int(re.search(r'\d+',x.split('/')[-1])[0])))
    
    n_valid = 120
    for path in paths:
        idx = int(re.search(r'\d+', path.split('/')[-1])[0])
        print(f'Converting {path} to smpl ...')
        data = np.load(path, allow_pickle=True)[()]
        motions = data['motion']
        motions = motions[:, :, :, :n_valid]
        # init Joint2SMPL
        model = Joint2SMPL(num_frames=motions.shape[-1], device_id=0, cuda=True)
        trans, poses = [], []
        for motion in tqdm.tqdm(motions):
            tran, pose = model.joint2smpl(motion.transpose(2, 0, 1))
            tran, pose = tran.cpu().numpy(), pose.cpu().numpy()
            trans.append(tran)
            poses.append(pose)
        trans = np.stack(trans)
        poses = np.stack(poses)
        data.update({'tran': trans, 'pose': poses})
        pickle.dump(data, open(f'{save_dir}/results_{idx}.pkl', 'wb'))
    
    
if __name__ == '__main__':
    # this process is extremely time consuming, may take several days
    joint2smpl()
    
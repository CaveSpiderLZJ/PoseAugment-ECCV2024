import torch
import numpy as np
from glob import glob
from infer_mvae import infer_mvae

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

import utils
import articulate as art
from config import Paths, TrainMVAE, JointSet
from articulate.vae import MVAE
from dynamics import VAEOptimizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(Paths.smpl_file)


def visualize_augmented_poses():
    ''' Show more qualitative results of augmented poses.
    '''
    # prepare MVAE model
    mvae = MVAE()
    model_path = f'{Paths.model_dir}/mvae_weights.pt'
    mvae.load_state_dict(torch.load(model_path, map_location=TrainMVAE.device))
    mvae.eval()
    
    # select pose data to visualize
    idx = 0
    dataset_name = 'ACCAD'
    start, end = 0, 500
    data_path = f'data/dataset_work/amass/{dataset_name}/data_0.pt'
    data = torch.load(data_path)
    pose, tran, joint = (data[key][idx][start:end] for key in ('pose', 'tran', 'joint'))
    n_aug = 9           # number of pose sequence to be augmented
    n_sample = 2           # number of frames to sample each time
    pose_aug = infer_mvae(mvae, pose, tran, joint, n_aug, n_sample)

    # setup pose viewer
    v = Viewer()
    v.playback_fps = 60.0
    smpl_layer = SMPLLayer(model_type="smpl", gender='male')

    if True:    # visualize ground truth pose
        pose_body = pose[:, 1:, :].view(-1, 69).numpy()
        pose_root = pose[:, 0, :].numpy()
        smpl_seq = SMPLSequence(poses_body=pose_body, smpl_layer=smpl_layer, poses_root=pose_root, trans=tran)
        smpl_seq.mesh_seq.color = (0, 0.5, 0, 0.5)  # green
        v.scene.add(smpl_seq)
            
    if True:    # visualize pose optimized by VAEOptimizer
        optimizer = VAEOptimizer()
        for i in range(n_aug):
            print(f'### Optimizing pose {i}')
            optimizer.reset_states()
            pose_aug_mat = art.math.axis_angle_to_rotation_matrix(pose_aug[i]).view(-1, 24, 3, 3)
            glb_rot, glb_joint = body_model.forward_kinematics(pose_aug_mat, tran=tran)
            joint_vel = utils.syn_vel(glb_joint)

            pose_opt, reaction_force = [], []
            for j, (p_, v_) in enumerate(zip(pose_aug_mat, joint_vel)):
                # print(f'### frame {j}: ', end='')
                p_opt, _, rf = optimizer.optimize_frame_sparse(p_, v_, calc_rf=True)
                pose_opt.append(p_opt)
                reaction_force.append(rf)
            reaction_force = np.stack(reaction_force).reshape(-1, JointSet.n_full, 3)
            
            pose_opt = art.math.rotation_matrix_to_axis_angle2(torch.stack(pose_opt)).view(-1, 24, 3)
            pose_body = pose_opt[:, 1:, :].view(-1, 69).numpy()
            pose_root = pose_opt[:, 0, :].numpy()
            smpl_seq = SMPLSequence(poses_body=pose_body, smpl_layer=smpl_layer,
                poses_root=pose_root, trans=tran)
            smpl_seq.mesh_seq.color = (0.5, 0, 0, 0.5)  # red
            v.scene.add(smpl_seq)
                    
    v.run()


if __name__ == '__main__':
    utils.init_rand_seed(TrainMVAE.seed)
    visualize_augmented_poses()

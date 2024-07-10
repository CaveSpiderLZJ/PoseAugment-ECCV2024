import os
import json
import tqdm
import torch
import pandas as pd

import utils
from config import General, Paths, JointSet, TrainTransPose
import articulate as art
from utils import normalize_and_concat
from transpose_net import TransPoseNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
body_model = art.ParametricModel(Paths.smpl_file)


class TransPoseEvaluator:
    
    
    def __init__(self):
        # joint arms: upper legs and upper arms, used in SIP error.
        self._eval_fn = art.FullMotionEvaluator(Paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))


    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, JointSet.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, JointSet.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100])

    
    @staticmethod
    def print(errors):
        metric_names = ['SIP Error (deg)', 'Angular Error (deg)',
            'Positional Error (cm)', 'Mesh Error (cm)', 'Jitter Error (100m/s^3)']
        for i, name in enumerate(metric_names):
            print(f'{name}: {errors[i,0]:.3f} ({errors[i,1]:.3f})')
            
    
    @staticmethod
    def print_short(errors):
        for i in range(5):
            print(f'{errors[i,0]:.3f}, ', end='')
        print()


def evaluate(model:TransPoseNet):
    model.eval()
    # load test data
    data = torch.load(f'{Paths.dipimu_dir}/test.pt')
    pose_all, acc_all, rot_all = data['pose'], data['vacc'], data['vrot']
    xs = [normalize_and_concat(a, r, acc_scale=TrainTransPose.acc_scale).to(device) for a, r in zip(acc_all, rot_all)]
    ys = [art.math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3) for p in pose_all]
    
    errs = []
    evaluator = TransPoseEvaluator()
    for x, y, rot in tqdm.tqdm(list(zip(xs, ys, rot_all))):
        x, y = x.to(device), y.to(device)
        leaf_joint_pos = model.pose_s1.forward([x])[0][0]
        full_joint_pos = model.pose_s2.forward([torch.cat([leaf_joint_pos, x], dim=1)])[0][0]
        global_reduced_pose = model.pose_s3.forward([torch.cat((full_joint_pos, x), dim=1)])[0][0]
        pose_p = model._reduced_glb_6d_to_full_local_mat(rot[:,-1].cpu(), global_reduced_pose.cpu())
        errs.append(evaluator.eval(pose_p, y))
    errs = torch.stack(errs).mean(dim=0)
    evaluator.print_short(errs)
    return errs.numpy()


def evaluate_augment(prefix:str, suffix:str, errs1=None):
    model = TransPoseNet(is_train=False).to(device)
    
    if errs1 is None:
        model_name = f'{prefix}1{suffix}'
        model.pose_s1.load_state_dict(torch.load(
            f'{Paths.transpose_dir}/{model_name}/pose_s1/best.pt', map_location=device))
        model.pose_s2.load_state_dict(torch.load(
            f'{Paths.transpose_dir}/{model_name}/pose_s2/best.pt', map_location=device))
        model.pose_s3.load_state_dict(torch.load(
            f'{Paths.transpose_dir}/{model_name}/pose_s3/best.pt', map_location=device))
        errs1 = evaluate(model)

    min_err1, best_model1 = 1e10, None
    min_err2, best_model2 = 1e10, None
    min_err3, best_model3 = 1e10, None
    for aug_scale in range(2, 6):
        model_name = f'{prefix}{aug_scale}{suffix}'
        res_dict = json.load(open(f'{Paths.transpose_dir}/{model_name}/pose_s1/result.json', 'r'))
        if res_dict['err'] < min_err1: min_err1, best_model1 = res_dict['err'], model_name
        res_dict = json.load(open(f'{Paths.transpose_dir}/{model_name}/pose_s2/result.json', 'r'))
        if res_dict['err'] < min_err2: min_err2, best_model2 = res_dict['err'], model_name
        res_dict = json.load(open(f'{Paths.transpose_dir}/{model_name}/pose_s3/result.json', 'r'))
        if res_dict['err'] < min_err3: min_err3, best_model3 = res_dict['err'], model_name
    model.pose_s1.load_state_dict(torch.load(
        f'{Paths.transpose_dir}/{best_model1}/pose_s1/best.pt', map_location=device))
    model.pose_s2.load_state_dict(torch.load(
        f'{Paths.transpose_dir}/{best_model2}/pose_s2/best.pt', map_location=device))
    model.pose_s3.load_state_dict(torch.load(
        f'{Paths.transpose_dir}/{best_model3}/pose_s3/best.pt', map_location=device))
    errs2 = evaluate(model)
    return errs1, errs2


def evaluate_mdm():
    # NOTE: test_id should be the same as the one in train_transpose.py
    test_id = 0
    dataset_version = f''
    save_file_name = f'transpose_test_{test_id}'
    (dataset_names, dataset_versions, augmented, err_sip, err_angular,
        err_pos, err_mesh, err_jitter) = [], [], [], [], [], [], [], []
    dataset_name_list = ['mdm', 'mdm_jitter', 'mdm_PoseAugment']
    errs1 = None
    for dataset_name in dataset_name_list:
        epoch = int(1000 / General.mdm_m2m_scale)
        prefix = f'{test_id}_{dataset_name}_aug'
        suffix = f'_epoch{epoch}'
        if errs1 is None:
            errs1, errs2 = evaluate_augment(prefix, suffix)
        else: errs1, errs2 = evaluate_augment(prefix, suffix, errs1=errs1)
        dataset_names.extend([dataset_name] * 2)
        dataset_versions.extend([dataset_version] * 2)
        augmented.extend([False, True])
        err_sip.extend([errs1[0,0], errs2[0,0]])
        err_angular.extend([errs1[1,0], errs2[1,0]])
        err_pos.extend([errs1[2,0], errs2[2,0]])
        err_mesh.extend([errs1[3,0], errs2[3,0]])
        err_jitter.extend([errs1[4,0], errs2[4,0]])
    res = {'dataset_name': dataset_names, 'dataset_version': dataset_versions,
        'augmented': augmented, 'err_sip': err_sip, 'err_angular': err_angular,
        'err_pos': err_pos, 'err_mesh': err_mesh, 'err_jitter': err_jitter}
    
    # evaluation results will be saved to Paths.analysis_dir
    os.makedirs(Paths.analysis_dir, exist_ok=True)
    df = pd.DataFrame(data=res)
    df.to_csv(f'{Paths.analysis_dir}/{save_file_name}.csv', index=False)
    
    
def evaluate_mdm_m2m():
    # NOTE: test_id should be the same as the one in train_transpose.py
    test_id = 1
    dataset_version = f''
    save_file_name = f'transpose_test_{test_id}'
    (dataset_names, dataset_versions, augmented, err_sip, err_angular,
        err_pos, err_mesh, err_jitter) = [], [], [], [], [], [], [], []
    dataset_name_list = ['mdm_m2m', 'mdm_m2m_jitter', 'mdm_m2m_PoseAugment']
    errs1 = None
    for dataset_name in dataset_name_list:
        epoch = int(1000 / General.mdm_m2m_scale)
        prefix = f'{test_id}_{dataset_name}_aug'
        suffix = f'_epoch{epoch}'
        if errs1 is None:
            errs1, errs2 = evaluate_augment(prefix, suffix)
        else: errs1, errs2 = evaluate_augment(prefix, suffix, errs1=errs1)
        dataset_names.extend([dataset_name] * 2)
        dataset_versions.extend([dataset_version] * 2)
        augmented.extend([False, True])
        err_sip.extend([errs1[0,0], errs2[0,0]])
        err_angular.extend([errs1[1,0], errs2[1,0]])
        err_pos.extend([errs1[2,0], errs2[2,0]])
        err_mesh.extend([errs1[3,0], errs2[3,0]])
        err_jitter.extend([errs1[4,0], errs2[4,0]])
    res = {'dataset_name': dataset_names, 'dataset_version': dataset_versions,
        'augmented': augmented, 'err_sip': err_sip, 'err_angular': err_angular,
        'err_pos': err_pos, 'err_mesh': err_mesh, 'err_jitter': err_jitter}
    
    # evaluation results will be saved to Paths.analysis_dir
    os.makedirs(Paths.analysis_dir, exist_ok=True)
    df = pd.DataFrame(data=res)
    df.to_csv(f'{Paths.analysis_dir}/{save_file_name}.csv', index=False)
    

def evaluate_actor():
    # NOTE: test_id should be the same as the one in train_transpose.py
    test_id = 2
    dataset_version = f''
    save_file_name = f'transpose_test_{test_id}'
    (dataset_names, dataset_versions, augmented, err_sip, err_angular,
        err_pos, err_mesh, err_jitter) = [], [], [], [], [], [], [], []
    dataset_name_list = ['actor', 'actor_jitter', 'actor_PoseAugment']
    for dataset_name in dataset_name_list:
        epoch = int(1000 / General.actor_scale)
        prefix = f'{test_id}_{dataset_name}_aug'
        suffix = f'_epoch{epoch}'
        errs1, errs2 = evaluate_augment(prefix, suffix)
        dataset_names.extend([dataset_name] * 2)
        dataset_versions.extend([dataset_version] * 2)
        augmented.extend([False, True])
        err_sip.extend([errs1[0,0], errs2[0,0]])
        err_angular.extend([errs1[1,0], errs2[1,0]])
        err_pos.extend([errs1[2,0], errs2[2,0]])
        err_mesh.extend([errs1[3,0], errs2[3,0]])
        err_jitter.extend([errs1[4,0], errs2[4,0]])
    res = {'dataset_name': dataset_names, 'dataset_version': dataset_versions,
        'augmented': augmented, 'err_sip': err_sip, 'err_angular': err_angular,
        'err_pos': err_pos, 'err_mesh': err_mesh, 'err_jitter': err_jitter}
    
    # evaluation results will be saved to Paths.analysis_dir
    os.makedirs(Paths.analysis_dir, exist_ok=True)
    df = pd.DataFrame(data=res)
    df.to_csv(f'{Paths.analysis_dir}/{save_file_name}.csv', index=False)
    

def evaluate_motionaug():
    # NOTE: test_id should be the same as the one in train_transpose.py
    test_id = 3
    dataset_version = f''
    save_file_name = f'transpose_test_{test_id}'
    (dataset_names, dataset_versions, augmented, err_sip, err_angular,
        err_pos, err_mesh, err_jitter) = [], [], [], [], [], [], [], []
    dataset_name_list = ['motionaug', 'motionaug_jitter', 'motionaug_PoseAugment']
    for dataset_name in dataset_name_list:
        epoch = int(1000 / General.motionaug_scale)
        prefix = f'{test_id}_{dataset_name}_aug'
        suffix = f'_epoch{epoch}'
        errs1, errs2 = evaluate_augment(prefix, suffix)
        dataset_names.extend([dataset_name] * 2)
        dataset_versions.extend([dataset_version] * 2)
        augmented.extend([False, True])
        err_sip.extend([errs1[0,0], errs2[0,0]])
        err_angular.extend([errs1[1,0], errs2[1,0]])
        err_pos.extend([errs1[2,0], errs2[2,0]])
        err_mesh.extend([errs1[3,0], errs2[3,0]])
        err_jitter.extend([errs1[4,0], errs2[4,0]])
    res = {'dataset_name': dataset_names, 'dataset_version': dataset_versions,
        'augmented': augmented, 'err_sip': err_sip, 'err_angular': err_angular,
        'err_pos': err_pos, 'err_mesh': err_mesh, 'err_jitter': err_jitter}
    
    # evaluation results will be saved to Paths.analysis_dir
    os.makedirs(Paths.analysis_dir, exist_ok=True)
    df = pd.DataFrame(data=res)
    df.to_csv(f'{Paths.analysis_dir}/{save_file_name}.csv', index=False)


if __name__ == '__main__':
    utils.init_rand_seed(TrainTransPose.seed)
    # NOTE: choose one to run
    evaluate_mdm()
    # evaluate_mdm_m2m()
    # evaluate_actor()
    # evaluate_motionaug()
    
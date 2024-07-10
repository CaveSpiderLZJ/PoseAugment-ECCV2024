r"""
    Config for paths, joint set, and normalizing scales.
"""

import torch


class General:
    acc_scale = 30
    vel_scale = 3
    # 1961: left wrist, 5424: right wrist
    # 1176: left knee, 4662: right knee
    # 411: head, 3021: pelvis
    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
    # datasets (directory names) in AMASS
    # e.g., for ACCAD, the path should be `Paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
    amass = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
        'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
        'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust_67']
    amass_scale = {'ACCAD': 1.000, 'KIT': 23.943, 'CMU': 19.220, 'BioMotionLab_NTroje': 18.617,
        'Eyes_Japan_Dataset': 11.132, 'BMLmovi': 6.277, 'MPI_HDM05': 4.874, 'BMLhandball': 3.612,
        'TotalCapture': 1.370, 'EKUT': 1.156, 'MPI_Limits': 0.698, 'Transitions_mocap': 0.551, 'SFU': 0.524,
        'DFaust_67': 0.427, 'HumanEva': 0.291, 'MPI_mosh': 0.284, 'TCD_handMocap': 0.264, 'SSM_synced': 0.081}
    mdm_scale = 20.138
    mdm_m2m_scale = 20.138
    actor_scale = 5.035
    motionaug_scale = 6.000
    

class Paths:
    raw_amass_dir = 'data/dataset_raw/amass'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = 'data/dataset_work/amass'         # output path for the synthetic AMASS dataset
    amass_aug_dir = 'data/dataset_aug/amass'      # changed to ablation temporarily

    raw_dipimu_dir = 'data/dataset_raw/dip'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'data/dataset_work/dip'      # output path for the preprocessed DIP-IMU dataset
    dipimu_aug_dir = 'data/dataset_aug/dip'
    
    mdm_dir = 'data/dataset_work/mdm'                   # pose data generated by MDM
    mdm_aug_dir = 'data/dataset_aug/mdm'                # preprocessed MDM data, and augmented MDM data by PoseAugment
    mdm_m2m_dir = 'data/dataset_work/mdm_m2m'           # pose data generated by MDM-M2M
    mdm_m2m_aug_dir = 'data/dataset_aug/mdm_m2m'        # preprocessed MDM-M2M data, and augmented MDM-M2M data by PoseAugment
    actor_dir = 'data/dataset_work/actor'               # pose data generated by ACTOR
    actor_aug_dir = 'data/dataset_aug/actor'            # preprocessed actor data, and augmented actor data by PoseAugment
    motionaug_dir = 'data/dataset_work/motionaug'       # pose data generated by MotionAugment
    motionaug_aug_dir = 'data/dataset_aug/motionaug'    # preprocessed motionaug data, and augmented motionaug data by PoseAugment

    model_dir = f'data/model'                                   # trained models and physical models
    vae_dir = f'data/vae'                                       # vae model dir
    transpose_dir = f'data/transpose'                           # transpose model dir
    analysis_dir = f'data/analysis'
    smpl_file = 'data/model/smpl_male.pkl'                      # official SMPL model path
    physics_model_file = 'data/model/physics.urdf'              # physics body model path
    transpose_weights_file = 'data/model/transpose_weights.pt'  # transpose weight file
    pip_weights_file = 'data/model/pip_weights.pt'              # pip weight file
    physics_parameter_file = 'physics_parameters.json'          # physics hyperparameters


class JointSet:
    leaf = [7, 8, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    # 1, 2, 3 should be excluded from joint_pos and joint_vel in pose_frame
    aug = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]   # no 10, 11, 22, 23

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
    n_aug = len(aug)
    
    
class TrainTransPose:
    seed = 3407
    base_epoch = 1000   # base epoch when aug_scale = 1
    lr1 = 1e-4
    lr2 = 5e-4
    lr3 = 5e-4
    batch_size1 = 128
    batch_size2 = 64
    batch_size3 = 64
    n_val_steps1 = 25
    n_val_steps2 = 50
    n_val_steps3 = 50
    test_batch_size = 10
    acc_scale = 30
    model_name = 'ACCAD5_aug5_combined_epoch500_lr5e-4'
    aug_scale: int = 5
    aug_type: str = 'combined'   # in {'vae', 'jitter', 'mag_warp', 'time_warp', 'combined'}
    
    
class TrainMVAE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 3407
    lr = 2e-5
    n_warmup_epoch = 10     # learning rate: from 0.1 * lr -> lr, use supervized learning
    n_teacher_epoch = 50    # 50, supervized learning (p = 1)
    n_ramping_epoch = 150    # 150, scheduled sampling (p from 1 to 0)
    n_student_epoch = 200   # 200, autoregressive prection (p = 0)
    mini_batch_size = 30    # use scheduled sampling within each mini batch
    batch_size = 512
    test_batch_size = 32
    model_name = '97_mvae13_ckl3e-3_epoch400_lr2e-5'
    c_kl = 3e-3
    val_steps = 2           # 2
    mean_ratio = 0.1
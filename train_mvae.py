import os
import time
import tqdm
import shutil
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from typing import List

import utils
import articulate as art
from articulate.vae import *
from config import Paths, JointSet, TrainMVAE, General


BODY_MODEL = art.ParametricModel(Paths.smpl_file)


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader):
    print(f'### len(train_dataloader): {len(train_dataloader)}')
    print(f'### len(val_dataloader): {len(val_dataloader)}')
    print(f'### len(test_dataloader): {len(test_dataloader)}')
    
    log_save_dir = f'data/log/{TrainMVAE.model_name}'
    model_save_dir = f'data/vae/{TrainMVAE.model_name}'
    if os.path.exists(log_save_dir): shutil.rmtree(log_save_dir)
    os.makedirs(log_save_dir, exist_ok=True)
    if os.path.exists(model_save_dir): shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir, exist_ok=True)
    
    n_warmup_epoch = TrainMVAE.n_warmup_epoch
    n_teacher_epoch = TrainMVAE.n_teacher_epoch
    n_ramping_epoch = TrainMVAE.n_ramping_epoch
    n_student_epoch = TrainMVAE.n_student_epoch
    n_epoch = n_warmup_epoch + n_teacher_epoch + n_ramping_epoch + n_student_epoch
    p_supervised = np.concatenate([np.ones(n_warmup_epoch + n_teacher_epoch),
        np.linspace(1, 0, n_ramping_epoch), np.zeros(n_student_epoch)])
    mini_bs = TrainMVAE.mini_batch_size
    model = model.to(TrainMVAE.device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainMVAE.lr)
    # in each mini batch, the model will update (mini_bs - 1) times
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        start_factor=0.1, end_factor=1.0, total_iters=n_warmup_epoch)
    train_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
        schedulers=[warmup_scheduler, train_scheduler], milestones=[n_warmup_epoch])
    logger = SummaryWriter(log_save_dir)
    
    min_val_loss = 1e30
    mean_train_loss, mean_train_reconstruction_loss = [], []
    mean_val_loss, mean_val_reconstruction_loss = [], []
    model.train()
    optimizer.zero_grad()
    tic = time.perf_counter()
    
    for epoch, p in zip(range(n_epoch), p_supervised):
        train_loss = []
        train_reconstruction_loss = []
        for data, _ in train_dataloader:   # label not used when training MVAE
            data = data.to(TrainMVAE.device)
            output: torch.Tensor = None
            for i in range(mini_bs - 1):
                x = data[:, i+1, :]
                c = data[:, i, :] if (output is None or np.random.randn() <= p) else output.detach()
                output, mu, logvar = model(x, c)
                reconstruction_loss: torch.Tensor = loss_fn(output, x)
                kl_loss = -0.5 * TrainMVAE.c_kl * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + kl_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss.append(loss.detach().item())
                train_reconstruction_loss.append(reconstruction_loss.detach().item())
        
        scheduler.step()
        train_loss = np.mean(train_loss)
        train_reconstruction_loss = np.mean(train_reconstruction_loss)
        mean_train_loss.append(train_loss)
        mean_train_reconstruction_loss.append(train_reconstruction_loss)
        logger.add_scalar('Train Loss', train_loss, epoch)
        logger.add_scalar('Train Reconstruction Loss', train_reconstruction_loss, epoch)
        logger.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        if (epoch + 1) % TrainMVAE.val_steps != 0: continue
        model.eval()
        val_loss = []
        val_reconstruction_loss = []
        with torch.no_grad():
            for data, _ in val_dataloader:
                data = data.to(TrainMVAE.device)
                output: torch.Tensor = data[:, 0, :]
                for i in range(mini_bs - 1):
                    x = data[:, i+1, :]
                    output, mu, logvar = model(x, output.detach())
                    reconstruction_loss: torch.Tensor = loss_fn(output, x)
                    kl_loss = -0.5 * TrainMVAE.c_kl * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = reconstruction_loss + kl_loss
                    val_loss.append(loss.detach().item())
                    val_reconstruction_loss.append(reconstruction_loss.detach().item())
        val_loss = np.mean(val_loss)
        val_reconstruction_loss = np.mean(val_reconstruction_loss)
        mean_val_loss.append(val_loss)
        mean_val_reconstruction_loss.append(val_reconstruction_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{model_save_dir}/best.model')
        torch.save(model.state_dict(), f'{model_save_dir}/last.model')
        logger.add_scalar('Val Loss', val_loss, epoch)
        logger.add_scalar('Val Reconstruction Loss', val_reconstruction_loss, epoch)
        timestamp = time.perf_counter() - tic
        print(f'### timestamp: {timestamp:.3f} s, epoch: {epoch}, train_loss: ({train_loss:.4e}, {train_reconstruction_loss:.4e}), ' + \
            f'val loss: ({val_loss:.4e}, {val_reconstruction_loss:.4e})')
        model.train()
    
    logger.close()
    cnt = max(1, int(len(mean_train_loss) * TrainMVAE.mean_ratio))
    mean_train_loss = np.mean(mean_train_loss[-cnt:])
    mean_train_reconstruction_loss = np.mean(mean_train_reconstruction_loss[-cnt:])
    cnt = max(1, int(len(mean_val_loss) * TrainMVAE.mean_ratio))
    mean_val_loss = np.mean(mean_val_loss[-cnt:])
    mean_val_reconstruction_loss = np.mean(mean_val_reconstruction_loss[-cnt:])
    print(f'### mean train loss: ({mean_train_loss:.4e}, {mean_train_reconstruction_loss:.4e})')
    print(f'### mean val loss: ({mean_val_loss:.4e}, {mean_val_reconstruction_loss:.4e})')
    
    
def get_mvae_frames(pose: torch.Tensor, tran: torch.Tensor, joint: torch.Tensor):
    N = pose.shape[0]
    root_vel = utils.syn_vel(tran, smooth_n=2)
    root_rot = art.math.rotation_matrix_to_r6d(art.math.axis_angle_to_rotation_matrix(pose[:,0,:]))
    joint_pos = joint[:, JointSet.aug, :] - joint[:, 0:1, :]
    joint_pos = joint_pos.bmm(art.math.axis_angle_to_rotation_matrix(pose[:,0,:]))
    joint_vel = utils.syn_vel(joint_pos, smooth_n=2)
    joint_rot = torch.cat([torch.zeros(N,1,3), pose[:,1:,:]], dim=1)
    joint_rot = BODY_MODEL.forward_kinematics_R(art.math.axis_angle_to_rotation_matrix(joint_rot).view(-1,24,3,3))
    joint_rot = art.math.rotation_matrix_to_r6d(joint_rot[:,JointSet.aug,:]).view(-1, JointSet.n_aug, 6)
    frames = torch.cat([tran, root_vel, root_rot, joint_pos.flatten(1),
        joint_vel.flatten(1), joint_rot.flatten(1)], dim=1) # [N, 240]
    return frames
    
    
def train_mvae():
    
    
    def get_dataset(data_paths: List[str]):
        mini_batch_size = TrainMVAE.mini_batch_size
        all_data, all_label = [], []
        for data_path in tqdm.tqdm(data_paths):
            data = torch.load(data_path)
            for pose, tran, joint in zip(data['pose'], data['tran'], data['joint']):
                if pose.shape[0] < mini_batch_size: continue
                frames = get_mvae_frames(pose, tran, joint)
                frames = torch.split(frames, mini_batch_size, dim=0)
                all_data.extend(frames[:-1])
                all_label.extend([torch.empty(0)] * (len(frames)-1))
                if frames[-1].shape[0] == mini_batch_size:
                    all_data.append(frames[-1])
                    all_label.append(torch.empty(0))
        return VAEDataset(all_data, all_label)

    
    def get_train_dataset():
        data_paths = []
        for dataset_name in General.amass:
            data_paths.extend(glob(f'{Paths.amass_dir}/{dataset_name}/data_*.pt'))
        data_paths.append(f'{Paths.dipimu_dir}/data_0.pt')
        return get_dataset(data_paths)
    
    
    def get_test_dataset():
        # use DIP-IMU-test for testing
        data_paths = [f'{Paths.dipimu_dir}/test.pt']
        return get_dataset(data_paths)
        

    model = MVAE()
    train_dataloader = torch.utils.data.DataLoader(get_train_dataset(), batch_size=TrainMVAE.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(get_test_dataset(), batch_size=TrainMVAE.batch_size, shuffle=True)
    train(model, train_dataloader, test_dataloader, test_dataloader)


if __name__ == '__main__':
    utils.init_rand_seed(TrainMVAE.seed)
    train_mvae()

import torch
import src.utils.rotation_conversions as geometry

from .smpl import SMPL, JOINTSTYPE_ROOT
from .get_model import JOINTSTYPES


class Rotation2smpl:
    ''' This class is modified from Rotation2xyz.
        Customized to generated poses for PoseAugment.
    '''
    
    
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)


    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):
        
        if False:
            print(f'### rot2smpl called')
            print(f'\t{x.shape=}, {mask.shape=}')
            print(f'\t{pose_rep=}, {translation=}, {glob=}, {jointstype=}, {vertstrans=}')
        
        if pose_rep == "xyz": return x
        if mask is None: mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)  # all True
        
        # glob = True, glob_rot = [pi, 0, 0]
        if not glob and glob_rot is None: raise TypeError("You must specify global rotation if glob is False")
        # jointstype = 'smpl'
        if jointstype not in JOINTSTYPES: raise NotImplementedError("This jointstype is not implemented.")

        if translation: x_translations, x_rotations = x[:, -1, :3], x[:, :-1]   # [O]
        else: x_rotations = x   # [X]

        x_rotations = x_rotations.permute(0, 3, 1, 2)       # [n_sample * 12, n_frame, 24, 6]
        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":    # [X]
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":  # [X]
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat": # [X]
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":   # [O]
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else: raise NotImplementedError("No geometry for this one.")
        # rotations.shape = [12 * n_sample * n_frame, 24, 3, 3]
        
        # apply glob_rot to rotations
        glob_rot = geometry.axis_angle_to_matrix(torch.tensor(glob_rot, device=x.device))
        glob_rot = glob_rot.view(1, 3, 3).repeat(nsamples*time, 1, 1)
        rotations[:, 0] = torch.bmm(glob_rot, rotations[:, 0])
        
        # matrix to axis_angle
        out_smpl = geometry.matrix_to_axis_angle(rotations)
        out_smpl = out_smpl.reshape(nsamples, time, njoints, 3).permute(0, 2, 3, 1)
        
        # should output [N, 24, 3, 60]
        return out_smpl

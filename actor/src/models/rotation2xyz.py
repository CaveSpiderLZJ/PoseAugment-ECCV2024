import torch
import src.utils.rotation_conversions as geometry

from .smpl import SMPL, JOINTSTYPE_ROOT
from .get_model import JOINTSTYPES


class Rotation2xyz:
    def __init__(self, device):
        self.device = device
        self.smpl_model = SMPL().eval().to(device)

    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, **kwargs):
        
        if False:
            print(f'### rot2xyz called')
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

        if not glob:    # [X]
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:   # [O]
            global_orient = rotations[:, 0]     # [N, 3, 3]
            rotations = rotations[:, 1:]        # [N, 23, 3, 3]
            
        if betas is None:   # [O]
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
        # betas are all zero
        
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype]

        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]

        return x_xyz

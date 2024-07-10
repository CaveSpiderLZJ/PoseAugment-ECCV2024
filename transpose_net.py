import torch
import torch.nn as nn

from config import General, JointSet, Paths
import articulate as art
from articulate.rnn import RNN


class TransPoseNet(nn.Module):
    ''' Whole pipeline for pose and translation estimation.
    '''
    
    def __init__(self, num_past_frame=20, num_future_frame=5, hip_length=None, upper_leg_length=None,
                 lower_leg_length=None, prob_threshold=(0.5, 0.9), gravity_velocity=-0.018, is_train:bool=False):
        r"""
        :param num_past_frame: Number of past frames for a biRNN window.
        :param num_future_frame: Number of future frames for a biRNN window.
        :param hip_length: Hip length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param upper_leg_length: Upper leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param lower_leg_length: Lower leg length in meters. SMPL mean length is used by default. Float or tuple of 2.
        :param prob_threshold: The probability threshold used to control the fusion of the two translation branches.
        :param gravity_velocity: The gravity velocity added to the Trans-B1 when the body is not on the ground.
        """
        super().__init__()
        n_imu = 6 * 3 + 6 * 9   # acceleration (vector3) and rotation matrix (matrix3x3) of 6 IMUs
        self.pose_s1 = RNN(n_imu,                         JointSet.n_leaf * 3,        256)
        self.pose_s2 = RNN(JointSet.n_leaf * 3 + n_imu,  JointSet.n_full * 3,        64)
        self.pose_s3 = RNN(JointSet.n_full * 3 + n_imu,  JointSet.n_reduced * 6,     128)
        self.tran_b1 = RNN(JointSet.n_leaf * 3 + n_imu,  2,                          64)
        self.tran_b2 = RNN(JointSet.n_full * 3 + n_imu,  3,                          256,    bidirectional=False)

        # lower body joint
        m = art.ParametricModel(Paths.smpl_file)
        j, _ = m.get_zero_pose_joint_and_vertex()
        b = art.math.joint_position_to_bone_vector(j[JointSet.lower_body].unsqueeze(0),
                                                   JointSet.lower_body_parent).squeeze(0)
        bone_orientation, bone_length = art.math.normalize_tensor(b, return_norm=True)
        if hip_length is not None:
            bone_length[1:3] = torch.tensor(hip_length)
        if upper_leg_length is not None:
            bone_length[3:5] = torch.tensor(upper_leg_length)
        if lower_leg_length is not None:
            bone_length[5:7] = torch.tensor(lower_leg_length)
        b = bone_orientation * bone_length
        b[:3] = 0

        # constant
        self.global_to_local_pose = m.inverse_kinematics_R
        self.lower_body_bone = b
        self.num_past_frame = num_past_frame
        self.num_future_frame = num_future_frame
        self.num_total_frame = num_past_frame + num_future_frame + 1
        self.prob_threshold = prob_threshold
        self.gravity_velocity = torch.tensor([0, gravity_velocity, 0])
        self.feet_pos = j[10:12].clone()
        self.floor_y = j[10:12, 1].min().item()

        # variable
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)
        self.reset()

        if is_train == False:
            self.load_state_dict(torch.load(Paths.transpose_weights_file))
            self.eval()


    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        # print(f'### glb_reduced_pose.shape: {glb_reduced_pose.shape}')
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, JointSet.n_reduced, 3, 3)
        # print(f'### glb_reduced_pose.shape: {glb_reduced_pose.shape}')
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, JointSet.reduced] = glb_reduced_pose
        pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        pose[:, JointSet.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose


    def _prob_to_weight(self, p):
        return (p.clamp(self.prob_threshold[0], self.prob_threshold[1]) - self.prob_threshold[0]) / \
               (self.prob_threshold[1] - self.prob_threshold[0])


    def reset(self):
        r"""
        Reset online forward states.
        """
        self.rnn_state = None
        self.imu = None
        self.current_root_y = 0
        self.last_lfoot_pos, self.last_rfoot_pos = self.feet_pos
        self.last_root_pos = torch.zeros(3)


    def forward(self, imu, rnn_state=None):
        # print(f'### in forward')
        # print(f'### imu.shape: {imu.shape}')
        leaf_joint_position = self.pose_s1.forward(imu)[0]
        # print(f'### leaf_joint_position.shape: {leaf_joint_position.shape}')
        full_joint_position = self.pose_s2.forward(torch.cat((leaf_joint_position, imu), dim=1))[0]
        # print(f'### full_joint_position.shape: {full_joint_position.shape}')
        global_reduced_pose = self.pose_s3.forward(torch.cat((full_joint_position, imu), dim=1))[0]
        # print(f'### global_reduced_pose.shape: {global_reduced_pose.shape}')
        contact_probability = self.tran_b1.forward(torch.cat((leaf_joint_position, imu), dim=1))[0]
        velocity, rnn_state = self.tran_b2.forward(torch.cat((full_joint_position, imu), dim=1), rnn_state)
        return leaf_joint_position, full_joint_position, global_reduced_pose, contact_probability, velocity, rnn_state


    @torch.no_grad()
    def forward_offline(self, imu):
        r"""
        Offline forward.

        :param imu: Tensor in shape [num_frame, input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [num_frame, 24, 3, 3] and translation tensor in shape [num_frame, 3].
        """
        # print(f'### in forward_offline()')
        # print(f'### imu.shape: {imu.shape}')
        _, _, global_reduced_pose, contact_probability, velocity, _ = self.forward(imu)
        # print(f'### global_reduced_pose.shape: {global_reduced_pose.shape}')
        # print(f'### contact_probability.shape: {contact_probability.shape}')
        # print(f'### velocity.shape: {velocity.shape}')

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[:, -9:].view(-1, 3, 3)
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation.cpu(), global_reduced_pose.cpu())
        # print(f'### pose.shape: {pose.shape}')

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        j = art.math.forward_kinematics(pose[:, JointSet.lower_body],
                                        self.lower_body_bone.expand(pose.shape[0], -1, -1),
                                        JointSet.lower_body_parent)[1]
        tran_b1_vel = self.gravity_velocity + art.math.lerp(
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 7] - j[1:, 7])),
            torch.cat((torch.zeros(1, 3, device=j.device), j[:-1, 8] - j[1:, 8])),
            contact_probability.max(dim=1).indices.view(-1, 1).cpu()
        )
        tran_b2_vel = root_rotation.bmm(velocity.unsqueeze(-1)).squeeze(-1).cpu() * General.vel_scale / 60   # to world space
        weight = self._prob_to_weight(contact_probability.cpu().max(dim=1).values.sigmoid()).view(-1, 1)
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_root_y = 0
        for i in range(velocity.shape[0]):
            current_foot_y = current_root_y + j[i, 7:9, 1].min().item()
            if current_foot_y + velocity[i, 1].item() <= self.floor_y:
                velocity[i, 1] = self.floor_y - current_foot_y
            current_root_y += velocity[i, 1].item()
        return pose, self.velocity_to_root_position(velocity)


    @torch.no_grad()
    def forward_online(self, x):
        r"""
        Online forward.

        :param x: A tensor in shape [input_dim(6 * 3 + 6 * 9)].
        :return: Pose tensor in shape [24, 3, 3] and translation tensor in shape [3].
        """
        # print(f'### x.shape: {x.shape}')
        # get num_total_frame frames of imu data
        imu = x.repeat(self.num_total_frame, 1) if self.imu is None else torch.cat((self.imu[1:], x.view(1, -1)))
        # print(f'### imu.shape: {imu.shape}')
        _, _, global_reduced_pose, contact_probability, velocity, self.rnn_state = self.forward(imu, self.rnn_state)
        contact_probability = contact_probability[self.num_past_frame].sigmoid().view(-1).cpu()

        # calculate pose (local joint rotation matrices)
        root_rotation = imu[self.num_past_frame, -9:].view(3, 3).cpu()
        global_reduced_pose = global_reduced_pose[self.num_past_frame].cpu()
        pose = self._reduced_glb_6d_to_full_local_mat(root_rotation, global_reduced_pose).squeeze(0)

        # calculate velocity (translation between two adjacent frames in 60fps in world space)
        lfoot_pos, rfoot_pos = art.math.forward_kinematics(pose[JointSet.lower_body].unsqueeze(0),
                                                           self.lower_body_bone.unsqueeze(0),
                                                           JointSet.lower_body_parent)[1][0, 7:9]
        if contact_probability[0] > contact_probability[1]:
            tran_b1_vel = self.last_lfoot_pos - lfoot_pos + self.gravity_velocity
        else:
            tran_b1_vel = self.last_rfoot_pos - rfoot_pos + self.gravity_velocity
        tran_b2_vel = root_rotation.mm(velocity[self.num_past_frame].cpu().view(3, 1)).view(3) / 60 * General.vel_scale
        weight = self._prob_to_weight(contact_probability.max())
        velocity = art.math.lerp(tran_b2_vel, tran_b1_vel, weight)

        # remove penetration
        current_foot_y = self.current_root_y + min(lfoot_pos[1].item(), rfoot_pos[1].item())
        if current_foot_y + velocity[1].item() <= self.floor_y:
            velocity[1] = self.floor_y - current_foot_y

        self.current_root_y += velocity[1].item()
        self.last_lfoot_pos, self.last_rfoot_pos = lfoot_pos, rfoot_pos
        self.imu = imu
        self.last_root_pos += velocity
        return pose, self.last_root_pos.clone()


    @staticmethod
    def velocity_to_root_position(velocity):
        r"""
        Change velocity to root position. (not optimized)

        :param velocity: Velocity tensor in shape [num_frame, 3].
        :return: Translation tensor in shape [num_frame, 3] for root positions.
        """
        return torch.stack([velocity[:i+1].sum(dim=0) for i in range(velocity.shape[0])])

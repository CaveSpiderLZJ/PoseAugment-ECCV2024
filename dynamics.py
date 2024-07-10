import torch
import numpy as np
import scipy.sparse as sparse
from qpsolvers import solve_qp

import utils
import articulate as art
from config import Paths, JointSet
from articulate.rbdl_model import RBDLModel
from utils import Body, smpl_to_rbdl, rbdl_to_smpl


class PhysicsOptimizer:
    # no 'LANKLE', 'RANKLE', 'NECK', 'LWRIST', 'RWRIST', 'LCLAVICLE', 'RCLAVICLE'
    test_contact_joints = ['LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'SPINE3', 'LSHOULDER',
        'RSHOULDER', 'HEAD', 'LELBOW', 'RELBOW', 'LHAND', 'RHAND', 'LFOOT', 'RFOOT']
    joint_name_to_id = {'LHIP': 1, 'RHIP': 2, 'SPINE1': 3, 'LKNEE': 4, 'RKNEE': 5, 'SPINE2': 6,
        'SPINE3': 9, 'LSHOULDER': 16, 'RSHOULDER': 17, 'HEAD': 15, 'LELBOW': 18, 'RELBOW': 19,
        'LHAND': 22, 'RHAND': 23, 'LFOOT': 10, 'RFOOT': 11}

    def __init__(self, debug=False):
        mu = 0.6                # static friction coefficient (paper 3.2.3)
        supp_poly_size = 0.2    # contact square length (paper 3.2.3)
        self.debug = debug
        # black box model
        self.model = RBDLModel(Paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = utils.read_debug_param_values_from_json(Paths.physics_parameter_file)
        # TODO: how to use it?
        self.friction_constraint_matrix = np.array([[np.sqrt(2), -mu, 0],
                                                    [-np.sqrt(2), -mu, 0],
                                                    [0, -mu, np.sqrt(2)],
                                                    [0, -mu, -np.sqrt(2)]])
        self.support_polygon = np.array([[-supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,  -supp_poly_size / 2],
                                         [-supp_poly_size / 2,  0,   supp_poly_size / 2],
                                         [ supp_poly_size / 2,  0,   supp_poly_size / 2]])

        # states
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.reset_states()

    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)

    def optimize_frame(self, pose, jvel, contact, acc, calc_rf:bool=False):
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        v_ref = jvel.numpy()
        c_ref = contact.sigmoid().numpy()
        a_ref = acc.numpy()
        q = self.q
        qdot = self.qdot
        
        if q is None:
            self.q = q_ref
            if calc_rf: return pose, torch.zeros(3), np.zeros(JointSet.n_full*3, dtype=np.float64)
            return pose, torch.zeros(3)

        # determine the contact joints and points
        # TODO: why qddot is zero?
        # qdot_size = 75 = 25 * 3
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = [np.empty((0, self.model.qdot_size))]
        collision_points, collision_joints = [], []
        for joint_name in self.test_contact_joints:
            joint_id = vars(Body)[joint_name]
            pos = self.model.calc_body_position(q, joint_id)
            if joint_id == Body.LFOOT and c_ref[0] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03 or \
               joint_id == Body.RFOOT and c_ref[1] > 0.5 and pos[1] <= self.params['floor_y'] + 0.03 or \
               pos[1] <= self.params['floor_y']:
                collision_joints.append(joint_name)
                for ps in self.support_polygon + pos:   # will append Js 4 times
                    collision_points.append(ps)
                    pb = self.model.calc_base_to_body_coordinates(q, joint_id, ps)
                    Js.append(self.model.calc_point_Jacobian(q, joint_id, pb))
        Js = np.vstack(Js)  # shape = [:, 75]
        nc = len(collision_points)  # number of contact points

        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], [np.empty((0, nc * 3))], \
                                       [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None

        # joint angle PD controller
        # TODO: not consistent with the paper.
        # minimize_{\ddot{\theta}} (0.5 * (\ddot{\theta})^T - \ddot{\theta}_{des}) @ \ddot{\theta}_{des}
        if True:
            A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
            # b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            b = 2400 * art.math.angle_difference(q_ref[3:], q[3:]) - 60 * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72

        # joint position PD controller (using root velocity + ref pose to determine target joint position)
        if False:
            for joint_name in ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                               'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                               'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND']:
                joint_id = vars(Body)[joint_name]
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                cur_pos = self.model.calc_body_position(q, joint_id)
                tar_pos = self.model.calc_body_position(q_ref, joint_id) - q_ref[:3] + q[:3] + v_ref[0] * self.params['delta_t']
                a_des = 3600 * (tar_pos - cur_pos) - 60 * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * 2)
                bs1.append(b * 2)

        # joint position PD controller (using joint velocity to determine target joint position)
        # minimize_{\ddot{q}} (0.5 * J @ \ddot{q} + \ddot{r} - \ddot{r}_{res})^T @ J @ \ddot{q}
        if True:
            # no LHAND and RHAND.
            for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                                      'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                                      'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
                joint_id = vars(Body)[joint_name]
                if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                # a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
                a_des = 3600 * v * self.params['delta_t'] - 60 * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des
                As1.append(A * self.params['coeff_jvel'])
                bs1.append(b * self.params['coeff_jvel'])

        # joint velocity (without Jdot * qdot term)
        if False:
            for joint_name, v in zip(
                    ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
                     'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                     'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND'], v_ref):
                joint_id = vars(Body)[joint_name]
                A = self.model.calc_point_Jacobian(q, joint_id)
                b = (-self.model.calc_point_velocity(q, qdot, joint_id) + v) / self.params['delta_t']
                As1.append(A * 2)
                bs1.append(b * 2)

        # IMU acceleration
        if False:
            for joint_name, a in zip(['LWRIST', 'RWRIST', 'LKNEE', 'RKNEE', 'HEAD', 'ROOT'], a_ref):
                joint_id = vars(Body)[joint_name]
                offset = np.zeros(3)
                A = self.model.calc_point_Jacobian(q, joint_id, offset)
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(self.model.qdot_size), joint_id, offset) + a
                bs1.append(b * self.params['coeff_acc'])
                As1.append(A * self.params['coeff_acc'])

        # lambda size
        if False:
            As2.append(np.eye(nc * 3) * self.params['coeff_lambda_old'])
            bs2.append(np.zeros(nc * 3))

        # Signorini’s conditions of lambda
        # minimize_{\lambda} 0.5 * k_{\lambda}^2 * \sum_{c=1}^{n_c}(d_c^2 * norm(\lambda_c)^2)
        if True:
            if nc != 0:
                A = [np.eye(3) * max(cp[1] - self.params['floor_y'], 0.005) for cp in collision_points]
                A = art.math.block_diagonal_matrix_np(A)
                # As2.append(A * self.params['coeff_lambda'])
                As2.append(A * 10)
                bs2.append(np.zeros(nc * 3))

        # tau size
        # minimize_{\lambda} k_{res} * norm(\tau[:6])^2 + k_{\tau} * norm(\tau[6:])^2
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                # np.eye(6) * self.params['coeff_virtual'],
                np.eye(6) * 0.1,
                # np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']
                np.eye(self.model.qdot_size - 6) * 0.01
            ]))
            bs3.append(np.zeros(self.model.qdot_size))

        # contacting body joint velocity
        # paper 3.2.3 equation 12.
        # constraint 1: v + J * qddot * \Delta t >= [-0.1, 0, -0.1]
        # constraint 2: v + J * qddot * \Delta t <= [0.1, 100, 0.1]
        if True:
            for joint_name in self.test_contact_joints[:-2]:    # no LFOOT and RFOOT
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                if pos[1] <= self.params['floor_y']:
                    J = self.model.calc_point_Jacobian(q, joint_id)
                    v = self.model.calc_point_velocity(q, qdot, joint_id)
                    Gs1.append(-self.params['delta_t'] * J)
                    hs1.append(v - [-1e-1, 0, -1e-1])
                    Gs1.append(self.params['delta_t'] * J)
                    hs1.append(-v + [1e-1, 1e2, 1e-1])

        # contacting foot velocity
        # constraint 1: v + J * qddot * \Delta t >= [-th, th_y, -th]
        # constraint 2: v + J * qddot * \Delta t <= [th, max(th, th_y), th]
        if True:
            for joint_name, stable in zip(['LFOOT', 'RFOOT'], c_ref):
                joint_id = vars(Body)[joint_name]
                pos = self.model.calc_body_position(q, joint_id)
                J = self.model.calc_point_Jacobian(q, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id)

                th = -np.log(min(stable, 0.84999) / 0.85)
                th_y = (self.params['floor_y'] - pos[1]) / self.params['delta_t']
                Gs1.append(-self.params['delta_t'] * J)
                hs1.append(v - [-th, th_y, -th])
                Gs1.append(self.params['delta_t'] * J)
                hs1.append(-v + [th, max(th, th_y) + 1e-6, th])

        # GRF friction cone constraint
        # for all GRF \lambda on each contact point, we have:
        # constraint 1: \sqrt(2) * abs(\lambda_x) < \mu * \lambda_y
        # constraint 2: \sqrt(2) * abs(\lambda_z) < \mu * \lambda_y
        if True:
            if nc > 0:
                Gs2.append(art.math.block_diagonal_matrix_np([self.friction_constraint_matrix] * nc))
                hs2.append(np.zeros(nc * 4))

        # equation of motion (equality constraint)
        # A @ x = [-M | J_c^T | I] @ [qddot | \lambda | \tau]^T
        #       = -M @ qddot + J_c^T @ \lambda + \tau = h, which <=>
        # \tau + J_c^T @ \lambda = M @ qddot + h (paper 3.2.3, equation 7, equation of motion) 
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
            b_ = h

        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        # G_ will be [[Gs1, 0, 0], [0, Gs2, 0], [0, 0, Gs3]]
        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='proxqp', initvals=init)

        if x is None or np.linalg.norm(x) > 10000:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)
    
        qddot = x[:self.model.qdot_size]
        GRF = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]

        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        self.q = q
        self.qdot = qdot
        self.last_x = x

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]

        if calc_rf:
            reaction_force = np.zeros(JointSet.n_full*3, dtype=np.float64)
            for i, joint_name in enumerate(collision_joints):
                joint_id = self.joint_name_to_id[joint_name]
                rf = GRF[i*12:(i+1)*12].reshape(4, 3).sum(axis=0)
                reaction_force[(joint_id-1)*3:joint_id*3] = rf
            return pose_opt, tran_opt, reaction_force

        return pose_opt, tran_opt
    
    
class VAEOptimizer:
    

    def __init__(self):
        mu = 0.6                # static friction coefficient (paper 3.2.3)
        # black box model
        self.model = RBDLModel(Paths.physics_model_file, update_kinematics_by_hand=True)
        self.params = utils.read_debug_param_values_from_json(Paths.physics_parameter_file)
        self.lambda_constraint_matrix = np.array([[np.sqrt(2), -mu, 0], [-np.sqrt(2), -mu, 0],
            [0, -mu, np.sqrt(2)], [0, -mu, -np.sqrt(2)]])
        
        # the names of 24 joints
        self.joint_names = ['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE', 'RANKLE',
            'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER', 'RSHOULDER',
            'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST', 'LHAND', 'RHAND']
        
        # the ids of 24 joints
        self.joint_ids = [vars(Body)[name] for name in self.joint_names]

        # states
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.reset_states()


    def reset_states(self):
        self.last_x = []
        self.q = None
        self.qdot = np.zeros(self.model.qdot_size)
        self.qddot = np.zeros(self.model.qdot_size)


    def optimize_frame(self, pose, jvel):
        # import qpsolvers
        # print(qpsolvers.available_solvers)
        # exit()
        # pose [24, 3, 3], jvel [24, 3].
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        v_ref = jvel.numpy()
        q = self.q
        qdot = self.qdot
        qddot = self.qddot
        
        if q is None:
            self.q = q_ref
            return pose, torch.zeros(3)
        
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = np.empty((JointSet.n_full*3, self.model.qdot_size))
        root_pos = self.model.calc_body_position(q, self.joint_ids[0])
        dis_to_root = np.empty((JointSet.n_full, 3))
        for i, joint_id in enumerate(self.joint_ids[1:]):
            pos = self.model.calc_body_position(q, joint_id)
            dis_to_root[i,:] = pos - root_pos
            pb = self.model.calc_base_to_body_coordinates(q, joint_id, pos)
            Js[i*3:(i+1)*3,:] = self.model.calc_point_Jacobian(q, joint_id, pb)    
        dis_to_root = np.sqrt(np.sum(np.square(dis_to_root), axis=1))

        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], \
            [np.empty((0, JointSet.n_full * 3))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, self.model.qdot_size))], [np.empty(0)], \
            [np.empty((0, JointSet.n_full * 3))], [np.empty(0)], [np.zeros((0, self.model.qdot_size))], [np.empty(0)]
        A_, b_ = None, None

        # joint angle PD controller
        # minimize_{\ddot{\theta}} (0.5*\ddot{\theta} - \ddot{\theta}_{des})^T @ \ddot{\theta}
        # <=> minimize_{\ddot{\theta}} ||\ddot{\theta} - \ddot{\theta}_{des}||^2
        if True:
            A = np.hstack((np.zeros((self.model.qdot_size - 3, 3)), np.eye((self.model.qdot_size - 3))))
            b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72

        # joint position PD controller (using joint velocity to determine target joint position)
        # minimize_{\ddot{q}} (0.5*J@\ddot{q} + \dot{J}@\dot{q} - \ddot{r}_{des})^T @ (J @ \ddot{q})
        # <=> minimize_{\ddot{q}} ||\ddot{r} - \ddot{r}_{des}||^2 (\ddot{r} = J@\ddot{q} + \dot{J}@\dot{q})
        if True:
            # no LHAND, RHAND, LFOOT and RFOOT.
            for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE',
                    'RANKLE', 'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                    'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
                joint_id = vars(Body)[joint_name]
                if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id) # [3, 75]
                # calc_point_acc(q, qdot, 0) = \dot{J} @ \dot{q}
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des    # [3,]
                As1.append(A * self.params['coeff_jvel'])
                bs1.append(b * self.params['coeff_jvel'])
                
        # joints closer to ROOT should have smaller supporting force (i.e. larger penalty)
        if True:
            A = art.math.block_diagonal_matrix_np([np.eye(3) * (1/d) for d in dis_to_root])
            As2.append(A * self.params['coeff_lambda'])
            bs2.append(np.zeros(JointSet.n_full * 3))

        # tau size
        # minimize_{\lambda} k_{res} * norm(\tau[:6])^2 + k_{\tau} * norm(\tau[6:])^2
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual'],
                np.eye(self.model.qdot_size - 6) * self.params['coeff_tau']
            ]))
            bs3.append(np.zeros(self.model.qdot_size))
            
        # restrict qddot based on fixed mask using Gs1.
        # if False:
        #     G = np.zeros((np.sum(fixed_mask)*3, self.model.qdot_size))
        #     i = 0
        #     for j, fixed in enumerate(fixed_mask):
        #         if fixed:
        #             G[i*3:(i+1)*3,(j+1)*3:(j+2)*3] = np.eye(3)
        #             i += 1
        #     Gs1.extend([G, -G])
        #     hs1.append(10 * np.ones(np.sum(fixed_mask)*6))
        
                
        # kinematics constraint on the supporting force
        # the dot product of velocity and supporting force should be near 0.
        if True:
            vs = []
            for joint_id in self.joint_ids[1:]:
                J = self.model.calc_point_Jacobian(q, joint_id)
                # v = self.model.calc_point_velocity(q, qdot, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id) + np.dot(J, qddot) * self.params['delta_t']
                vs.append(v[None,:])
            G = art.math.block_diagonal_matrix_np(vs)
            Gs2.extend([G, -G])
            hs2.append(10 * np.ones(JointSet.n_full * 2))
            
        # lambda should point upward, and the friction force should be inside the friction cone
        if True:
            Gs2.append(art.math.block_diagonal_matrix_np([self.lambda_constraint_matrix] * JointSet.n_full))
            hs2.append(np.zeros(JointSet.n_full * 4))

        # equation of motion (equality constraint)
        # A @ x = [-M | J_c^T | I] @ [qddot | \lambda | \tau]^T
        #       = -M @ qddot + J_c^T @ \lambda + \tau = h, which <=>
        # \tau + J_c^T @ \lambda = M @ qddot + h (paper 3.2.3, equation 7, equation of motion) 
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(self.model.qdot_size)))
            b_ = h

        # construct the qp equation
        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        # G_ will be [[Gs1, 0, 0], [0, Gs2, 0], [0, 0, Gs3]]
        G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        h_ = np.concatenate((hs1, hs2, hs3))
        P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))

        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        # solve_qp(P, q, G, h, A, b, lb, ub)
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver='quadprog', initvals=init)
        if x is None or np.linalg.norm(x) > 1e4:  # 80 times slower
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='cvxopt', initvals=init)
    
        qddot = x[:self.model.qdot_size]
        GRF = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]
        
        if False:   # print left and right foot GRF
            left_foot = GRF[18:21] + GRF[27:30]
            right_foot = GRF[21:24] + GRF[30:33]
            for item in left_foot: print(f'{item:.3f}, ', end='')
            for item in right_foot: print(f'{item:.3f}, ', end='')
            total_y = np.sum(GRF[1::3])
            print(f'{total_y:.3f}')
        
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        
        self.q = q
        self.qdot = qdot
        self.qddot = qddot
        self.last_x = x

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]
        return pose_opt, tran_opt
    
    
    def optimize_frame_sparse(self, pose, jvel, calc_rf:bool=False):
        ''' Use sparse qpsolvers to boost efficiency.
        '''
        # pose [24, 3, 3], jvel [24, 3].
        q_ref = smpl_to_rbdl(pose, torch.zeros(3))[0]
        v_ref = jvel.numpy()
        q = self.q
        qdot = self.qdot
        qddot = self.qddot
        
        if q is None:
            self.q = q_ref
            if calc_rf: return pose, torch.zeros(3), np.zeros(JointSet.n_full*3, dtype=np.float64)
            return pose, torch.zeros(3)
        
        self.model.update_kinematics(q, qdot, np.zeros(self.model.qdot_size))
        Js = np.empty((JointSet.n_full*3, self.model.qdot_size))
        root_pos = self.model.calc_body_position(q, self.joint_ids[0])
        dis_to_root = np.empty((JointSet.n_full, 3))
        for i, joint_id in enumerate(self.joint_ids[1:]):
            pos = self.model.calc_body_position(q, joint_id)
            dis_to_root[i,:] = pos - root_pos
            pb = self.model.calc_base_to_body_coordinates(q, joint_id, pos)
            Js[i*3:(i+1)*3,:] = self.model.calc_point_Jacobian(q, joint_id, pb)    
        dis_to_root = np.sqrt(np.sum(np.square(dis_to_root), axis=1))
        
        # minimize   ||A1 * qddot - b1||^2     for A1, b1 in zip(As1, bs1)
        #            + ||A2 * lambda - b2||^2  for A2, b2 in zip(As2, bs2)
        #            + ||A3 * tau - b3||^2     for A3, b3 in zip(As3, bs3)
        # s.t.       G1 * qddot <= h1          for G1, h1 in zip(Gs1, hs1)
        #            G2 * lambda <= h2         for G2, h2 in zip(Gs2, hs2)
        #            G3 * tau <= h3            for G3, h3 in zip(Gs3, hs3)
        #            A_ * x = b_
        
        # lengths of three matrices
        len1, len2, len3 = self.model.qdot_size, JointSet.n_full * 3, self.model.qdot_size
        As1, bs1, As2, bs2, As3, bs3 = [np.zeros((0, len1))], [np.empty(0)], \
            [np.empty((0, len2))], [np.empty(0)], [np.zeros((0, len3))], [np.empty(0)]
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = [np.zeros((0, len1))], [np.empty(0)], \
            [np.empty((0, len2))], [np.empty(0)], [np.zeros((0, len3))], [np.empty(0)]
        A_, b_ = None, None
        
        # joint angle PD controller
        # minimize_{\ddot{\theta}} (0.5*\ddot{\theta} - \ddot{\theta}_{des})^T @ \ddot{\theta}
        # <=> minimize_{\ddot{\theta}} ||\ddot{\theta} - \ddot{\theta}_{des}||^2
        if True:
            A = np.hstack((np.zeros((len1 - 3, 3)), np.eye((len1 - 3))))
            b = self.params['kp_angular'] * art.math.angle_difference(q_ref[3:], q[3:]) - self.params['kd_angular'] * qdot[3:]
            As1.append(A)  # 72 * 75
            bs1.append(b)  # 72
            
        # joint position PD controller (using joint velocity to determine target joint position)
        # minimize_{\ddot{q}} (0.5*J@\ddot{q} + \dot{J}@\dot{q} - \ddot{r}_{des})^T @ (J @ \ddot{q})
        # <=> minimize_{\ddot{q}} ||\ddot{r} - \ddot{r}_{des}||^2 (\ddot{r} = J@\ddot{q} + \dot{J}@\dot{q})
        if True:
            # no LHAND, RHAND, LFOOT and RFOOT.
            for joint_name, v in zip(['ROOT', 'LHIP', 'RHIP', 'SPINE1', 'LKNEE', 'RKNEE', 'SPINE2', 'LANKLE',
                    'RANKLE', 'SPINE3', 'LFOOT', 'RFOOT', 'NECK', 'LCLAVICLE', 'RCLAVICLE', 'HEAD', 'LSHOULDER',
                    'RSHOULDER', 'LELBOW', 'RELBOW', 'LWRIST', 'RWRIST'], v_ref[:22]):
                joint_id = vars(Body)[joint_name]
                if joint_id == Body.LFOOT or joint_id == Body.RFOOT: continue
                cur_vel = self.model.calc_point_velocity(q, qdot, joint_id)
                a_des = self.params['kp_linear'] * v * self.params['delta_t'] - self.params['kd_linear'] * cur_vel
                A = self.model.calc_point_Jacobian(q, joint_id) # [3, 75]
                # Note: how to understand calc_point_acceleration?
                # let calc_point_acceleration be function f
                # we have \ddot{r} = f(q, \dot{q}, \ddot{q}) (1), which is a general representation
                # at the same time, we have \dot{r} = J @ \dot{q}
                # take derivatives, => \ddot{r} = J @ \ddot{q} + \dot{J} @ \dot{q} (2)
                # (1) + (2) => f(q, \dot{q}, \ddot{q}) = J @ \ddot{q} + \dot{J} @ \dot{q}
                # cancel out \ddot{q} => f(q, \dot{q}, 0) = \dot{J} @ \dot{q}
                # therefore, calc_point_acceletation(q, \dot{q}, 0) = \dot{J} @ \dot{q}
                b = -self.model.calc_point_acceleration(q, qdot, np.zeros(75), joint_id) + a_des    # [3,]
                As1.append(A * self.params['coeff_jvel'])
                bs1.append(b * self.params['coeff_jvel'])
                
        # joints closer to ROOT should have smaller supporting force (i.e. larger penalty)
        if True:
            # coeff_lambda = 0.0  # ablation on dis-to-root principle
            coeff_lambda = self.params['coeff_lambda']
            A = art.math.block_diagonal_matrix_np([np.eye(3) * (1/d) for d in dis_to_root])
            As2.append(A * coeff_lambda)
            bs2.append(np.zeros(JointSet.n_full * 3))
            
        # tau size
        # minimize_{\lambda} k_{res} * norm(\tau[:6])^2 + k_{\tau} * norm(\tau[6:])^2
        if True:
            As3.append(art.math.block_diagonal_matrix_np([
                np.eye(6) * self.params['coeff_virtual'],
                np.eye(len3 - 6) * self.params['coeff_tau']
            ]))
            bs3.append(np.zeros(len3))
            
        # restrict qddot based on fixed mask using Gs1.
        # if False:
        #     G = np.zeros((np.sum(fixed_mask)*3, self.model.qdot_size))
        #     i = 0
        #     for j, fixed in enumerate(fixed_mask):
        #         if fixed:
        #             G[i*3:(i+1)*3,(j+1)*3:(j+2)*3] = np.eye(3)
        #             i += 1
        #     Gs1.extend([G, -G])
        #     hs1.append(10 * np.ones(np.sum(fixed_mask)*6))
                
        # kinematics constraint on the supporting force
        # the dot product of velocity and supporting force should be near 0.
        if True:   # ablation on stationary support constraint
            vs = []
            for joint_id in self.joint_ids[1:]:
                J = self.model.calc_point_Jacobian(q, joint_id)
                # v = self.model.calc_point_velocity(q, qdot, joint_id)
                v = self.model.calc_point_velocity(q, qdot, joint_id) + np.dot(J, qddot) * self.params['delta_t']
                vs.append(v[None,:])
            G = art.math.block_diagonal_matrix_np(vs)
            Gs2.extend([G, -G])
            hs2.append(10 * np.ones(JointSet.n_full * 2))
            
        # lambda should point upward, and the friction force should be inside the friction cone
        if True:
            Gs2.append(art.math.block_diagonal_matrix_np([self.lambda_constraint_matrix] * JointSet.n_full))
            hs2.append(np.zeros(JointSet.n_full * 4))
            
        # equation of motion (equality constraint)
        # A @ x = [-M | J_c^T | I] @ [qddot | \lambda | \tau]^T
        #       = -M @ qddot + J_c^T @ \lambda + \tau = h, which <=>
        # \tau + J_c^T @ \lambda = M @ qddot + h (paper 3.2.3, equation 7, equation of motion) 
        if True:
            M = self.model.calc_M(q)
            h = self.model.calc_h(q, qdot)
            A_ = np.hstack((-M, Js.T, np.eye(len3)))
            b_ = h
            
        # construct the qp equation
        As1, bs1, As2, bs2, As3, bs3 = np.vstack(As1), np.concatenate(bs1), np.vstack(As2), np.concatenate(bs2), np.vstack(As3), np.concatenate(bs3)
        Gs1, hs1, Gs2, hs2, Gs3, hs3 = np.vstack(Gs1), np.concatenate(hs1), np.vstack(Gs2), np.concatenate(hs2), np.vstack(Gs3), np.concatenate(hs3)
        
        # construct P_
        # P_ = art.math.block_diagonal_matrix_np([np.dot(As1.T, As1), np.dot(As2.T, As2), np.dot(As3.T, As3)])
        As1_dot = np.dot(As1.T, As1)    # dense
        As2_dot = np.dot(As2.T, As2)    # diagnal
        As3_dot = np.dot(As3.T, As3)    # diagnal
        nonzero_mask = (As1_dot != 0.0)
        data = np.concatenate([As1_dot[nonzero_mask], np.diag(As2_dot), np.diag(As3_dot)])
        indices = np.tile(np.arange(len1), len1).reshape(len1, len1)
        indices = np.concatenate([indices[nonzero_mask], np.arange(len2+len3)+len1])
        idxptr = np.concatenate([np.sum(nonzero_mask, axis=0), np.ones(len2+len3, dtype=int)])
        idxptr = np.concatenate([np.zeros(1, dtype=int), np.cumsum(idxptr)])
        P_ = sparse.csc_array((data, indices, idxptr), shape=(len1+len2+len3,len1+len2+len3))
        
        # construct G_
        # G_ = art.math.block_diagonal_matrix_np([Gs1, Gs2, Gs3])
        nonzero_mask = (Gs2.T != 0.0)
        data = Gs2.T[nonzero_mask]
        indices = np.tile(np.arange(Gs2.shape[0]), len2).reshape(len2, Gs2.shape[0])
        indices = indices[nonzero_mask]
        idxptr = np.concatenate([np.zeros(len1+1, dtype=int), np.sum(nonzero_mask, axis=1), np.zeros(len3, dtype=int)])
        idxptr = np.cumsum(idxptr)
        G_ = sparse.csc_array((data, indices, idxptr), shape=(Gs2.shape[0],len1+len2+len3))
        
        # construct A_
        A_ = sparse.csc_array(A_)
                
        h_ = np.concatenate((hs1, hs2, hs3))
        q_ = np.concatenate((-np.dot(As1.T, bs1), -np.dot(As2.T, bs2), -np.dot(As3.T, bs3)))
        
        # fast solvers are less accurate/robust, and may fail
        init = self.last_x if len(self.last_x) == len(q_) else None
        # solve_qp(P, q, G, h, A, b, lb, ub)
        # Bug fix: if init is None, proxqp may stuck
        solver = 'clarabel' if init is None else 'proxqp'
        x = solve_qp(P_, q_, G_, h_, A_, b_, solver=solver, initvals=init)
        if x is None or np.linalg.norm(x) > 1e4:
            x = solve_qp(P_, q_, G_, h_, A_, b_, solver='clarabel', initvals=init)
            
        # after solve qp, x may be None, may trigger NoneType error here.
        qddot = x[:self.model.qdot_size]
        reaction_force = x[self.model.qdot_size:-self.model.qdot_size]
        tau = x[-self.model.qdot_size:]
        
        if False:   # print left and right foot GRF
            GRF = reaction_force
            left_foot = GRF[18:21] + GRF[27:30]
            right_foot = GRF[21:24] + GRF[30:33]
            for item in left_foot: print(f'{item:.3f}, ', end='')
            for item in right_foot: print(f'{item:.3f}, ', end='')
            total_y = np.sum(GRF[1::3])
            print(f'{total_y:.3f}')
        
        qdot = qdot + qddot * self.params['delta_t']
        q = q + qdot * self.params['delta_t']
        
        self.q = q
        self.qdot = qdot
        self.qddot = qddot
        self.last_x = x

        pose_opt, tran_opt = rbdl_to_smpl(q)
        pose_opt = torch.from_numpy(pose_opt).float()[0]
        tran_opt = torch.from_numpy(tran_opt).float()[0]

        if calc_rf: return pose_opt, tran_opt, reaction_force
        return pose_opt, tran_opt
    
    
if __name__ == '__main__':
    optimizer = VAEOptimizer()
    print(optimizer.model.qdot_size)

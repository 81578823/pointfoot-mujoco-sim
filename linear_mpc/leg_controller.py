import os
import sys
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from robot_data import RobotData

class LegController():
    '''Compute torques for each joint.

    For stance legs, joint torques are computed with $ tau_i = Jv.T @ -f_i $, 
    where Jv (3 x 3 matrix) is the foot Jacobian, and f_i is the contact force 
    related to foot i, calculated by the predictive controller. All the 
    quantities are expressed in world frame.

    For swing legs, joint torques are computed by a PD controller (by 
    assuming the mass of the leg is not too large) to track the given swing 
    foot targets. All the quantities are expressed in world frame.

    Params
    ------
    Kp_swing: np.ndarray
        The P gain for swing leg PD control.
    Kd_swing: np.ndarray
        The D gain for swing leg PD control.
    '''
    def __init__(self, Kp_swing: np.ndarray, Kd_swing: np.ndarray):
        self.__Kp_swing = Kp_swing
        self.__Kd_swing = Kd_swing
        self.__torque_cmds = np.zeros(6, dtype=np.float32)
        self.alpha = 0.2
        self.fist_run=True

    @property
    def torque_cmds(self) -> np.ndarray:
        return self.__torque_cmds

    def update(
        self, 
        robot_data: RobotData, 
        contact_forces: np.ndarray, 
        swing_states: List[int],
        pos_targets_swingfeet: np.ndarray,
        vel_targets_swingfeet: np.ndarray
    ):
        '''Update joint torques using current data.

        Args
        ----
        robot_data: RobotData
            records current robot data. You need to update the current robot 
            data before computing the joint torques.
        contact_forces: np.ndarray, shape = (6, )
            contact forces of each foot expressed in world frame, computed 
            by the predictive controller.
        swing_states: List[int]
            identify whether each leg is in swing (=1) or stance (=0).
        pos_targets_swingfeet: np.ndarray, shape = (2, 3)
            target position of each swing foot relative to base, expressed in 
            base frame.
        vel_targets_swingfeet: np.ndarray, shape = (2, 3)
            target velocity of each swing foot relative to base, expressed in 
            base frame.

        Returns
        -------
        torque_cmds: np.ndarray, shape = (6, )
            torque commands of each joint.
        '''
        if self.fist_run:
            self.pos_targets_filtered = pos_targets_swingfeet
            self.vel_targets_filtered = vel_targets_swingfeet
            self.vel_error_prev=np.zeros((2,3))
            self.contact_forces_filtered = contact_forces
            self.__torque_cmds = np.zeros(3 * 2, dtype=np.float32)
            self.lambda_transition = np.zeros(2, dtype=np.float32)
            self.fist_run = False
        
        Jv_feet = robot_data.Jv_feet
        R_base = robot_data.R_base
        base_vel_base_feet = robot_data.base_vel_base_feet
        base_pos_base_feet = robot_data.base_pos_base_feet

        for leg_idx in range(2):
            Jvi = Jv_feet[leg_idx]
            
            # 平滑目标位置和速度
            self.pos_targets_filtered[leg_idx] = self.alpha * self.pos_targets_filtered[leg_idx] + (1 - self.alpha) * pos_targets_swingfeet[leg_idx]
            self.vel_targets_filtered[leg_idx] = self.alpha * self.vel_targets_filtered[leg_idx] + (1 - self.alpha) * vel_targets_swingfeet[leg_idx]
            
            if swing_states[leg_idx]:
                # base_pos_swingfoot_des = pos_targets_swingfeet[leg_idx, :]
                # base_vel_swingfoot_des = vel_targets_swingfeet[leg_idx, :]
                
                pos_err= R_base @ self.pos_targets_filtered[leg_idx] - R_base @ base_pos_base_feet[leg_idx]
                vel_err = R_base @ self.vel_targets_filtered[leg_idx] - R_base @ base_vel_base_feet[leg_idx]
                
                # 平滑微分项
                vel_error_filtered = self.alpha * self.vel_error_prev[leg_idx] + (1 - self.alpha) * vel_err
                self.vel_error_prev[leg_idx] = vel_error_filtered
                
                swing_err = self.__Kp_swing @ pos_err\
                    + self.__Kd_swing @ vel_error_filtered              
                # tau_i = Jvi.T @ swing_err
                # print("tau_i: ", tau_i)
                # cmd_i = tau_i[3*leg_idx : 3*(leg_idx+1)]
                # self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] = cmd_i
            else:
                # 支撑状态的接触力控制
                contact_forces_filtered = self.alpha * self.contact_forces_filtered[3*leg_idx:3*(leg_idx+1)] + \
                                        (1 - self.alpha) * contact_forces[3*leg_idx:3*(leg_idx+1)]
                self.contact_forces_filtered[3*leg_idx:3*(leg_idx+1)] = contact_forces_filtered

                # tau_i = Jvi.T @ -contact_forces_filtered
                # cmd_i = tau_i[3*leg_idx:3*(leg_idx+1)]
                # self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] = cmd_i
            
            # 状态平滑切换
            transition_rate = 0.1
            self.lambda_transition[leg_idx] = min(1.0, max(0.0, self.lambda_transition[leg_idx] + transition_rate if swing_states[leg_idx] else -transition_rate))
            # 混合摆动和支撑的力矩
            tau_swing = Jvi.T @ swing_err if swing_states[leg_idx] else np.zeros(6,)
            tau_stance = Jvi.T @ -contact_forces_filtered if not swing_states[leg_idx] else np.zeros(6,)
            tau_i = self.lambda_transition[leg_idx] * tau_swing[3*leg_idx:3*(leg_idx+1)] + (1 - self.lambda_transition[leg_idx]) * tau_stance[3*leg_idx:3*(leg_idx+1)]

            # 输出平滑扭矩
            self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] = self.alpha * self.__torque_cmds[3*leg_idx:3*(leg_idx+1)] + (1 - self.alpha) * tau_i   

        return self.__torque_cmds
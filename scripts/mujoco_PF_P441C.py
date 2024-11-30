import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../linear_mpc'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/'))
# os.environ['MUJOCO_GL'] = 'nvidia'

import mujoco
import mujoco.viewer as Viewer
import numpy as np

from gait import Gait
from leg_controller import LegController
from linear_mpc_configs import LinearMpcConfig
from mpc import ModelPredictiveController
from robot_configs import PF_P441CConfig
from robot_data import RobotData
from swing_foot_trajectory_generator import SwingFootTrajectoryGenerator


STATE_ESTIMATION = False

def reset(data, robot_config):
    print("data.qpos: ", data.qpos)
    data.qpos[:] = np.array([
        0, 0, robot_config.base_height_des,
        1, 0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ])
    
    data.qvel[:] = np.zeros(12)
    mujoco.mj_forward(data.model, data)  # Apply changes and reinitialize simulation


def get_true_simulation_data(model, data):
    pos_base = data.xpos[1]  # Base position
    vel_base = data.cvel[1][:3]  # Base linear velocity
    quat_base = data.sensordata[0:4]
    omega_base = data.sensordata[4:7]
    pos_joint = data.sensordata[10:16] 
    vel_joint = data.sensordata[16:22] 
    touch_state = data.sensordata[22:24]
    
    # 获取 foot_L_Link 和 foot_R_Link 的几何体索引
    geom_id_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_L_collison')
    geom_id_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'foot_R_collison')

    # 获取几何体附属的身体索引
    # body_id_L = model.geom_bodyid[geom_id_L]
    # body_id_R = model.geom_bodyid[geom_id_R]
    
    # 初始化速度数组
    vel_L =np.zeros(6)
    vel_R =np.zeros(6)

    # 获取几何体的线速度和角速度
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id_L, vel_L, True)  
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id_R, vel_R, True)  

     
    # 获取 hip_L_Link 和 hip_R_Link 的几何体索引
    hip_id_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'hip_L_Link')
    hip_id_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'hip_R_Link')

    
    # 获取几何体的位置
    pos_foothold = [
        data.geom_xpos[geom_id_L],
        data.geom_xpos[geom_id_R]
    ]

    vel_foothold = [
        vel_L[:3],
        vel_R[:3]
    ]
    
    # 获取大腿位置
    pos_thigh = [
        data.xpos[hip_id_L],
        data.xpos[hip_id_R]
    ]


    true_simulation_data = [
        pos_base, 
        vel_base, 
        quat_base, 
        omega_base, 
        pos_joint,  
        vel_joint, 
        touch_state,  
        pos_foothold, 
        vel_foothold, 
        pos_thigh
    ]
    return true_simulation_data

def get_simulated_sensor_data(sim):
    imu_quat = sim.sensordata[0:4]
    imu_gyro = sim.sensordata[4:7]
    imu_accelerometer = sim.sensordata[7:10]
    pos_joint = sim.sensordata[10:16]
    vel_joint = sim.sensordata[16:22]
    touch_state = sim.sensordata[22:24]
    print("imu_quat: ", imu_quat)
    simulated_sensor_data = [
        imu_quat, 
        imu_gyro, 
        imu_accelerometer, 
        pos_joint, 
        vel_joint, 
        touch_state
        ]
    print("simulated_sensor_data: ", simulated_sensor_data)
    return simulated_sensor_data

paused = False
reset_flag = False

def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused
    elif keycode == 8:
        reset_flag = True

def main():
    global reset_flag
    cur_path = os.path.dirname(__file__)
    mujoco_xml_path = os.path.join(cur_path, '../PF_P441C/xml/robot.xml')
    model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    data_ = mujoco.MjData(model)
    viewer = Viewer.launch_passive(model, data_,key_callback=key_callback)
    start = time.time()
    
    while viewer.is_running() and time.time() - start < 30:
        robot_config = PF_P441CConfig

        reset(data_, robot_config)
        mujoco.mj_forward(model, data_)

        urdf_path = os.path.join(cur_path, '../PF_P441C/urdf/robot.urdf')
        robot_data = RobotData(urdf_path, state_estimation=STATE_ESTIMATION)
        # initialize_robot(sim, viewer, robot_config, robot_data)

        predictive_controller = ModelPredictiveController(LinearMpcConfig, PF_P441CConfig)
        leg_controller = LegController(robot_config.Kp_swing, robot_config.Kd_swing)

        gait = Gait.STANDING
        swing_foot_trajs = [SwingFootTrajectoryGenerator(leg_idx) for leg_idx in range(2)]

        vel_base_des = np.array([0.5, 0., 0.])
        yaw_turn_rate_des = 0.

        iter_counter = 0

        while True:

            if not STATE_ESTIMATION:
                data = get_true_simulation_data(model,data_)
            else:
                data = get_simulated_sensor_data(data_)
                
            print("data: ", data)
            print("data.shape: ", len(data))

            robot_data.update(
                pos_base=data[0],
                lin_vel_base=data[1],
                quat_base=data[2],
                ang_vel_base=data[3],
                q=data[4],
                qdot=data[5]
            )

            gait.set_iteration(predictive_controller.iterations_between_mpc, iter_counter)
            swing_states = gait.get_swing_state()
            gait_table = gait.get_gait_table()

            predictive_controller.update_robot_state(robot_data)

            contact_forces = predictive_controller.update_mpc_if_needed(iter_counter, vel_base_des, 
                yaw_turn_rate_des, gait_table, solver='drake', debug=False, iter_debug=0) 

            pos_targets_swingfeet = np.zeros((2, 3))
            vel_targets_swingfeet = np.zeros((2, 3))

            for leg_idx in range(2):
                if swing_states[leg_idx] > 0:   # leg is in swing state
                    swing_foot_trajs[leg_idx].set_foot_placement(
                        robot_data, gait, vel_base_des, yaw_turn_rate_des
                    )
                    base_pos_base_swingfoot_des, base_vel_base_swingfoot_des = \
                        swing_foot_trajs[leg_idx].compute_traj_swingfoot(
                            robot_data, gait
                        )
                    pos_targets_swingfeet[leg_idx, :] = base_pos_base_swingfoot_des
                    vel_targets_swingfeet[leg_idx, :] = base_vel_base_swingfoot_des

            torque_cmds = leg_controller.update(robot_data, contact_forces, swing_states, pos_targets_swingfeet, vel_targets_swingfeet)
            data_.ctrl[:] = torque_cmds

            if not paused:
                viewer.cam.lookat = data_.body('base_Link').subtree_com
                mujoco.mj_step(model, data_)            
                viewer.sync()
                iter_counter += 1
                
            if reset_flag:
                iter_counter = 0
                reset_flag=False
                reset(data_, robot_config)

                
            if iter_counter == 50000:
                reset(data_, robot_config)
                iter_counter = 0
                break

        
if __name__ == '__main__':
    main()
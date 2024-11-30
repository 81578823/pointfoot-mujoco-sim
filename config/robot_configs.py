import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

import numpy as np

from dynamics import make_com_inertial_matrix

class RobotConfig:
    mass_base: float
    base_height_des: float
    base_inertia_base: np.ndarray

    fz_max: float
    
    swing_height: float
    Kp_swing: np.ndarray
    Kd_swing: np.ndarray


class PF_P441CConfig(RobotConfig):
    mass_base: float = 9.567
    base_height_des: float = 0.65
    # 惯性矩参数，基于URDF中`base_Link`的惯性
    base_inertia_base = make_com_inertial_matrix(
        ixx=0.136700555,  # URDF中的惯性矩值
        ixy=-0.000114115,
        ixz=0.029870339,
        iyy=0.111197923,
        iyz=0.000177889,
        izz=0.098252391
    )

    fz_max = 500.

    swing_height = 0.1
    Kp_swing = np.diag([200., 200., 200.])
    Kd_swing = np.diag([20., 20., 20.])

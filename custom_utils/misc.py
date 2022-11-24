import math
import numpy as np
from .compute_points import compute_points


TASK_OFFSETS = {
    'pickup_object': np.array([[-108, 108], [0, 90], [-108, 108]], dtype=float),
    'reorient_object': np.array([[-90, 90], [0, 90], [-90, 90]], dtype=float),
    'open_drawer': np.array([[-108, 108], [0, 130], [-108, 108]], dtype=float),
    'close_drawer': np.array([[-108, 108], [0, 130], [-108, 108]], dtype=float),
}

TASK_RESOLUTIONS = {
    'pickup_object': 384,
    'reorient_object': 320,
    'open_drawer': 384,
    'close_drawer': 384,
}


def collate_fn(batch):
    return batch


def action_diff(curr_act, prev_act):
    curr1, curr2 = curr_act
    prev1, prev2 = prev_act
    if np.linalg.norm(curr1-prev1) + np.linalg.norm(curr2-prev2) > 1e-2:
        return True
    else:
        return False


def get_pose_world(trans_rel, rot_rel, robot_pos, robot_rot):
    if rot_rel is not None:
        rot = robot_rot @ rot_rel
    else:
        rot = None

    if trans_rel is not None:
        trans = robot_rot @ trans_rel + robot_pos
    else:
        trans = None

    return trans, rot


def get_pose_relat(trans, rot, robot_pos, robot_rot):
    inv_rob_rot = robot_rot.T

    if trans is not None:
        trans_rel = inv_rob_rot @ (trans - robot_pos)
    else:
        trans_rel = None

    if rot is not None:
        rot_rel = inv_rob_rot @ rot
    else:
        rot_rel = None
    
    return trans_rel, rot_rel


def quat_mul(q0, q1):
    # wxyz
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = np.array([w, x, y, z])
    if q.ndim == 2:
        q = q.swapaxes(0, 1)
    assert q.shape == q0.shape
    return q


def euler_angles_to_quat(euler_angles: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert Euler XYZ angles to quaternion. Adapted from omni.isaac.core

    Args:
        euler_angles (np.ndarray):  Euler XYZ angles.
        degrees (bool, optional): Whether input angles are in degrees. Defaults to False.

    Returns:
        np.ndarray: quaternion (w, x, y, z).
    """
    roll, pitch, yaw = euler_angles
    if degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    w = (cr * cp * cy) + (sr * sp * sy)
    x = (sr * cp * cy) - (cr * sp * sy)
    y = (cr * sp * cy) + (sr * cp * sy)
    z = (cr * cp * sy) - (sr * sp * cy)
    return np.array([w, x, y, z])


def create_pcd_hardcode(camera, depth, cm_to_m=True):
    height = camera['resolution']['height']
    width =  camera['resolution']['width']
    focal_length = camera['focal_length'] 
    horiz_aperture = camera['horizontal_aperture']
    vert_aperture = height / width * horiz_aperture
    
    fx = width * focal_length / horiz_aperture
    fy = height * focal_length / vert_aperture
    
    cx = cy = height/2

    points_cam = compute_points(height, width, depth, fx, fy, cx, cy)
    
    T = camera['pose'].T
    Rotation = T[:3, :3]
    t = T[:3, 3]
    points_world = points_cam @ np.transpose(Rotation) + t
    points_world = np.array(points_world).reshape(height, width,3)
    # swap y and z to make z upward
    points_world[:,:,[1,2]] = points_world[:,:,[2,1]]
    
    return points_world / 100 if cm_to_m else points_world


def get_bounds(robot_base, offset, cm_to_m=True):
    # offset shape: [3, 2]
    robot_base = robot_base.reshape(3, 1)
    bounds = robot_base + offset
    bounds[[1,2]] = bounds[[2,1]]
    return bounds / 100 if cm_to_m else bounds


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_rgb: np.ndarray,
                 left_depth: np.ndarray,
                #  left_mask: np.ndarray,
                 left_point_cloud: np.ndarray,
                 base_rgb: np.ndarray,
                 base_depth: np.ndarray,
                #  base_mask: np.ndarray,
                 base_point_cloud: np.ndarray,
                #  overhead_rgb: np.ndarray,
                #  overhead_depth: np.ndarray,
                #  overhead_mask: np.ndarray,
                #  overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                #  wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,

                 wrist_bottom_rgb: np.ndarray,
                 wrist_bottom_depth: np.ndarray,
                #  wrist_bottom_mask: np.ndarray,
                 wrist_bottom_point_cloud: np.ndarray,

                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                #  front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                joint_velocities: np.ndarray,
                joint_positions: np.ndarray,
                #  joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                #  gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                #  gripper_touch_forces: np.ndarray,
                #  task_low_dim_state: np.ndarray,
                #  ignore_collisions: np.ndarray,
                 misc: dict
                 ):
        self.left_rgb = left_rgb
        self.left_depth = left_depth
        # self.left_mask = left_mask
        self.left_point_cloud = left_point_cloud

        self.base_rgb = base_rgb
        self.base_depth = base_depth
        # self.base_mask = base_mask
        self.base_point_cloud = base_point_cloud
        # self.overhead_rgb = overhead_rgb
        # self.overhead_depth = overhead_depth
        # self.overhead_mask = overhead_mask
        # self.overhead_point_cloud = overhead_point_cloud

        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        # self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud

        self.wrist_bottom_rgb = wrist_bottom_rgb
        self.wrist_bottom_depth = wrist_bottom_depth
        # self.wrist_bottom_mask = wrist_bottom_mask
        self.wrist_bottom_point_cloud = wrist_bottom_point_cloud

        self.front_rgb = front_rgb
        self.front_depth = front_depth
        # self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = None
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = None
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = None
        # self.task_low_dim_state = task_low_dim_state
        self.ignore_collisions = np.array(0)
        
        self.misc = misc


CAMERAS = ['front', 'left', 'base', 'wrist', 'wrist_bottom']
def get_obs(franka, cspace_controller, gt, device, time_step, type='rgb'):
    # added: convert xzy to xyz
    from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
    def get_ee(cspace_controller):
        ee_pos, ee_rot = cspace_controller.get_motion_policy().get_end_effector_pose(
                        cspace_controller.get_articulation_motion_policy().get_active_joints_subset().get_joint_positions()
                    )
        ee_rot = rot_matrices_to_quats(ee_rot)
        return (ee_pos, ee_rot)

    obs = {}
    misc = {}

    robot_base = franka.get_world_pose()
    # robot_base_pos = robot_base[0] / 100.0   # cm to m
    robot_base_pos = robot_base[0]
    robot_base_rot = robot_base[1][[1,2,3,0]]   # wxyz to xyzw
    misc['robot_base'] = (robot_base_pos, robot_base_rot)

    position_rotation_world = get_ee(cspace_controller)
    # gripper_pose_trans = position_rotation_world[0] / 100.0   # cm to m
    gripper_pose_trans = position_rotation_world[0]
    quat = position_rotation_world[1][[1,2,3,0]]   # wxyz to xyzw
    gripper_pose = [*gripper_pose_trans, *quat.tolist()]

    gripper_joint_positions = franka.gripper.get_joint_positions()

    for camera_idx in [0,1,2,3,4]:
        if type == 'rgb':
            rgb = gt['images'][camera_idx]['rgb'][:,:,:3]
        elif type == 'mask':
            rgb = gt['images'][camera_idx]['semanticSegmentation'][:,:,np.newaxis].repeat(3,-1) * 50
        else:
            raise ValueError('observation type should be either rgb or mask')
        
        depth = gt['images'][camera_idx]['depthLinear']
        camera = gt['images'][camera_idx]['camera']
        point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
        obs[CAMERAS[camera_idx]+'_rgb'] = rgb
        obs[CAMERAS[camera_idx]+'_depth'] = depth
        obs[CAMERAS[camera_idx]+'_point_cloud'] = point_cloud
    
    gripper_open = gripper_joint_positions[0] > 3.9 and gripper_joint_positions[1] > 3.9

    ob = Observation(
        left_rgb=obs['left_rgb'], 
        left_depth=obs['left_depth'],
        left_point_cloud=obs['left_point_cloud'],
        front_rgb=obs['front_rgb'],
        front_depth=obs['front_depth'],
        front_point_cloud=obs['front_point_cloud'],
        base_rgb=obs['base_rgb'],
        base_depth=obs['base_depth'],
        base_point_cloud=obs['base_point_cloud'],
        wrist_rgb=obs['wrist_rgb'],
        wrist_depth=obs['wrist_depth'],
        wrist_point_cloud=obs['wrist_point_cloud'],
        wrist_bottom_rgb=obs['wrist_bottom_rgb'],
        wrist_bottom_depth=obs['wrist_bottom_depth'],
        wrist_bottom_point_cloud=obs['wrist_bottom_point_cloud'],
        gripper_open=gripper_open,
        gripper_pose=gripper_pose,
        gripper_joint_positions = gripper_joint_positions,
        misc=misc,
        joint_positions= franka.get_joint_positions(),
        joint_velocities= franka.get_joint_velocities(),
    )
    
    return ob

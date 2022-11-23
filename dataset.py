import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from cliport6d.utils.utils import get_fused_heightmap
from custom_utils.misc import get_pose_world, create_pcd_hardcode, get_bounds, TASK_OFFSETS

pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL


class ArnoldDataset(Dataset):
    def __init__(self, data_path, task, obs_type='rgb'):
        """
        Dataset structure: {
            object_id: {
                'act1': List,
                'act2': List
            }
        }
        """
        super().__init__()
        self.data_path = data_path
        self.task = task
        self.pixel_size = 5.625e-3
        self.obs_type = obs_type
        self.task_offset = TASK_OFFSETS[task]
        self.sample_weights = [0.2, 0.8]
        self.episode_dict = {}
        self._load_keyframes()
    
    def _load_keyframes(self):
        """
        Generally, there are 4 frames in each demonstration:
            0: initial state
            1: pre-grasping state
            2: grasping state
            3: final goal state
        In this setting, two samples are extracted for training:
            - observation at initial state (0) as visual input, gripper state at grasping state (2) as action label
            - observation at grasping state (2) as visual input, gripper state at final goal state (3) as action label
        For tasks involving water, the difference is as follows:
            3: to the position before tilting
            4: to the orientation before reverting the cup
            5: final upwards state
            - combination of position at frame 3 and orientation at frame 4 as action label
        All template actions are in world frame.
        """
        for fname in os.listdir(self.data_path):
            if fname.endswith('npz'):
                obj_id = fname.split('-')[2]
                if obj_id not in self.episode_dict:
                    self.episode_dict.update({
                        obj_id: {
                            'act1': [],
                            'act2': []
                        }
                    })
                
                gt_frames = np.load(os.path.join(self.data_path, fname), allow_pickle=True)['gt']
                language_instructions = gt_frames[0]['instruction']

                # pick phase
                step = gt_frames[0]
                bounds = get_bounds(step['robot_base'][0], offset=self.task_offset)

                cmap, hmap = self.get_step_obs(step, bounds, self.pixel_size, type=self.obs_type)
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)

                obj_pos = gt_frames[2]['position_rotation_world'][0] / 100

                act_pos, act_rot = gt_frames[2]['position_rotation_world']
                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)
                
                episode_dict1 = {
                    'img': img,   # [H, W, 6]
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "language": language_instructions,   # str
                    "bounds": bounds,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                }

                self.episode_dict[obj_id]['act1'].append(episode_dict1)

                # place phase
                step = gt_frames[2]
                bounds = get_bounds(step['robot_base'][0], offset=self.task_offset)

                cmap, hmap = self.get_step_obs(step, bounds, self.pixel_size, type=self.obs_type)
                hmap = np.tile(hmap[..., None], (1,1,3))
                img = np.concatenate([cmap, hmap], axis=-1)
                
                if self.task in ['pour_water', 'transfer_water']:
                    # water, compose actions of two frames
                    act_pos = gt_frames[3]['position_rotation_world'][0]
                    act_rot = gt_frames[4]['position_rotation_world'][1]
                else:
                    # default
                    act_pos, act_rot = gt_frames[3]['position_rotation_world']
                
                act_pos /= 100
                act_rot = act_rot[[1,2,3,0]]   # wxyz to xyzw
                target_points = self.get_act_label_from_abs(pos_abs=act_pos, rot_abs=act_rot)

                episode_dict2 = {
                    'img': img,   # [H, W, 6]
                    "attention_points": obj_pos,   # [3,]
                    "target_points": target_points,   # [6,]
                    "language": language_instructions,   # str
                    "bounds": bounds,   # [3, 2]
                    "pixel_size": self.pixel_size,   # scalar
                }

                self.episode_dict[obj_id]['act2'].append(episode_dict2)
        
        print(f'Loaded {[len(v["act1"]) for k,v in self.episode_dict]} demos')

    def __len__(self):
        num_demos = 0
        for k, v in self.episode_dict:
            num_demos += len(v['act'])
        return num_demos
    
    def __getitem__(self, index):
        obj_idx = np.random.randint(len(self.episode_dict), size=1)
        act_idx = 1 + np.random.choice(2, size=1, p=self.sample_weights)
        return random.choice(self.episode_dict[obj_idx][f'act{act_idx}'])

    def get_step_obs(self, step, bounds, pixel_size, type='rgb'):
        imgs = step['images']
        colors = []
        pcds = []
        for camera_obs in imgs:
            camera = camera_obs['camera']

            if type == 'rgb':
                color = camera_obs['rgb'][:, :, :3]
            elif type == 'mask':
                color = camera_obs['semanticSegmentation'][:,:,np.newaxis].repeat(3,-1) * 50
            else:
                raise ValueError('observation type should be either rgb or mask')
            colors.append(color)

            depth = camera_obs['depthLinear']

            point_cloud = create_pcd_hardcode(camera, depth, cm_to_m=True)
            pcds.append(point_cloud)
        
        cmap, hmap = get_fused_heightmap(colors, pcds, bounds, pixel_size)
        return cmap, hmap
    
    def get_act_label_from_rel(self, pos_rel, rot_rel, robot_base):
        robot_pos, robot_rot = robot_base
        robot_rot = R.from_quat(robot_rot).as_matrix()
        pos_world, rot_world = get_pose_world(pos_rel, rot_rel, robot_pos, robot_rot)
        if rot_world is None:
            return pos_world
        else:
            rot_world = R.from_matrix(rot_world).as_euler('zyx', degrees=True)
            return np.concatenate([pos_world, rot_world])
    
    def get_act_label_from_abs(self, pos_abs, rot_abs):
        if rot_abs is None:
            return pos_abs
        else:
            rot_abs = R.from_quat(rot_abs).as_euler('zyx', degrees=True)
            return np.concatenate([pos_abs, rot_abs])

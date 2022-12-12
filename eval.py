"""
For example, run:
    python eval.py --data_dir /mnt/huangjiangyong/VRKitchen/pickup_object --task pickup_object --agent peract --lang_encoder clip --obs_type rgb \
                   --use_gt 0 0 --visualize 0 --checkpoint_path /mnt/huangjiangyong/VRKitchen/pickup_object/ckpt_peract/peract_pickup_object_rgb_clip_best.pth.pth
"""

import os
import sys
import copy
import torch
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from dataset import InstructionEmbedding
from cliport6d.agent import TwoStreamClipLingUNetLatTransporterAgent
from train_peract import create_agent, create_lang_encoder
from peract.utils import get_obs_batch_dict
from custom_utils.misc import get_obs, get_pose_relat, get_pose_world, TASK_OFFSETS

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, '../'))   # NewPipeline root
sys.path.insert(0, os.path.join(CURRENT_DIR, '../../'))   # VRKitchen root
from utils.runner_utils import get_simulation
from motion_planning.common_utils import rotation_reached, position_reached


def load_data(data_path):
    demos = list(Path(data_path).iterdir())
    demo_path = sorted([str(item) for item in demos if not item.is_dir()])
    data = []

    for npz_path in demo_path:
        data.append(np.load(npz_path, allow_pickle=True))
    return data

def load_scene(data, scene_loader, simulation_app, scene_properties, use_gpu_physics):
    from SceneLoader import SceneLoader
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    if len(data) > 0:
        npz = data.pop(0)
        info = npz['info'].item()

        scene_parameters = info['scene_parameters']
        scene_parameters[0].usd_path = os.path.abspath(scene_parameters[0].usd_path).split(os.path.sep)
        path_idx = scene_parameters[0].usd_path.index('VRKitchen2.0')
        scene_parameters[0].usd_path = os.path.join('/home/huangjiangyong/repo', os.path.sep.join(scene_parameters[0].usd_path[path_idx:]))
        
        # scene_parameters[0].traj_dir = os.path.abspath(scene_parameters[0].traj_dir).split(os.path.sep)
        # path_idx = scene_parameters[0].traj_dir.index('VRKitchen2.0')
        # scene_parameters[0].traj_dir = os.path.join('/home/huangjiangyong/repo', os.path.sep.join(scene_parameters[0].traj_dir[path_idx:]))

        robot_parameters = info['robot_parameters']
        robot_parameters[0].usd_path = os.path.abspath(robot_parameters[0].usd_path).split(os.path.sep)
        path_idx = robot_parameters[0].usd_path.index('VRKitchen2.0')
        robot_parameters[0].usd_path = os.path.join('/home/huangjiangyong/repo', os.path.sep.join(robot_parameters[0].usd_path[path_idx:]))

        objects_parameters = info['objects_parameters']
        objects_parameters[0][0].usd_path = os.path.abspath(objects_parameters[0][0].usd_path).split(os.path.sep)
        path_idx = objects_parameters[0][0].usd_path.index('VRKitchen2.0')
        objects_parameters[0][0].usd_path = os.path.join('/home/huangjiangyong/repo', os.path.sep.join(objects_parameters[0][0].usd_path[path_idx:]))

        robot_shift = info['robot_shift']

        if scene_loader is None:
            scene_loader = SceneLoader(simulation_app,scene_parameters, robot_parameters, objects_parameters, scene_properties, sensor_resolution=(128,128), use_gpu_physics=use_gpu_physics)
        else:
            scene_loader.reinitialize(scene_parameters, robot_parameters, objects_parameters, new_stage=False)

        franka = scene_loader.robots[0]
        franka_pose_init = franka.get_world_pose()
        franka.set_world_pose(franka_pose_init[0] + np.array(robot_shift), franka_pose_init[1])

        if 'object_angle' in info:
            object_angle = info['object_angle']
            object = XFormPrim(scene_loader.objects[0][0].GetPath().pathString)
            object_pose_init = object.get_world_pose()
            object.set_world_pose(object_pose_init[0], euler_angles_to_quat(object_angle))

    return scene_loader


def load_agent(args, device):
    if args.agent == 'cliport6d':
        model_cfg = {
            'train': {
                'attn_stream_fusion_type': 'add',
                'trans_stream_fusion_type': 'conv',
                'lang_fusion_type': 'mult',
                'n_rotations': 36,
                'batchnorm': False
            }
        }
        agent = TwoStreamClipLingUNetLatTransporterAgent(name='cliport_6dof', device=device, cfg=model_cfg, z_roll_pitch=True)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint['state_dict'])
        agent.eval()
        agent.to(device)
    elif args.agent == 'peract':
        agent = create_agent(args, train=False, device=device)
        agent.load_model(args.checkpoint_path)
    else:
        raise ValueError(f'{args.agent} agent not supported')
    return agent


def get_action(scene_loader, simulation_context, agent, franka, c_controller, npz_file, offset, timestep, device, agent_type, obs_type='rgb', lang_embed_cache=None):
    gt = scene_loader.render(simulation_context)

    obs = get_obs(franka, c_controller, gt, type=obs_type)
    robot_pos, robot_rot = obs.misc['robot_base']

    instruction = npz_file['gt'][0]['instruction']

    if agent_type == 'cliport6d':
        bounds = offset / 100

        # y-up to z-up
        bounds = bounds[[0, 2, 1]]
        obs.front_point_cloud = obs.front_point_cloud[:, :, [0, 2, 1]]
        obs.left_point_cloud = obs.left_point_cloud[:, :, [0, 2, 1]]
        obs.base_point_cloud = obs.base_point_cloud[:, :, [0, 2, 1]]
        obs.wrist_point_cloud = obs.wrist_point_cloud[:, :, [0, 2, 1]]
        obs.wrist_bottom_point_cloud = obs.wrist_bottom_point_cloud[:, :, [0, 2, 1]]

        inp_img, lang_goal, p0, output_dict = agent.act(obs, [instruction], bounds=bounds, pixel_size=5.625e-3)
        
        trans = np.ones(3) * 100
        # m to cm
        trans[[0,2]] *= output_dict['place_xy']
        trans[1] *= output_dict['place_z']
        trans += robot_pos

        rotation = np.array([output_dict['place_theta'], output_dict['pitch'], output_dict['roll']])
        rotation = R.from_euler('zyx', rotation, degrees=False).as_matrix()
        rot_transition = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).astype(float)
        rotation = R.from_matrix(rot_transition @ rotation).as_quat()

        # print(
        #     f'Model output: xy={output_dict["place_xy"]}, z={output_dict["place_z"]}, '
        #         f'theta={output_dict["place_theta"]/np.pi*180}, pitch={output_dict["pitch"]/np.pi*180}, roll={output_dict["roll"]/np.pi*180}'
        # )
    
    elif agent_type == 'peract':
        input_dict = {}

        obs_dict = get_obs_batch_dict(obs)
        input_dict.update(obs_dict)

        lang_goal_embs = lang_embed_cache.get_lang_embed(instruction)
        gripper_open = obs.gripper_open
        gripper_joint_positions = np.clip(obs.gripper_joint_positions, 0, 0.04)
        low_dim_state = np.array([gripper_open, *gripper_joint_positions, timestep]).reshape(1, -1)
        input_dict.update({
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state
        })

        for k, v in input_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v)
            input_dict[k] = v.to(device)
        
        output_dict = agent.predict(input_dict)
        trans = output_dict['pred_action']['continuous_trans'].detach().cpu().numpy() * 100 + robot_pos
        rotation = output_dict['pred_action']['continuous_quat']
    
    else:
        raise ValueError(f'{agent_type} agent not supported')

    print(f'action prediction: trans={trans}, orient(euler XYZ)={R.from_quat(rotation).as_euler("XYZ", degrees=True)}')

    rotation = rotation[[3, 0, 1, 2]]   # xyzw to wxyz

    return trans, rotation


def get_pre_grasp_action(grasp_action, robot_base, task):
    """
    grasp_action: ( pos_world, rot_world (wxyz) )
    robot_base: ( robot_pos, robot_rot (wxyz) )
    position represented in cm
    return pre-grasping action ( pre_pos_world, pre_rot_world (wxyz) )
    """
    pos_world, rot_world = grasp_action
    robot_pos, robot_rot = robot_base

    # wxyz to xyzw
    rot_world = rot_world[[1,2,3,0]]
    robot_rot = robot_rot[[1,2,3,0]]
    # to matrix
    rot_world = R.from_quat(rot_world).as_matrix()
    robot_rot = R.from_quat(robot_rot).as_matrix()

    # relative action
    pre_pos_world, pre_rot_world = get_pose_relat(trans=pos_world, rot=rot_world, robot_pos=robot_pos, robot_rot=robot_rot)

    if task in ['pickup_object', 'open_drawer', 'close_drawer', 'open_cabinet', 'close_cabinet']:
        # x - 5cm
        pre_pos_world[0] -= 5
    elif task in ['reorient_object']:
        # z at 15cm
        pre_pos_world[2] = 15
    else:
        # water, z + 5cm
        pre_pos_world[2] += 5
    
    # world action
    pre_pos_world, pre_rot_world = get_pose_world(trans_rel=pre_pos_world, rot_rel=pre_rot_world, robot_pos=robot_pos, robot_rot=robot_rot)
    pre_rot_world = R.from_matrix(pre_rot_world).as_quat()
    pre_rot_world = pre_rot_world[[3,0,1,2]]
    return pre_pos_world, pre_rot_world


def interpolate_xz(trans_previous, trans_target, alpha):
    previous_x, previous_z = trans_previous[0], trans_previous[2]
    target_x, target_z = trans_target[0], trans_target[2]
    x, z = alpha * np.array([target_x, target_z]) + (1 - alpha) * np.array([previous_x, previous_z])
    trans_interp = trans_target.copy()
    trans_interp[0], trans_interp[2] = x, z
    return trans_interp


NUM_STAGES = {
    'pickup_object': 3,
    'reorient_object': 3,
    'open_drawer': 3,
    'close_drawer': 3,
    'open_cabinet': 3,
    'close_cabinet': 3,
    'pour_water': 5,
    'transfer_water': 5
}

GRIPPER_OPEN = {
    'pickup_object': [True, False, False],
    'reorient_object': [True, False, False],
    'open_drawer': [True, False, False],
    'close_drawer': [True, False, False],
    'open_cabinet': [True, False, False],
    'close_cabinet': [True, False, False],
    'pour_water': [True, False, False, False, False],
    'transfer_water': [True, False, False, False, False]
}

NEED_INTERPOLATION = {
    'pickup_object': False,
    'reorient_object': False,
    'open_drawer': True,
    'close_drawer': True,
    'open_cabinet': True,
    'close_cabinet': True,
    'pour_water': True,
    'transfer_water': True
}

EVAL_SPLITS = ['test', 'novel_scene', 'novel_object', 'novel_state']


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    render = args.visualize
    use_gpu_physics = True if 'water' in args.task else False

    simulation_app, simulation_context, _ = get_simulation(headless=False, gpu_id=0)

    from parameters.parameters import StageProperties
    from SceneLoader import SceneLoader
    from omni.isaac.franka.controllers import RMPFlowController

    light_usd_path = '/home/huangjiangyong/repo/VRKitchen2.0/sample/light/skylight.usd'
    scene_properties = StageProperties(light_usd_path, "y", 0.01, gravity_direction=[0,-1,0], gravity_magnitude=981)

    agent = load_agent(args, device=device)
    if args.agent == 'peract':
        lang_encoder = create_lang_encoder(encoder_key=args.lang_encoder, device=device)
        lang_embed_cache = InstructionEmbedding(lang_encoder)
    else:
        lang_embed_cache = None

    eval_log = []
    for eval_split in EVAL_SPLITS:
        eval_log.append(f'Evaluating {eval_split}\n')
        data = load_data(data_path=os.path.join(args.data_dir, eval_split))

        scene_loader: SceneLoader = None
        correct = 0
        total = 0
        while len(data) > 0:
            anno = data[0]
            scene_loader = load_scene(data, scene_loader, simulation_app, scene_properties, use_gpu_physics)
            horizon = 2000
            franka = scene_loader.robots[0]
            gripper_controller = franka.gripper
            c_controller = RMPFlowController(name="cspace_controller", robot_articulation = franka, physics_dt = 1.0 / 120.0) 
            
            scene_loader.start(simulation_context=simulation_context)

            gt_frames = anno['gt']
            print(f'Instruction: {gt_frames[0]["instruction"]}')
            gt_actions = [
                gt_frames[1]['position_rotation_world'], gt_frames[2]['position_rotation_world'], gt_frames[3]['position_rotation_world']
            ]
            
            print(f'Ground truth action:')
            for gt_action, grip_open in zip(gt_actions, GRIPPER_OPEN[args.task]):
                act_pos, act_rot = gt_action
                act_rot = R.from_quat(act_rot[[1,2,3,0]]).as_euler('XYZ', degrees=True)
                print(f'trans={act_pos}, orient(euler XYZ)={act_rot}, gripper_open={grip_open}')
            
            offset = TASK_OFFSETS[args.task]

            use_gt = args.use_gt

            t_list = None
            doing_interpolation = False
            
            checker = scene_loader.time_line_callbacks[0][0]
            current_target = None
            
            stage = 0

            for step in range(horizon):
                if step % 120 == 0:
                    print("tick: ", step)
                
                if stage == NUM_STAGES[args.task]:
                    # stages exhausted, success check
                    for _ in range(250):
                        simulation_context.step(render=False)
                        if checker.success:
                            correct += 1
                            break
                    break
                
                if current_target is None:
                    grip_open = GRIPPER_OPEN[args.task][stage]
                    
                    if stage == 0:
                        if use_gt[0]:
                            trans_pre, rotation_pre = gt_actions[0]
                        else:
                            trans_pick, rotation_pick = get_action(
                                scene_loader, simulation_context, agent, franka, c_controller, anno, offset, timestep=0,
                                device=device, agent_type=args.agent, obs_type=args.obs_type, lang_embed_cache=lang_embed_cache
                            )
                            trans_pre, rotation_pre = get_pre_grasp_action(
                                grasp_action=(trans_pick, rotation_pick), robot_base=gt_frames[1]['robot_base'], task=args.task
                            )
                        current_target = (trans_pre, rotation_pre, grip_open)
                    elif stage == 1:
                        if use_gt[0]:
                            trans_pick, rotation_pick = gt_actions[1]
                        current_target = (trans_pick, rotation_pick, grip_open)
                    elif stage == 2:
                        # operation after grasping
                        if not doing_interpolation:
                            if use_gt[1]:
                                trans_target, rotation_target = gt_actions[2]
                            else:
                                trans_target, rotation_target = get_action(
                                    scene_loader, simulation_context, agent, franka, c_controller, anno, offset, timestep=1,
                                    device=device, agent_type=args.agent, obs_type=args.obs_type, lang_embed_cache=lang_embed_cache
                                )
                        
                        if NEED_INTERPOLATION[args.task]:
                            trans_previous = trans_pick
                            if t_list is None:
                                num_t = int(np.linalg.norm(trans_target - trans_previous) * 50)
                                t_list = list(np.linspace(start=0, stop=1, num=num_t))[1:]
                                doing_interpolation = True
                            
                            alpha = t_list.pop(0)
                            trans_interp = interpolate_xz(trans_previous, trans_target, alpha)
                            current_target = (trans_interp, rotation_target, grip_open)
                            if len(t_list) == 0:
                                t_list = None
                                doing_interpolation = False
                        else:
                            current_target = (trans_target, rotation_target, grip_open)
                    elif stage == 3:
                        # pour water
                        pass
                    elif stage == 4:
                        # cup return to upward orientation
                        pass
                
                tmp_target = copy.deepcopy(current_target[0])

                target_joint_positions = c_controller.forward(
                    target_end_effector_position=tmp_target, target_end_effector_orientation=current_target[1]
                )
                
                if position_reached(c_controller, tmp_target, franka, thres = 1.0) and rotation_reached(c_controller, current_target[1]):
                    if current_target is not None:
                        if current_target[2] < 0.5:
                            target_joint_positions_gripper = gripper_controller.forward(action="close")
                            for _ in range(10):
                                articulation_controller = franka.get_articulation_controller()
                                articulation_controller.apply_action(target_joint_positions_gripper)
                                simulation_context.step(render=render)
                            
                        else:
                            target_joint_positions_gripper = gripper_controller.forward(action="open")
                            for _ in range(10):
                                articulation_controller = franka.get_articulation_controller()
                                articulation_controller.apply_action(target_joint_positions_gripper)
                                simulation_context.step(render=render)

                    print("target reached!")
                    current_target = None
                    if not doing_interpolation:
                        stage += 1

                articulation_controller = franka.get_articulation_controller()
                articulation_controller.apply_action(target_joint_positions)

                simulation_context.step(render=render)
                if checker.success:
                    correct += 1
                    break
            
            scene_loader.stop()
            total += 1
            log_str = f'correct: {correct} | total: {total} | remaining: {len(data)}'
            print(log_str)
            eval_log.append(f'{log_str}\n')

        with open(f'eval_{args.task}.log', 'w') as f:
            f.writelines(eval_log)
    
    simulation_app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--agent', type=str)
    parser.add_argument('--lang_encoder', type=str)
    parser.add_argument('--obs_type', type=str)
    parser.add_argument('--use_gt', type=int, nargs='+')
    parser.add_argument('--visualize', type=int)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, metavar='PATH')
    args = parser.parse_args()

    main(args)

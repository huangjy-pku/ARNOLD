"""
For example, run:
    python train_peract.py --data_dir /mnt/huangjiangyong/VRKitchen/pickup_object --task pickup_object --obs_type rgb \
                           --batch_size 4 --steps 20000 --checkpoint_path /mnt/huangjiangyong/VRKitchen/pickup_object/ckpt_peract > train.log
"""

import os
import time
import clip
import torch
import numpy as np
from tqdm import trange
from dataset import ArnoldDataset
from peract.model import PerceiverIO, PerceiverActorAgent
from peract.utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler
from peract.utils import CAMERAS, IMAGE_SIZE, VOXEL_SIZES, ROTATION_RESOLUTION, SCENE_BOUNDS


def create_lang_encoder(encoder_key, device):
    if encoder_key == 'clip':
        model, preprocess = clip.load("RN50", device=device)
        lang_encoder = model.transformer
    elif encoder_key == 't5':
        raise NotImplementedError()
    else:
        raise ValueError('Language encoder key not supported')
    
    return lang_encoder


def create_agent(device):
    perceiver_encoder = PerceiverIO(
        depth=6,
        iterations=1,
        voxel_size=VOXEL_SIZES[0],
        initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4,
        layer=0,
        num_rotation_classes=72,
        num_grip_classes=2,
        num_collision_classes=2,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,
        voxel_patch_stride=5,
        final_dim=64,
    )

    peract_agent = PerceiverActorAgent(
        coordinate_bounds=SCENE_BOUNDS,
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=batch_size,
        voxel_size=VOXEL_SIZES[0],
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )

    peract_agent.build(training=True, device=device)

    return peract_agent


def main(args):
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = ArnoldDataset(data_path=os.path.join(args.data_dir, 'train'), task=args.task, obs_type=args.obs_type)
    
    writer = SummaryWriter(log_dir=args.checkpoint_path)

    agent = create_agent(device)
    agent.to(device)

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            start_step = agent.load_model(args.resume)
            print("=> loaded checkpoint '{}' (step {})".format(args.resume, start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    agent.train()

    print(f'Training {args.steps} steps with batch_size = {args.batch_size}')
    start_time = time.time()
    for iteration in trange(start_step, args.steps):
        batch_data = train_dataset.sample(batch_size)

        obs_dict = {}
        language_instructions = []
        target_points = []
        low_dim_state = []
        for data in batch_data:
            for k, v in data['obs_dict'].items():
                if k not in obs_dict:
                    obs_dict[k] = []
                else:
                    obs_dict[k].append(v)
            
            language_instructions.append(data['language'])
            target_points.append(data['target_points'])
            low_dim_state.append(data['low_dim_state'])

        for k, v in obs_dict.items():
            obs_dict[k] = np.stack(v, axis=0)
        
        target_points = np.stack(target_points, axis=0)
        low_dim_state = np.stack(low_dim_state, axis=0)

        # peract
        bs = len(language_instructions)

        trans_action_coords = target_points[:, [0, 2, 1]]
        trans_action_indices = point_to_voxel_index(trans_action_coords, vox_size, bounds)

        rot_action_quat = target_points[:, 3:]
        rot_action_quat = normalize_quaternion(rot_action_quat)
        rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, ROTATION_RESOLUTION)
        rot_grip_action_indices = np.concatenate([rot_action_indices, grip_action.reshape(-1, 1)], axis=-1)

        lang_goal_embs = dataset.get_lang_embed(lang_encoder, language_instructions)

        inp = {}
        inp.update({obs_dict})
        inp.update({
            'trans_action_indices': trans_action_indices,
            'rot_grip_action_indices': rot_grip_action_indices,
            'ignore_collisions': np.ones((bs, 1)),
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state
        })

        for k, v in inp.items():
            if not isinstance(v, torch.Tensor):
                inp[k] = torch.from_numpy(v)
            inp[k] = inp[k].to(device)

        update_dict = agent.update(iteration, inp)
        running_loss = update_dict['total_loss']

        if iteration % args.log_freq == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print(f'Total Loss: {running_loss} | Elapsed Time: {elapsed_time} mins')
        
        if iteration % args.save_freq == 0:
            path = os.path.join(args.checkpoint_path, f'model_{iteration}.pth')
            agent.save_model(path, iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--obs_type', type=str)
    parser.add_argument('--batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--steps', type=int, help='Set optimization steps for training')
    parser.add_argument('--log-freq', default=50, type=int)
    parser.add_argument('--save-freq', default=5000, type=int)
    parser.add_argument('--checkpoint_path', type=str, metavar='PATH')
    parser.add_argument('--resume', default=None, type=str, help='resume training from checkpoint file')

    args = parser.parse_args()

    main(args)
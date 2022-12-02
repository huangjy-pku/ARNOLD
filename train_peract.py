"""
For example, run:
    python train_peract.py --data_dir /mnt/huangjiangyong/VRKitchen/pickup_object --task pickup_object \
                           --obs_type rgb --lang_encoder clip --batch_size 4 --steps 20000 \
                           --checkpoint_path /mnt/huangjiangyong/VRKitchen/pickup_object/ckpt_peract > train.log
"""

import os
import time
import torch
import argparse
import numpy as np
from math import ceil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ArnoldDataset, InstructionEmbedding
from peract.agent import CLIP_encoder, T5_encoder, PerceiverIO, PerceiverActorAgent
from peract.utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler
from custom_utils.misc import CAMERAS, IMAGE_SIZE, TASK_OFFSET_BOUNDS, VOXEL_SIZES, ROTATION_RESOLUTION, T5_CFG, LANG_EMBED_DIM, collate_fn


def create_lang_encoder(encoder_key, device):
    if encoder_key == 'clip':
        lang_encoder = CLIP_encoder(device)
    elif encoder_key == 't5':
        lang_encoder = T5_encoder(T5_CFG, device)
    else:
        raise ValueError('Language encoder key not supported')
    
    return lang_encoder


def create_agent(args, train, device):
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
        lang_embed_dim=LANG_EMBED_DIM[args.lang_encoder]
    )

    peract_agent = PerceiverActorAgent(
        coordinate_bounds=TASK_OFFSET_BOUNDS[args.task],
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=args.batch_size,
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

    peract_agent.build(training=train, device=device)

    return peract_agent


def prepare_batch(batch_data, args, lang_embed_cache, device):
    obs_dict = {}
    language_instructions = []
    target_points = []
    gripper_open = []
    low_dim_state = []
    for data in batch_data:
        for k, v in data['obs_dict'].items():
            if k not in obs_dict:
                obs_dict[k] = [v]
            else:
                obs_dict[k].append(v)
        
        target_points.append(data['target_points'])
        gripper_open.append(data['target_gripper'])
        language_instructions.append(data['language'])
        low_dim_state.append(data['low_dim_state'])

    for k, v in obs_dict.items():
        v = np.stack(v, axis=0)
        obs_dict[k] = v.transpose(0, 3, 1, 2)   # peract requires input as [C, H, W]
    
    bs = len(language_instructions)
    target_points = np.stack(target_points, axis=0)
    gripper_open = np.array(gripper_open).reshape(bs, 1)
    low_dim_state = np.stack(low_dim_state, axis=0)

    trans_action_coords = target_points[:, :3]
    trans_action_indices = point_to_voxel_index(trans_action_coords, VOXEL_SIZES[0], TASK_OFFSET_BOUNDS[args.task])

    rot_action_quat = target_points[:, 3:]
    rot_action_quat = normalize_quaternion(rot_action_quat)
    rot_action_indices = quaternion_to_discrete_euler(rot_action_quat, ROTATION_RESOLUTION)
    rot_grip_action_indices = np.concatenate([rot_action_indices, gripper_open], axis=-1)

    lang_goal_embs = lang_embed_cache.get_lang_embed(language_instructions)

    inp = {}
    inp.update(obs_dict)
    inp.update({
        'trans_action_indices': trans_action_indices,
        'rot_grip_action_indices': rot_grip_action_indices,
        'ignore_collisions': np.ones((bs, 1)),
        'lang_goal_embs': lang_goal_embs,
        'low_dim_state': low_dim_state
    })

    for k, v in inp.items():
        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(v)
        inp[k] = v.to(device)
    
    return inp


def main(args):
    if not os.path.exists(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = ArnoldDataset(data_path=os.path.join(args.data_dir, 'train'), task=args.task, obs_type=args.obs_type)
    val_dataset = ArnoldDataset(data_path=os.path.join(args.data_dir, 'val'), task=args.task, obs_type=args.obs_type)

    # train set used for iterative sampling, val set for enumeration
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.batch_size//4, pin_memory=True, collate_fn=collate_fn,
    )
    
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_path, f'{args.obs_type}_{args.lang_encoder}'))

    agent = create_agent(args, train=True, device=device)

    lang_encoder = create_lang_encoder(args.lang_encoder, device)
    lang_embed_cache = InstructionEmbedding(lang_encoder)

    start_step = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            start_step = agent.load_model(args.resume)
            print("=> loaded checkpoint '{}' (step {})".format(args.resume, start_step))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_val_loss = float('inf')
    args.epochs = ceil( args.steps / (len(train_dataset)/args.batch_size) )
    print(f'Training epochs: {args.steps} steps / ({len(train_dataset)} demos / {args.batch_size} batch_size) = {args.epochs}')
    start_time = time.time()
    for iteration in range(start_step, args.steps):
        # train
        batch_data = train_dataset.sample(args.batch_size)
        inp = prepare_batch(batch_data, args, lang_embed_cache, device)
        update_dict = agent.update(iteration, inp)
        running_loss = update_dict['total_loss']

        if iteration % args.log_freq == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print(f'Iteration: {iteration} | Total Loss: {running_loss} | Elapsed Time: {elapsed_time} mins')
            writer.add_scalar('total_loss', running_loss, iteration)
        
        if iteration % args.save_freq == 0:
            # val
            val_loss = []
            for batch_data in val_loader:
                inp = prepare_batch(batch_data, args, lang_embed_cache, device)
                update_dict = agent.update(iteration, inp)
                val_loss.append(update_dict['total_loss'])
            
            val_loss = np.mean(val_loss)
            print(f'Iteration: {iteration} | Val_loss: {val_loss:.4f} | Previous best loss: {best_val_loss:.4f}')
            writer.add_scalar('val_loss', val_loss, iteration)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                path = os.path.join(args.checkpoint_path, f'model_{args.task}_{args.obs_type}_{args.lang_encoder}_{iteration}.pth')
                agent.save_model(path, iteration)
    
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--obs_type', type=str)
    parser.add_argument('--batch_size', type=int, default=16, metavar='N')
    parser.add_argument('--steps', type=int, help='Set optimization steps for training')
    parser.add_argument('--log-freq', default=50, type=int)
    parser.add_argument('--save-freq', default=5000, type=int)
    parser.add_argument('--checkpoint_path', type=str, metavar='PATH')
    parser.add_argument('--resume', default=None, type=str, help='resume training from checkpoint file')
    parser.add_argument('--lang_encoder', type=str)

    args = parser.parse_args()

    main(args)

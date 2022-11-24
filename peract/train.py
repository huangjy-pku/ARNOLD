import numpy as np

import clip
import torch
from .utils import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_euler
from .utils import BATCH_SIZE, CAMERAS, VOXEL_SIZES, ROTATION_RESOLUTION, SCENE_BOUNDS, IMAGE_SIZE
import os
data_path = '/mnt/huangjiangyong/VRKitchen/pickup_object/train'
num_demos = len(os.listdir(data_path))
train_replay_storage_dir = '/mnt/huangjiangyong/VRKitchen/pickup_object/replay'
batch_size = BATCH_SIZE

train_replay_buffer = create_replay(
        batch_size=batch_size,
        timesteps=1,
        save_dir=train_replay_storage_dir,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("RN50", device=device) # CLIP-ResNet50


fill_replay(
            data_path= data_path,
            replay=train_replay_buffer,
            start_idx=0,
            num_demos=num_demos,
            demo_augmentation=True,
            # demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            clip_model=clip_model,
            device=device,
            )

# delete the CLIP model since we have already extracted language features
del clip_model
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
# wrap buffer with PyTorch dataset and make iterator
train_wrapped_replay = PyTorchReplayBuffer(train_replay_buffer)
train_dataset = train_wrapped_replay.dataset()
train_data_iter = iter(train_dataset)

from arm.utils import visualise_voxel

from voxel_grid import VoxelGrid
from custom_utils import SCENE_BOUNDS, _preprocess_inputs
# initialize voxelizer
vox_grid = VoxelGrid(
    coord_bounds=SCENE_BOUNDS,
    voxel_size=VOXEL_SIZES[0],
    device=device,
    batch_size=batch_size,
    feature_size=3,
    max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
)

from .model import PerceiverIO, PerceiverActorAgent

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
    num_latents=NUM_LATENTS,
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
# resume training
# path = '/mnt/data/model/model_8000.pth'
initial_iterations = 0
# initial_iterations = peract_agent.load_model(path)

import time

LOG_FREQ = 50
SAVE_FREQ = 1000
TRAINING_ITERATIONS = 20002
from tqdm import tqdm
from dataset import ArnoldDataset
from torch.utils.data import DataLoader
from custom_utils.misc import collate_fn
dataset = ArnoldDataset(data_path='/mnt/huangjiangyong/VRKitchen/overfit/train', task='pickup_object')
train_loader = DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0,
    pin_memory=True, drop_last=False, collate_fn=collate_fn, persistent_workers=True
)

def train(data_loader, model, optimizer, scheduler, epoch, losses, args, timer):
    model.train()
    running_loss = 0
    start_time = time.time()
    for batch_step, batch_data in enumerate(data_loader):
        if len(batch_data)==0:
            continue

        language_instructions = [], []
        target_points = [], []
        for data in batch_data:
            language_instructions.append(data['language'])
            target_points.append(data['target_points'])

        target_points = np.stack(target_points, axis=0)

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
            'ignore_collisions': np.ones(bs),
            'lang_goal_embs': lang_goal_embs,
            'low_dim_state': low_dim_state
        })

        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        update_dict = peract_agent.update(iteration, batch)
        running_loss += update_dict['total_loss']
        if iteration % LOG_FREQ == 0:
            elapsed_time = (time.time() - start_time) / 60.0
            print("Total Loss: %f | Elapsed Time: %f mins  running loss %f" % (update_dict['total_loss'], elapsed_time, running_loss))
            running_loss = 0
        
        if iteration % SAVE_FREQ == 0:
            path = f'/mnt/data/model/model_{iteration}.pth'
            peract_agent.save_model(path, iteration)

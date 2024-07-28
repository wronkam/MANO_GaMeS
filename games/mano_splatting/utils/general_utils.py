#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#
import gc
import json
import os, re
import queue
import random
import threading
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, SimpleQueue
from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from scene.dataset_readers import CameraInfo
from games.mano_splatting.MANO import ManoConfig


def write_mesh_obj(
        vertices: torch.tensor,
        faces: torch.tensor,
        filepath,
        verbose=False
):
    """Simple save vertices and face as an obj file."""
    vertices = vertices.detach().cpu().numpy()
    with open(filepath, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', f'{filepath}.obj')


def rot_to_quat_batch(rot: torch.Tensor):
    """
    Implementation based on pytorch3d implementation
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot.reshape(-1, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
          F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
          ].reshape(-1, 4)

    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = torch.Tensor([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

    return rot_matrix


class InterHandsSceneInfo(NamedTuple):  # (SceneInfo):
    # TODO: fix inheritance
    mano_config: ManoConfig
    point_cloud: NamedTuple
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def loadInterHandCam(args, id, cam_info, resolution_scale, mano_config, thread=None):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return InterHandCamera(cam_info=cam_info, mano_config=mano_config,
                           resolution=resolution, uid=id, data_device=args.data_device, thread=thread)


def extract_number(string):
    return re.sub('[^0-9]', '', string)


class Storage(ABC):
    @abstractmethod
    def __init__(self, size):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_next(self, lock=False):
        pass

    @abstractmethod
    def store(self, data):
        pass


class ConstantStorage(Storage):
    def __init__(self, size):
        self.storage = []
        # synchronised queue, infinite max size
        self.size = size
        self.iter = 0

    def __len__(self):
        return len(self.storage)

    def get_next(self, lock=False):
        if self.iter >= len(self.storage):
            self.iter = 0
        if len(self.storage) == 0:
            warnings.warn('get_next of an empty storage unit was called')
            return None, False
        data = self.storage[self.iter]
        self.iter += 1
        return data, True

    def store(self, data):
        if len(self.storage) >= self.size:
            return
        self.storage.append(data)


class QueueStorage(Storage):
    def __init__(self, size):
        self.storage = Queue(maxsize=0)
        # synchronised queue, infinite max size
        self.size = size

    def __len__(self):
        return self.storage.qsize()

    def get_next(self, lock=False):
        try:
            data = self.storage.get(block=lock)
        except queue.Empty:
            return None, False
        return data, True

    def store(self, data):
        self.storage.put(data, block=True)


class AtomicCounter:
    def __init__(self, cv=None):
        # intended use is main thread sets it to some value
        # (assumes it is zero and none wants to decrease it at the time)
        self.cv = cv  # proxy for associated conditional variable
        self.counter = 0
        self.lock = threading.Lock()

    def set(self, value):
        with self.lock:
            assert self.counter == 0
            self.counter = value

    def decrease(self):
        with self.lock:
            self.counter -= 1

    def get_value(self):
        return self.counter


# noinspection SpellCheckingInspection
class InterHandCamera:
    def __init__(self, cam_info: CameraInfo, mano_config: ManoConfig, resolution: Tuple[int, int],
                 uid, data_device, thread: ThreadPoolExecutor = None):
        self.mano_config = mano_config
        self.resolution = resolution
        #  resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        self.uid = uid
        self.cam_info = cam_info
        self.cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_device = self.cuda_device if mano_config.load_on_gpu else torch.device('cpu')

        self.annotation = os.path.join(mano_config.InterHands_annots_path)
        self.capture = extract_number(self.mano_config.InterHands_keys[1])  # captureX with X in [0..7]

        cv = threading.Condition()
        self.job_counter = AtomicCounter(cv)
        self.thread = thread

        camera_name = cam_info.image_name
        self.camera_name = camera_name

        with open(self.annotation, 'r') as f:  # will slow down init, won't impact memory
            annots = json.load(f)
        annot_frmes = sorted(list(annots[str(self.capture)].keys()))
        del annots
        gc.collect()

        annot_frmes = [extract_number(x) for x in annot_frmes]  # str
        annot_frmes = [x for x in annot_frmes
                       if self.mano_config.first_capture <= int(x) <= self.mano_config.last_capture]

        self.all_frames_idx = {x: t for t, x in enumerate(annot_frmes)}  # gives global index of frame
        self.all_frames = annot_frmes  # gives frame by index

        self.mask_path = os.path.join(
            *([self.mano_config.InterHands_masks_path] + mano_config.InterHands_keys + [f'cam{camera_name}']))
        self.masks = {extract_number(f): f for f in os.listdir(self.mask_path)
                      if (os.path.isfile(os.path.join(self.mask_path, f))
                          and extract_number(f) in self.all_frames)}

        self.image_path = os.path.join(
            *([self.mano_config.InterHands_images_path] + mano_config.InterHands_keys + [f'cam{camera_name}']))
        self.images = {extract_number(f): f for f in os.listdir(self.image_path)
                       if os.path.isfile(os.path.join(self.image_path, f))
                       and (extract_number(f) in self.masks.keys())}  # images are jpg and masks are png
        # assuemes all cam dirs have all images
        self.frames = []
        self.schedule = self.mano_config.camera_schedule
        self.schedule = {time: camera_list for time, camera_list in self.schedule.items() if len(camera_list) > 0}
        self.next_update = -1
        self.index_provider = []  # inf size, resupplied on frames added with update_schedule and when empty from frames
        self.update_schedule(0)  # sets frames as a list of frames that were enabled
        # takes frame_idx from schedule and checks if all_frames[frame_idx] is in masks key

        if mano_config.load_all_images:
            self.size = len(self.masks)
            self.storage = ConstantStorage(self.size)
        else:
            self.size = mano_config.preload_size
            self.storage = QueueStorage(self.size)
        if len(self.frames) > 0:
            self.preload(-1)  # for const it will load all now, and nothing later

    def __len__(self):
        return len(self.frames)

    def preload(self, num=1):
        with self.job_counter.cv:
            while self.job_counter.get_value() > 0:
                # if there are jobs currently loaded for this camera, wait for them to finish before scheduling new ones
                self.job_counter.cv.wait()
            if num <= 0:
                num = max(0, min(self.size - len(self.storage), len(self.frames)))
                # constant storage extends index provider only with new frames -> it will add new frames to const jit
            self.job_counter.set(num)
            for num_idx in range(num):
                idx = self.get_index()
                frame = self.frames[idx]
                name = frame
                global_frame_idx = self.all_frames_idx[frame]
                kwargs = {
                    'storage': self.storage,
                    'frame_idx': global_frame_idx,
                    'image_path': os.path.join(self.image_path, self.images[frame]),
                    'mask_path': os.path.join(self.mask_path, self.masks[frame]),
                    'resolution': self.resolution,
                    'uid': self.uid,
                    'cam_info': self.cam_info,
                    'data_device': self.data_device,
                    'name': name,
                    'is_rhand': self.mano_config.mano_is_rhand,
                    'capture': self.capture,
                    'annotation_path': self.annotation,
                    'cam_name': self.camera_name,
                    'job_counter': self.job_counter,
                }
                if self.thread is None:
                    InterHandCamera.load_from_mem(**kwargs)
                else:
                    self.thread.submit(InterHandCamera.load_from_mem, **kwargs)


    def next(self, iteration=0):
        self.update_schedule(iteration)  # updates index provider if new frames get added (used in preload -> get_index)
        data, success = self.storage.get_next(lock=False)
        while not success:
            # preload and wait for it if Q is empty
            self.preload(1)
            data, success = self.storage.get_next(lock=True)
            self.preload(-1)  # replenish up to self.size, does nothing for once loaded const storage
        camera, frame, mano_pose = data
        if self.data_device != self.cuda_device:
            camera = camera.to(self.cuda_device)
        return camera, mano_pose, frame

    @staticmethod
    def load_from_mem(storage, frame_idx,
                      image_path, mask_path, resolution, uid, cam_info, data_device,
                      name, is_rhand, capture, annotation_path, job_counter, cam_name=None):
        mano_pose = InterHandCamera.get_mano(name, is_rhand, annotation_path, capture)
        camera = InterHandCamera.get_cam(image_path, mask_path, resolution, uid, cam_info, data_device)
        storage.store((camera, frame_idx, mano_pose))
        with job_counter.cv:
            job_counter.decrease()
            job_counter.cv.notify()

    @staticmethod
    def get_cam(image_path, mask_path, resolution, uid, cam_info, data_device):
        gt_image = Image.open(image_path)
        loaded_mask = Image.open(mask_path)

        gt_image = PILtoTorch(gt_image, resolution)
        loaded_mask = PILtoTorch(loaded_mask, resolution)
        if loaded_mask.shape[0] > 1:  # probably 3 x H x W
            loaded_mask = loaded_mask.mean(0).unsqueeze(0)
        return Camera(colmap_id=uid, R=cam_info.R, T=cam_info.T,
                      FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                      image=gt_image, gt_alpha_mask=loaded_mask,
                      image_name=cam_info.image_name, uid=id, data_device=data_device)

    def shuffle(self):
        order = list(range(len(self.frames)))
        random.shuffle(order)
        self.index_provider = order + self.index_provider

    @staticmethod
    def get_mano(name, is_rhand, annotation_path, capture):
        hand = 'right' if is_rhand else 'left'
        with open(annotation_path, 'r') as f:
            annots = json.load(f)
        return annots[capture][name][hand]['pose']

    def get_index(self):
        if len(self.index_provider) == 0:
            self.shuffle()
        return self.index_provider.pop(-1)

    def update_schedule(self, iteration):
        if iteration < self.next_update:
            return
        timestamps = sorted([t for t in self.schedule.keys() if t <= iteration])
        new_frames = []
        for timestamp in timestamps:
            frame_list = self.schedule[timestamp]
            for frame_idx in frame_list:
                if (0 <= frame_idx < len(self.all_frames)) and self.all_frames[frame_idx] in self.masks.keys():
                    new_frames.append(self.all_frames[frame_idx])
        if len(new_frames) > 0:
            self.schedule = {k: v for k, v in self.schedule.items() if k not in timestamps}
            self.index_provider = [len(self.frames) + i for i, _ in enumerate(new_frames)] + self.index_provider
            # add new frame indexes to index_provider (list of next frames to process) and reshuffle
            random.shuffle(self.index_provider)
            self.frames += new_frames
            # update next_update
            if len(self.schedule.keys()) == 0:
                self.next_update = 10e15
            else:
                self.next_update = min(list(self.schedule.keys()))

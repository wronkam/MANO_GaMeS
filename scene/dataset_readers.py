#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: NamedTuple
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except ValueError: # gs ply can have no color
        colors = np.zeros_like(positions)
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    # TODO: reuse to read interhands
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        print('fovx',fovx)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"][2:] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def _get_int_from_str(_str):
    return int("".join([x for x in _str if x.isnumeric()]))


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def readCamerasFromTransformsInterHand(path_to_set: str, white_background: bool, subject: str = "test/Capture0/ROM04_RT_Occlusion", frame_no: int = 0) -> list:    
    path_to_set = Path(path_to_set)
    phase = subject.split('/')[0]
    capture = _get_int_from_str(subject.split('/')[1])
    
    images_dir = Path(path_to_set, "images", subject)
    cameras_json_path = Path(path_to_set.parent, "annotations", phase, "InterHand2.6M_" + phase + "_camera.json")
    masks_dir = Path(path_to_set, "masks_removeblack", subject)
    
    all_cam_names = [x.stem for x in masks_dir.glob("*") if x.is_dir()]
    
    wanted_mask_name = None
    bg = np.array([[[1,1,1]]]) if white_background else np.array([[[0, 0, 0]]])
    
    with open(cameras_json_path, "r") as f:
        cameras = json.load(f)
    
    cam_infos = []
    adjustment = np.diag(np.array([1, -1, -1]))
    
    cam_names_with_mask = []
    img_paths = []
    poses = []
    images = []
    fovxs = []
    fovys = []
    
    for cam_name in all_cam_names:
        mask_cam_path = Path(masks_dir, cam_name)
        if wanted_mask_name is None:
            mask_names = sorted([x.name for x in mask_cam_path.glob("*png")])
            wanted_mask_name = mask_names[frame_no]
        mask_path = Path(masks_dir, cam_name, wanted_mask_name)
        try:
            mask = np.array(Image.open(mask_path), dtype=np.float32)[:, :, 0] / 255
        except FileNotFoundError:
            continue
        img_path = Path(images_dir, cam_name, wanted_mask_name).with_suffix(".jpg")
        img_paths.append(img_path)
        img = np.array(Image.open(img_path), dtype=np.float32)
        cam_id = _get_int_from_str(cam_name)
        
        cam_names_with_mask.append(cam_name)
        cam_id = _get_int_from_str(cam_name)
        init_T, R = np.array(cameras[str(capture)]['campos'][str(cam_id)], dtype=np.float32), np.array(cameras[str(capture)]['camrot'][str(cam_id)], dtype=np.float32)
        focal, princpt = np.array(cameras[str(capture)]['focal'][str(cam_id)], dtype=np.float32), np.array(cameras[str(capture)]['princpt'][str(cam_id)], dtype=np.float32)

        delta_x = (img.shape[1] / 2) - princpt[0]
        delta_y = (img.shape[0] / 2) - princpt[1]
        trans_matrix = np.eye(3, dtype=np.float32)[:2, :]
        trans_matrix[0, 2] = delta_x
        trans_matrix[1, 2] = delta_y
        
        mask_rgb = np.repeat(mask[:, :, None], 3, -1)
        trans_mask = cv2.warpAffine(
            mask_rgb,
            trans_matrix,
            (img.shape[1], img.shape[0])
        )
        trans_img = cv2.warpAffine(
            img,
            trans_matrix,
            (img.shape[1], img.shape[0])
        )
        
        masked_img = (trans_img * trans_mask + 255 * bg * (1 - trans_mask)).astype(np.float32)
        masked_img = Image.fromarray(masked_img.astype(dtype=np.byte), "RGB")
        images.append(masked_img)
        
        fovxs.append(2 * np.arctan(img.shape[1] / (2 * focal[0])))
        fovys.append(2 * np.arctan(img.shape[0] / (2 * focal[1])))
        
        R = np.array(R)
        R = np.matmul(adjustment, R).T
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = init_T
        poses.append(c2w)
    poses = np.stack(poses)
    poses = recenter_poses(poses)
    
    for idx, (img_path, pose, img, fovx, fovy) in enumerate(zip(img_paths, poses, images, fovxs, fovys)):
        c2w = pose
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        
        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=fovy,
                FovX=fovx,
                image=img,
                image_path=img_path,
                image_name=img_path.name,
                width=img.width,
                height=img.height
            )
        )
    return cam_infos


def readInterHandInfo(path_to_set: Path, white_background: bool, frame_no: int = 0, val_perc: float = 0.1):
    subject = "test/Capture0/ROM04_RT_Occlusion"
    
    cam_infos = readCamerasFromTransformsInterHand(path_to_set, white_background, subject, frame_no)
    
    nerf_normalization = getNerfppNorm(cam_infos)
    
    ply_path = os.path.join(path_to_set, f"{'_'.join(subject.split('/'))}_{frame_no}_points3d.ply")
    # Since this data set has no colmap data, we start with random points
    num_pts = 600_000
    print(f"Generating random point cloud ({num_pts})...")

    init_pos = np.array([0., 0., -300.], dtype=np.float32)
    
    cube_size = 200
    xyz = np.random.random((num_pts, 3)) * cube_size - (cube_size / 2) + init_pos
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    no_of_test_cams = int(len(cam_infos) * val_perc)
    test_cams = cam_infos[:no_of_test_cams]
    train_cams = cam_infos[no_of_test_cams:]
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cams,
                           test_cameras=test_cams,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
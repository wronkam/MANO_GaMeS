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
import os
import re
from glob import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from games.mano_splatting.MANO.MANO import MANO
from games.mano_splatting.MANO.config import ManoConfig
from games.mano_splatting.utils.general_utils import InterHandsSceneInfo
from games.mano_splatting.utils.graphics_utils import MANOPointCloud, matrix_to_euler_angles
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, \
    read_intrinsics_text, read_points3D_binary, read_points3D_text
from scene.dataset_readers import (
    getNerfppNorm,
    SceneInfo,
    readColmapCameras, storePly, fetchPly,
)
from utils.sh_utils import SH2RGB

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_mano(vertices, c=8):
    return vertices.squeeze(0)


def merge_indices(A, B, max_dist, vis):
    if A is None or len(A) == 0:
        return B
    if B is None or len(B) == 0:
        return A
    indices = torch.cat((A[:, 0], B[:, 0]))
    values = torch.cat((A[:, 1], B[:, 1]))
    out = torch.stack((indices, values), 1)
    out = torch.sort(out, dim=0)[0]
    mask = (out[:-1, 0] != out[1:, 0])
    out = torch.concat((out[0].unsqueeze(0), out[1:][mask]))
    return out


def closest_point(point_org, pcd_org, max_dist, k=2, visited=None, iter=0):
    if len(point_org.shape) == 1:
        point_org = point_org.unsqueeze(0)
    out = None
    for pointz in tqdm(point_org, desc=f"Iteration {iter}", leave=False):
        pcd = pcd_org.clone().detach()[visited]
        indices = torch.tensor(range(pcd_org.shape[0])).to(pcd.device)[visited]
        point = pointz.clone().detach()
        pcd = ((pcd - point) ** 2).sum(-1)
        closest = torch.topk(-pcd, k=k)  # indices
        # indices, distance
        closest = torch.stack((indices[closest[1].int()], -closest[0]), 1)
        closest = closest[closest[:, 1] < max_dist]
        out = merge_indices(out, closest, max_dist, visited)
    return out[:, 0].int(), out[:, 1]

from utils.general_utils import plt_scatter as _scatter


def reduce_points(vertices: torch.Tensor, start: torch.Tensor, knn: int,
                  max_distance: float, theta: float, iterations: int, show: bool):
    if start is None:
        start = vertices.mean(0).unsqueeze(0)
    visited = (vertices[:, 0] == vertices[:, 0])
    print("start", start)
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        _scatter(start, ax, 'y', 1.)

    dist = max_distance if max_distance is not None else (vertices.std(dim=0) ** 2).mean() ** (1 / 2)
    print("STD", vertices.std(0).shape)
    available = torch.where(visited, 1, 0).sum()
    tr = trange(iterations, desc="Reduction")
    for iter in tr:
        if iter == 0:
            cl_ind, dists = closest_point(start, vertices, 1e18, knn, visited, iter + 1)
        else:
            cl_ind, dists = closest_point(start, vertices, dist, knn, visited, iter + 1)
        if len(cl_ind) > 0:
            dist = dists.mean() * 1.6 * theta + (1 - theta) * dist
            closest_vert = vertices[cl_ind]
            start = torch.cat((closest_vert, start), 0)
            visited[cl_ind] = False
            if show:
                _scatter(closest_vert, ax, (0., 1 - iter / iterations, iter / iterations),
                         0.8)
            print(start.shape, closest_vert.shape)
        new_available = torch.where(visited, 1, 0).sum()
        tr.set_postfix({"Found":(available - new_available).item(), "left": new_available.item(), "Range": dist.item()})
        if new_available < knn or available - new_available == 0:
            break
        available = new_available
    if show:
        # _scatter(vertices[visited], ax, 'r', 0.1)
        fig.suptitle("Initial")
        plt.show()
    return vertices[~visited]


def get_rotation_diff(vdir:torch.Tensor, pdir:torch.Tensor):
    # Compute the cross product and dot product
    a = vdir.detach().cpu().numpy()
    b = pdir.detach().cpu().numpy()
    v = np.cross(a, b)
    c = np.dot(a, b)

    # Calculate the rotation matrix
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return torch.tensor(rotation_matrix).to(pdir.device)


def match_pca(points:torch.Tensor, vertices:torch.Tensor):
    pm = points.mean(0)
    vm = vertices.mean(0)
    ppca = torch.pca_lowrank(points - pm, 2, False, 8)
    vpca = torch.pca_lowrank(vertices - vm,2, False, 8)
    vdir = vpca[2]
    pdir = ppca[2]

    rotation = get_rotation_diff(vdir[:,0],pdir[:,0]).float()
    vdir = rotation @ vdir
    rotation2 = get_rotation_diff(vdir[:,1],pdir[:,1]).float()

    points2 = points - points.mean(0)
    pstd = points2.std(0)

    vertices = vertices - vertices.mean(0) # centralize
    vertices = (rotation2 @ rotation @ vertices.T).T # rotate
    vstd = vertices.std(0)

    transl = points.mean(0).unsqueeze(0).to(vertices.device)
    rot = matrix_to_euler_angles(rotation2 @ rotation).unsqueeze(0).to(vertices.device)
    # simplification, but hands should generally keep certain proportions
    scale = torch.tensor([[(pstd/vstd).mean()]]).to(vertices.device)

    return transl, rot, scale


def plot_pca(ax, points,pca,mean,c1='g', c2='r', alpha=0.1):
    pm = mean
    ppca = pca
    _scatter(points, ax, c1, alpha)
    _scatter(pm, ax, c2, 1)
    for dir, val in zip(ppca[2].permute(1, 0), torch.ones_like(ppca[1])):
        m = pm.detach().cpu().numpy()
        dir = (dir * val).detach().cpu().numpy()
        ax.plot([m[0], m[0] + dir[0]], [m[1], m[1] + dir[1]], [m[2], m[2] + dir[2]], c=c2)


def readColmapManoInfo(path, images, eval, llffhold=8):
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

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

    # print("Reading MANO object")
    mano_config = ManoConfig()
    mano_config.annot_path = os.path.join(path, "mano_params.json")
    mano_model = MANO(mano_config).to(mano_config.device)
    if mano_config.reduce_points:
        points = torch.tensor(pcd.points)
        points = reduce_points(points, mano_config.reduction_start, mano_config.reduction_knn,
                               mano_config.reduction_max_distance, mano_config.reduction_theta,
                               mano_config.reduction_max_iters, mano_config.plot_reduction)
    if mano_config.load_gs_points is not None:
        last_saved = {int(re.findall('\d+',path)[0]):path
                      for path in glob(os.path.join(mano_config.load_gs_points, 'iteration_*'))}
        last_saved = last_saved[max(list(last_saved.keys()))].replace('/', os.sep)
        points = torch.tensor(fetchPly(os.path.join(last_saved, 'point_cloud.ply')).points).to(mano_config.device)

    # sample points for matching
    perm = torch.randperm(points.shape[0])
    idx = perm[:min(points.shape[0],10000)]
    points = points[idx]
    vertices = mano_model().squeeze() # sample model

    transl,rot,scale = match_pca(points,vertices)

    mano_model.transl = torch.nn.Parameter(transl, requires_grad=False)
    mano_model.scale = torch.nn.Parameter(scale, requires_grad=False)
    mano_model.debug_scale = scale.clone()
    mano_model.rotation = torch.nn.Parameter(rot, requires_grad=False)

    vertices = mano_model()
    vertices = vertices.squeeze()
    faces = torch.tensor(mano_model.faces.astype(np.int32))
    faces = torch.squeeze(faces)
    faces = faces.to(mano_config.device).long()
    if mano_config.plot_reduction:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        _scatter(points,ax,'g',0.1)
        _scatter(vertices,ax,'r',0.15)
        plt.show()

    triangles = vertices[faces]
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = mano_config.points_per_face
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        ).to(mano_config.device)
        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        print("INIT xyz: ",xyz.mean(0),xyz.std(0))
        print("INIT vert:",vertices.mean(0),vertices.std(0))

        pcd = MANOPointCloud(
            alpha=alpha,
            points=xyz.cpu(),
            colors=SH2RGB(shs*0),
            normals=np.zeros((num_pts, 3)),
            mano_model=mano_model,
            faces=faces,
            vertices_init=vertices,
            transform_vertices_function=transform_vertices_mano,
            mano_model_shape_init=mano_model.shape,
            mano_model_pose_init=mano_model.pose,

            mano_model_scale_init=mano_model.scale,
            mano_model_transl_init=mano_model.transl,
            mano_model_rot_init=mano_model.rotation,
            vertices_enlargement_init=mano_config.vertices_enlargement
        )

    scene_info = InterHandsSceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           mano_config=mano_config)
    return scene_info


sceneLoadTypeCallbacks = {
}

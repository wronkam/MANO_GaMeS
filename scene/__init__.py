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
import concurrent.futures
import os
import random
import json
from functools import reduce
from typing import List

from games.mano_splatting.utils.general_utils import loadInterHandCam
from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class CameraListWrapper:
    def __init__(self, cameraList):
        self.cameraList = [cam for cam in cameraList if len(cam)>0]
        self.order: List[int] = reduce(lambda a, b: a + b,  # flatten
                                       [[idx for _ in range(len(cam))] for idx, cam in enumerate(self.cameraList)])
    def __len__(self):
        return len(self.cameraList)
    def __iter__(self):
        return self.cameraList.__iter__()
    def sample(self):
        """
        Get random camera with probability balanced by remaining usages
        Returns:
        """
        if len(self.order) == 0:
            self.order: List[int] = reduce(lambda a, b: a + b,  # flatten
                                           [[idx for _ in range(len(cam))] for idx, cam in enumerate(self.cameraList)])
        sample = random.randint(0, len(self.order) - 1)
        sample = self.order.pop(sample)
        return self.cameraList[sample]
    def __getitem__(self, idx):
        item = self.cameraList[idx]
        return item

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None,
                 shuffle=False, # no point shuffling, there is random used when picking cameras up
                 resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.mano_config = None
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_multi_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats, args.meshes
                )
            elif args.gs_type == "gs_mano":
                scene_info = sceneLoadTypeCallbacks["Colmap_MANO"](args.source_path, args.images, args.eval)
                self.mano_config = scene_info.mano_config
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if args.interhands:
                if self.mano_config.num_workers >= 1:
                    self.threads = concurrent.futures.ThreadPoolExecutor(max_workers=self.mano_config.num_workers)
                else:
                    self.threads = None
                cam_loader = lambda a,b,c,d: loadInterHandCam(a,b,c,d,self.mano_config,self.threads)
                print("Loading InterHands Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,args,loader=cam_loader)
                print("Loading InterHands Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,args,loader=cam_loader)
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.point_cloud = scene_info.point_cloud
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.scaled_train_cameras = {}
        self.scaled_test_cameras = {}

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        if scale not in self.scaled_train_cameras.keys():
            print("Loading Training Cameras")
            self.scaled_train_cameras[scale] = CameraListWrapper(self.train_cameras[scale])
        return self.scaled_train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        if scale not in self.scaled_test_cameras.keys():
            print("Loading Test Cameras")
            self.scaled_test_cameras[scale] = CameraListWrapper(self.test_cameras[scale])
        return self.scaled_test_cameras[scale]
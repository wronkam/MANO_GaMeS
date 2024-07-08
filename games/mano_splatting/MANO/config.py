import os
from typing import List

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManoConfig:
    def __init__(self):
        self.mano_rhand_path = os.path.join('games/mano_splatting/MANO/models/MANO_RIGHT.pkl')
        self.mano_lhand_path = os.path.join('games/mano_splatting/MANO/models/MANO_LEFT.pkl')
        self.InterHands_images_path = os.path.join('./data/InterHand/5/InterHand2.6M_5fps_batch1/images')
        self.InterHands_masks_path = os.path.join('./data/InterHand/5/InterHand2.6M_5fps_batch1/masks_removeblack')
        self.InterHands_annots_path = os.path.join(
            "./data/InterHand/5/annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json")
        self.InterHands_keys: List[str] = ['test', 'capture0', 'ROM04_RT_Occlusion']

        self.limit_frames = None
        self.limit_frames_center = 80

        self.load_all_images = True  # if not uses parallel preloading with following params
        self.load_on_gpu = False  # if not images are loaded to RAM and moved to gpu just for execution
        self.num_workers = 48
        self.preload_size = 12

        self.mano_is_rhand = True
        self.annot_path = None
        self.use_3D_translation = True
        self.batch_size = 1
        if self.batch_size != 1:
            raise NotImplementedError("graphic utils do not work with other batch sizes")
        self.pca = None
        self.device = device
        # repurposed as scales modifier
        # log2(ve+1+eps)
        self.vertices_enlargement = 4

        self.points_per_face: int = 250

        self.reduce_points: bool = False
        self.plot_reduction: bool = False
        self.reduction_start: torch.Tensor = None  # use mean if None
        self.reduction_knn: int = 9
        self.reduction_max_iters: int = 20
        self.reduction_theta: float = 0.3
        # use 2*running avg of mean distances from previous steps if None
        self.reduction_max_distance: float = 2e-2

        # OVERWRITES reduction
        self.load_gs_points: str = os.path.join('output/hands/point_cloud/iteration_10000/point_cloud.ply')

        self.scale_loss: int = 0  # 20

        self.use_adjustment_net: bool = True
        self.adjustment_warmup = 60000
        self.adjustment_net = {
            'frame_embedding': {
                'in': 1, 'out': 8,
                'width': 16, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 16), 'bn': False, 'dropout': 0.3},
            'mano_pose_shape_embedding': {
                'in': 48+10, 'out': 84,  # pose (always loaded), shape (learnable)
                'width': 96, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_geometric_embedding': {
                'in': 1+3+3, 'out': 7*16,  # scale, rotation, transl, 7*16=112
                'width': 64, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 16), 'bn': False, 'dropout': 0.3},
            'mano_embedding': {
                'in': 84+112, 'out': 128,  # mano_geometric_embedding, mano_geometric_embedding
                'width': 256, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_transl_net': {
               'in': 3, 'out': 3,
               'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_rotation_net': {
               'in': 3, 'out': 3,
               'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_scale_net': {
                'in': 1, 'out': 1,
                'width': 8, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_shape_net': {
                'in': 10, 'out': 10,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_pose_net': {
                'in': 48, 'out': 48,
                'width': 48, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'time_embedding': {
                'in': 128+8, 'out': 64,  # frame-embedding + mano_embedding
                'width': 128, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_adjustment': {  # input: time_embedding
                'in': 64, 'out': 1+3+3+48+10,  # out: scale, rotation, transl, pose, shape
                'width': 64, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            # ---------------------------------------------------------------------------------------------------------
            'joint_embedding': {  # alpha, scale
                'in': 3+1, 'out': 32*2,
                'width': 128, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 16), 'bn': True, 'dropout': 0.3},
            'main': {  # input: time_embedding, joint_embedding, face_embedding
                'in': 64+32*2, 'out': 3+1,  # 320->7
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'alpha_net': {
                'in': 3, 'out': 3,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'scale_net': {
                'in': 1, 'out': 1,
                'width': 8, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
        }

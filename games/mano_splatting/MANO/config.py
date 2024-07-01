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
        self.InterHands_annots_path = os.path.join("./data/InterHand/5/annotations/test/InterHand2.6M_test_MANO_NeuralAnnot.json")
        self.InterHands_keys: List[str] = ['test','capture0','ROM04_RT_Occlusion']

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

        self.points_per_face:int = 250

        self.reduce_points:bool = False
        self.plot_reduction:bool = False
        self.reduction_start:torch.Tensor = None # use mean if None
        self.reduction_knn:int = 9
        self.reduction_max_iters:int = 20
        self.reduction_theta:float = 0.3
        # use 2*running avg of mean distances from previous steps if None
        self.reduction_max_distance:float =  2e-2

        # OVERWRITES reduction
        self.load_gs_points:str = os.path.join('output/hands/point_cloud/iteration_10000/point_cloud.ply')

        self.scale_loss:int = 20

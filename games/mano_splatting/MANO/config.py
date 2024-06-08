import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManoConfig:
    def __init__(self):
        self.mano_rhand_path = os.path.join('games/mano_splatting/MANO/models/MANO_RIGHT.pkl')
        self.mano_lhand_path = os.path.join('games/mano_splatting/MANO/models/MANO_LEFT.pkl')
        self.mano_is_rhand = True
        self.annot_path = None
        self.use_3D_translation = True
        self.num_worker = True
        self.batch_size = 1
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

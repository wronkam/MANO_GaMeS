import os
import warnings
from typing import List, Dict

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
        self.InterHands_keys: List[str] = ['test', 'Capture0', 'ROM04_RT_Occlusion']

        self.limit_frames = 0  # if <=0 or None, all are loaded, otherwise take +-lim_fr around lim_fr_center
        if self.limit_frames <= 0:
            self.limit_frames = os.environ.get('LIMIT_FRAMES', None)
            self.limit_frames = None if self.limit_frames is None else int(self.limit_frames)
            if self.limit_frames <= 0:
                self.limit_frames = None
        self.limit_frames_center = 143  # position of image in sorted by name order

        self.load_all_images = True  # if not uses parallel preloading with following params
        self.load_on_gpu = False  # if not images are loaded to RAM and moved to gpu just for execution
        self.num_workers = 48  # num of threads in ThreadPool used for image preloading
        self.preload_size = 12  # how many images are buffered for future iterations for each camera

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

        # heuristic algorith that extracts hand from messy ColMap cloud point
        self.reduce_points: bool = False
        self.plot_reduction: bool = False
        self.reduction_start: torch.Tensor = None  # use mean if None
        self.reduction_knn: int = 9
        self.reduction_max_iters: int = 20
        self.reduction_theta: float = 0.3
        # use 2*running avg of mean distances from previous steps if None
        self.reduction_max_distance: float = 2e-2

        # OVERWRITES reduction, uses points form gs training, use only with properly masked hands
        self.load_gs_points: str = os.path.join('output/hands/point_cloud/')
        # both methods than match hand mesh to point cloud via 2-component PCA alignment, use plot_reduction to verify

        self.scale_loss: int = 0  # 20

        self.use_adjustment_net: bool = True
        self.chosen_variant = None
        if self.chosen_variant is None:
            self.chosen_variant = os.environ.get('MANO_VARIANT','ver')
        """
            none - no updates
            top - only mano (vertices) are updated
            gauss - update for all gaussians directly
            ver - gaussians updated with values adjusted with alpha (3 values per face @ gauss_alpha)
            ver+ - gaussians updated with vertice updated with added update_alpha (of ver_plus size)
        """
        ver_plus_size: int = 3

        self.adjustment_warmup = 40000
        embed_size: Dict[str, int] = {
            'pose_shape': 84,
            'geo': 16,
            'mano': 128,
            'frame': 16,
            'm_transl': 16,
            'm_rotation': 16,
            'm_scale': 16,
            'm_shape': 16,
            'm_pose': 64,
            'time': 64,
            'joint': 2 * 32,
            'alpha': 16,
            'scale': 16,
            'face': 32,
        }
        self.embed_size = embed_size
        self.adjustment_net = {
            'frame_embedding': {
                'in': 1, 'out': embed_size['frame'],
                'width': 32, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 32), 'bn': False, 'dropout': 0.3},
            'mano_pose_shape_embedding': {
                'in': 48 + 10, 'out': embed_size['pose_shape'],  # pose (always loaded), shape (learnable)
                'width': 96, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_geometric_embedding': {
                'in': 1 + 3 + 3, 'out': 7 * embed_size['geo'],  # scale 1, rotation 3, transl 3
                'width': 128, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 16), 'bn': False, 'dropout': 0.3},
            'mano_embedding': {
                'in': embed_size['pose_shape'] + 7 * embed_size['geo'],
                'out': embed_size['mano'],  # mano_geometric_embedding, mano_geometric_embedding
                'width': 256, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_transl_net': {
                'in': embed_size['m_transl'], 'out': 3,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_rotation_net': {
                'in': embed_size['m_rotation'], 'out': 3,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_scale_net': {
                'in': embed_size['m_scale'], 'out': 1,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_shape_net': {
                'in': embed_size['m_shape'], 'out': 10,
                'width': 16, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_pose_net': {
                'in': embed_size['m_pose'], 'out': 48,
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'time_embedding': {
                'in': embed_size['mano'] + embed_size['frame'],
                'out': embed_size['time'],  # frame-embedding + mano_embedding
                'width': 196, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_adjustment': {  # input: time_embedding
                'in': 64,  # out: scale, rotation, transl, pose, shape
                'out': embed_size['m_scale'] + embed_size['m_rotation']
                       + embed_size['m_transl'] + embed_size['m_pose'] + embed_size['m_shape'],
                'width': 256, 'depth': 2, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            # ---------------------------------------------------------------------------------------------------------
        }
        gauss_variant = {
            'joint_embedding': {  # alpha, scale
                'input': ['alpha', 'scale'],
                'in': 3 + 1, 'out': embed_size['joint'],
                'width': 128, 'depth': 2, 'activation': 'relu', 'embed': ('GFF+', 16),
                'bn': True, 'dropout': 0.3},
            'main': {  # input: time_embedding, joint_embedding
                'in': embed_size['time'] + embed_size['joint'],
                'out': embed_size['alpha'] + embed_size['scale'],
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'alpha_net': {
                'in': embed_size['alpha'], 'out': 3,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'scale_net': {
                'in': embed_size['scale'], 'out': 1,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
        }
        self.per_face_embed = 3 if self.chosen_variant == 'ver' else ver_plus_size
        ver_variant = {
            'joint_embedding': {  # face (for now, later use embedding params)
                'input': ['face'],
                'in': 3, 'out': embed_size['face'],
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('GFF+', 16),
                'bn': True, 'dropout': 0.3},
            'main': {  # input: time_embedding, joint_embedding
                'in': embed_size['time'] + embed_size['face'],
                'out': embed_size['alpha'] + embed_size['scale'],
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'alpha_net': {
                'in': embed_size['alpha'], 'out': 3 * self.per_face_embed,  # to be scaled with alphas
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'scale_net': {
                'in': embed_size['scale'], 'out': 1 * self.per_face_embed,  # to be scaled with alphas
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
        }
        variants = {'gauss': gauss_variant, 'ver': ver_variant, 'ver+': ver_variant, 'none': {}, 'top': {}}
        self.adjustment_net = self.adjustment_net | variants[self.chosen_variant]

        for name, net in self.adjustment_net.items():
            for key in ['in', 'out', 'width', 'depth', 'activation', 'embed', 'bn', 'dropout']:
                if key not in net.keys():
                    raise ValueError(f'Adjustment net {name} subnet does not have {key} defined')
            if net['in'] > net['width'] or net['out'] > net['width']:
                warnings.warn(f"Adjustment net {name}'s input or output is wider than the network.\n "
                              f"In:{net['in']}, Out:{net['out']}, Width:{net['width']}:")

    def __str__(self):
        out = ''
        for key, val in self.__dict__.items():
            if key.startswith('__'):
                continue
            out += f'{key} ::: {val}\n'
        return out

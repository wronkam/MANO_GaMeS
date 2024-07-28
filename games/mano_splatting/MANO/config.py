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

        self.first_capture = 17746  # used to sync frame numbers between cameras with different num of frames
        self.last_capture = 18652

        self.limit_frames = 0  # if <=0 or None, all are loaded, otherwise take +-lim_fr around lim_fr_center
        self.limit_frames_center = 77  # position of image in sorted by name order
        if self.limit_frames <= 0:
            self.limit_frames = os.environ.get('LIMIT_FRAMES', None)
            self.limit_frames = None if self.limit_frames is None else int(self.limit_frames)
            if self.limit_frames is not None and self.limit_frames <= 0:
                self.limit_frames = None

        self.load_all_images = True  # if not uses parallel preloading with following params
        self.load_on_gpu = False  # if not images are loaded to RAM and moved to gpu just for execution
        self.num_workers = 48  # num of threads in ThreadPool used for image preloading
        self.preload_size = 16  # how many images are buffered for future iterations for each camera

        enable_custom_camera_scheduling = True
        camera_scheduling_interval = 2500
        camera_scheduling_mode = 'neighbour'
        camera_number = 152  # how many
        self.camera_schedule: Dict[int, List[int]] = {0: [i for i in range(camera_number)]}  # use all from iteration 0
        # format {(iteration>=0 : int) : ([list of camera ids>=0: int]) incorrect camera_id are ignored
        # camera id denotes number of camera in sorted list of captures that are first_capture <= x <= last_capture
        if enable_custom_camera_scheduling:
            if camera_scheduling_mode == 'neighbour':
                # start with one camera at limit_frames_center and enable 1 from left and right each interval
                self.camera_schedule = {0: [self.limit_frames_center]}
                iters = max(camera_number - self.limit_frames_center, self.limit_frames_center) + 3
                for i in range(1, iters):
                    self.camera_schedule[i * camera_scheduling_interval] = [self.limit_frames_center - i,
                                                                            self.limit_frames_center + i]
        if self.limit_frames is not None and self.limit_frames > 0:
            for k, v in self.camera_schedule.items():
                self.camera_schedule[k] = [cam for cam in v if (
                        self.limit_frames_center - self.limit_frames <=
                        cam <= self.limit_frames_center + self.limit_frames)]
        self.camera_schedule = {t: frame_list for t, frame_list in self.camera_schedule.items() if len(frame_list) > 0}

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

        self.points_per_face: int = 300

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
            self.chosen_variant = os.environ.get('MANO_VARIANT', 'ver')
        """
            none - no updates
            top - only mano (vertices) are updated
            gauss - update for all gaussians directly
            ver - gaussians updated with values adjusted with alpha (3 values per face @ gauss_alpha)
            ver+ - gaussians updated with vertice updated with added update_alpha (of ver_plus size)
        """
        ver_plus_size: int = 6

        self.adjustment_warmup = int(os.environ.get('ADJUSTMENT_WARMUP', '40000'))
        embed_size: Dict[str, int] = {
            'pose_shape': 84,
            'geo': 32,
            'mano': 256,
            'frame': 32,
            'm_transl': 32,
            'm_rotation': 32,
            'm_scale': 32,
            'm_shape': 32,
            'm_pose': 96,
            'time': 360,
            'joint': 2 * 64,
            'alpha': 32,
            'scale': 32,
            'face': 64,
        }
        self.adjustment_net_out_activation = 'log_squish'
        self.embed_size = embed_size
        self.adjustment_net = {
            'frame_embedding': {
                'in': 1, 'out': embed_size['frame'],
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('GFF+', 32), 'bn': False, 'dropout': 0.3},
            'mano_pose_shape_embedding': {
                'in': 48 + 10, 'out': embed_size['pose_shape'],  # pose (always loaded), shape (learnable)
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_geometric_embedding': {
                'in': 1 + 3 + 3, 'out': 7 * embed_size['geo'],  # scale 1, rotation 3, transl 3
                'width': 256, 'depth': 4, 'activation': 'relu', 'embed': ('GFF+', 16), 'bn': False, 'dropout': 0.3},
            'mano_embedding': {
                'in': embed_size['pose_shape'] + 7 * embed_size['geo'],
                'out': embed_size['mano'],  # mano_geometric_embedding, mano_geometric_embedding
                'width': 512, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_transl_net': {
                'in': embed_size['m_transl'], 'out': 3,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_rotation_net': {
                'in': embed_size['m_rotation'], 'out': 3,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_scale_net': {
                'in': embed_size['m_scale'], 'out': 1,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_shape_net': {
                'in': embed_size['m_shape'], 'out': 10,
                'width': 32, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_pose_net': {
                'in': embed_size['m_pose'], 'out': 48,
                'width': 128, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'time_embedding': {
                'in': embed_size['mano'] + embed_size['frame'],
                'out': embed_size['time'],  # frame-embedding + mano_embedding
                'width': 512, 'depth': 8, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            'mano_adjustment': {  # input: time_embedding
                'in': embed_size['time'],  # out: scale, rotation, transl, pose, shape
                'out': embed_size['m_scale'] + embed_size['m_rotation']
                       + embed_size['m_transl'] + embed_size['m_pose'] + embed_size['m_shape'],
                'width': 512, 'depth': 6, 'activation': 'relu', 'embed': ('None', None), 'bn': False, 'dropout': 0.3},
            # ---------------------------------------------------------------------------------------------------------
        }
        gauss_variant = {
            'joint_embedding': {  # alpha, scale
                'input': ['alpha', 'scale'],
                'in': 3 + 1, 'out': embed_size['joint'],
                'width': 512, 'depth': 4, 'activation': 'relu', 'embed': ('GFF+', 16),
                'bn': True, 'dropout': 0.3},
            'main': {  # input: time_embedding, joint_embedding
                'in': embed_size['time'] + embed_size['joint'],
                'out': embed_size['alpha'] + embed_size['scale'],
                'width': 512, 'depth': 6, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'alpha_net': {
                'in': embed_size['alpha'], 'out': 3,
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'scale_net': {
                'in': embed_size['scale'], 'out': 1,
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
        }
        self.per_face_embed = 3 if self.chosen_variant == 'ver' else ver_plus_size
        ver_variant = {
            'joint_embedding': {  # face (for now, later use embedding params)
                'input': ['face'],
                'in': 3, 'out': embed_size['face'],
                'width': 512, 'depth': 6, 'activation': 'relu', 'embed': ('GFF+', 16),
                'bn': True, 'dropout': 0.3},
            'main': {  # input: time_embedding, joint_embedding
                'in': embed_size['time'] + embed_size['face'],
                'out': embed_size['alpha'] + embed_size['scale'],
                'width': 512, 'depth': 6, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'alpha_net': {
                'in': embed_size['alpha'], 'out': 3 * self.per_face_embed,  # to be scaled with alphas
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
            'scale_net': {
                'in': embed_size['scale'], 'out': 1 * self.per_face_embed,  # to be scaled with alphas
                'width': 64, 'depth': 4, 'activation': 'relu', 'embed': ('None', None), 'bn': True, 'dropout': 0.3},
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

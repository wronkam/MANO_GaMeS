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
from typing import Dict

import torch
import numpy as np
from games.mano_splatting.MANO import ManoConfig

from torch import nn

from arguments_games import OptimizationParamsMano
from games.mano_splatting.scene.adjustment_net import AdjustmentNet
from games.mano_splatting.utils.graphics_utils import MANOPointCloud
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from games.mano_splatting.utils.general_utils import rot_to_quat_batch
from utils.sh_utils import RGB2SH

copy_as_zeros = lambda x: torch.zeros_like(x.detach()).to(x.device)


class GaussianManoModel(GaussianModel):
    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud: MANOPointCloud = None
        self._alpha = torch.empty(0)
        self._adjustment_alpha = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.warm_up_steps = 15000
        self.warm_up = 0

        def taylor_softmax(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
            x = 1 + x + 0.5 * x ** 2 + eps / x.shape[-1]
            return x / x.sum(-1, keep_dim=True)

        # thanks to tactical ambiguity, m can be flat or same shaped tensor
        def soft_margin_taylor_softmax(x: torch.Tensor, m=None, eps: float = 1e-9) -> torch.Tensor:
            if m is None:
                m = x.var(-2, keepdim=True) ** (1 / 3)
            x1 = 1 + x + 0.5 * x ** 2 + eps / x.shape[-1]
            x = 1 + (x - m) + 0.5 * (x - m) ** 2 + eps / x.shape[-1]
            div = torch.sum(x1, -1, True) - x1 + x
            x = x / div
            return x / x.sum(-1, True)

        self.update_alpha_func = lambda x: soft_margin_taylor_softmax(x=x)

        self.vertices = None
        self.faces = None
        self._scales = torch.empty(0)
        self.faces = torch.empty(0)
        self._vertices_enlargement = torch.empty(0)
        self.time_frame = torch.tensor([-1], dtype=torch.int)

        self.mano_config: ManoConfig = None
        self.adjustment_net: AdjustmentNet = None
        self.adjustments: Dict[str, torch.Tensor] = {}
        self.mano_adjustments: Dict[str, torch.Tensor] = {}
        self.time_embedding = torch.empty(0)

        self._mano_shape = torch.empty(0)
        self._mano_pose = torch.empty(0)
        self._mano_transl = torch.empty(0)
        self._mano_rotation = torch.empty(0)
        self._mano_scale = torch.empty(0)

    @property
    def get_xyz(self):
        return self._xyz

    def reload_mano_pose(self, pose, time_frame=-1):
        self._mano_pose = torch.tensor(pose, dtype=self._mano_pose.dtype, requires_grad=False
                                       ).repeat(self._mano_pose.shape[0], 1).to(self._mano_pose.device)
        self._generate_adjustments(time_frame)
        self.update_alpha()
        self.prepare_scaling_rot()
        self.time_frame = time_frame

    def _generate_adjustments(self, time=None):
        self._generate_mano_adjustments(time)
        self._generate_gauss_adjustments()
        self.warm_up = min(self.warm_up + 1, self.warm_up_steps)

    def _generate_gauss_adjustments(self):
        if not self.mano_config.use_adjustment_net or self.mano_config.chosen_variant in ['none', 'top']:
            return
        assert self.time_embedding is not None
        input = {
            'alpha': self._alpha.detach(),
            'scale': self._scales.detach(),
            'face': self.point_cloud.mano_model.get_vertice_embedding().detach(),
        }
        self.adjustments = self.adjustment_net(
            values=input,
            time_embedding=self.time_embedding.detach()
        )

    def _generate_mano_adjustments(self, time=None):
        if not self.mano_config.use_adjustment_net or self.mano_config.chosen_variant in ['none']:
            return
        if time is not None:
            if type(time) is not torch.Tensor:
                time = torch.tensor([time], dtype=torch.int)
            # even if time == time_frame, make new adjustments as nets changed
            self.time_frame = time
        if self.time_frame.device != self._mano_pose.device:
            self.time_frame = self.time_frame.to(self._mano_pose.device)

        self.time_embedding = self.adjustment_net.get_time_embedding(
            frame=self.time_frame.detach(),
            mano_pose=self._mano_pose.detach(),
            mano_shape=self._mano_shape.detach(),
            scale=self._mano_scale.detach(),
            rotation=self._mano_rotation.detach(),
            transl=self._mano_transl.detach(),
        )

        self.mano_adjustments = self.adjustment_net.get_mano_adjustments(self.time_embedding)

    def create_from_pcd(self, pcd: MANOPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        self.mano_config = self.point_cloud.mano_model.config
        print(self.mano_config)
        self.adjustment_net = AdjustmentNet(self.mano_config).cuda()
        self.warm_up_steps = self.mano_config.adjustment_warmup

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scales = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.create_mano_params()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))
        if self.mano_config.chosen_variant in ['ver+']:
            alpha_shape = tuple(list(alpha_point_cloud.shape[:-1]) + [self.mano_config.per_face_embed])
            adjustment_alpha = torch.zeros(alpha_shape).to(self._alpha.device)
            self._adjustment_alpha = nn.Parameter(adjustment_alpha, requires_grad=True)
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        with torch.no_grad():
            self.reload_mano_pose(self._mano_pose)

    def create_mano_params(self):
        """
        Create manipulation parameters MANO model.

        Each parameter is responsible for something different,
        respectively: shape, pose, and orientation in space etc.
        """
        self._mano_shape = self.point_cloud.mano_model_shape_init.requires_grad_(False)
        self._mano_pose = self.point_cloud.mano_model_pose_init.requires_grad_(False)
        self._mano_transl = nn.Parameter(self.point_cloud.mano_model_transl_init.requires_grad_(True))
        self._mano_rotation = nn.Parameter(self.point_cloud.mano_model_rot_init.requires_grad_(True))
        self._mano_scale = nn.Parameter(self.point_cloud.mano_model_scale_init.requires_grad_(True))
        self.faces = self.point_cloud.faces

        vertices_enlargement = (self.point_cloud.vertices_enlargement_init * torch.ones(1)).requires_grad_(True)
        vertices_enlargement = vertices_enlargement.to(self.faces.device)
        self._vertices_enlargement = nn.Parameter(vertices_enlargement)

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.
        """
        _xyz = torch.matmul(
            self.alpha,  # was adjusted
            self.vertices[self.faces]
        )
        self._xyz = _xyz.reshape(
            _xyz.shape[0] * _xyz.shape[1], 3
        )

    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid
        to 2nd vertex onto subspace spanned by v0 and v1
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        triangles = self.vertices[self.faces]
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        s0 = eps * torch.ones_like(s1)
        scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
        scales = scales.broadcast_to((*self.alpha.shape[:2], 3))
        if 'scale' not in self.adjustments:
            self.adjustments['scale'] = copy_as_zeros(self._scales)

        scale_update = self.adjustments['scale']

        if (self.mano_config.chosen_variant in ['ver', 'ver+']
                and scale_update.shape != self._scales.shape):
            # scale is [face x 3] needs to be [face*gauss x 1], flatten alpha and dot product/length
            adjustment_alpha = self._alpha if self.mano_config.chosen_variant == 'ver' else self._adjustment_alpha
            adjustment_alpha = self.update_alpha_func(adjustment_alpha)
            scale_update = scale_update.unsqueeze(1).repeat(1, adjustment_alpha.shape[1], 1).flatten(0, 1)
            scale_update = torch.mean(scale_update * adjustment_alpha.flatten(0, -2), dim=-1).unsqueeze(-1)
        self._scaling = torch.log(
            torch.nn.functional.relu(
                torch.log2(torch.nn.functional.relu(self._vertices_enlargement) + 1 + eps)
                * (self._scales + scale_update * (self.warm_up / self.warm_up_steps))
                * scales.flatten(start_dim=0, end_dim=1)) + eps
        )

        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        #TODO
        check:
        # self.alpha = torch.relu(self._alpha)
        # self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)

        """
        if 'alpha' not in self.adjustments:
            # only on first load, xyz and scales are not yet set up
            self.adjustments['alpha'] = torch.flatten(copy_as_zeros(self._alpha), 0, 1)
            # xyz, and scale have face x id dimensions collapsed
        if (self.mano_config.chosen_variant in ['ver', 'ver+'] and
                torch.numel(self.adjustments['alpha']) != torch.numel(self._alpha)):
            adjustment_alpha = self._alpha if self.mano_config.chosen_variant == 'ver' else self._adjustment_alpha
            adjustment_alpha = self.update_alpha_func(adjustment_alpha)
            # update_alpha.shape = [face x gs_per_face x n]
            # adjustment has shape: [face x (3n)]
            # update_alpha2 = [face x gs_per_face x 1 x n]
            #  [face x 3*n] -> [face x 3 x n] -> [face x 1 x 3 x n] -> [face x gs_per_face x 3 x n] X update_alpha
            #  -> [face x gs_per_face x 3] via dot product of last dims
            # than flatten it to fit xyz and scale
            n = self.mano_config.per_face_embed
            self.adjustments['alpha'] = self.adjustments['alpha'].reshape(-1, 3, n)
            self.adjustments['alpha'] = self.adjustments['alpha'].unsqueeze(1).repeat(1, self._alpha.shape[1], 1, 1)
            assert adjustment_alpha.shape[-1] == n
            self.adjustments['alpha'] = torch.mean(self.adjustments['alpha'] * adjustment_alpha.unsqueeze(-2), dim=-1)
            self.adjustments['alpha'] = torch.flatten(self.adjustments['alpha'], 0, 1)
            # to fit xyz, and scale shape to be consistent with gauss mode

        alpha_d = torch.reshape(self.adjustments['alpha'], self._alpha.shape)
        self.alpha = self.update_alpha_func(self._alpha + alpha_d * (self.warm_up / self.warm_up_steps))
        if 'scale' not in self.mano_adjustments:
            # used only on model load
            self.mano_adjustments = {
                'mano_shape': copy_as_zeros(self._mano_shape),
                'mano_pose': copy_as_zeros(self._mano_pose),
                'transl': copy_as_zeros(self._mano_transl),
                'scale': copy_as_zeros(self._mano_scale),
                'rotation': copy_as_zeros(self._mano_rotation),
            }
            self._generate_mano_adjustments()
        frac = self.warm_up / self.warm_up_steps
        vertices = self.point_cloud.mano_model(
            shape=self._mano_shape.detach() + self.mano_adjustments['mano_shape'] * frac,
            pose=self._mano_pose.detach() + self.mano_adjustments['mano_pose'] * frac,
            transl=self._mano_transl.detach() + self.mano_adjustments['transl'] * frac,
            scale=self._mano_scale.detach() + self.mano_adjustments['scale'] * frac,
            rotation=self._mano_rotation.detach() + self.mano_adjustments['rotation'] * frac,
        )
        self.vertices = self.point_cloud.transform_vertices_function(
            vertices,
            None
        )
        self._calc_xyz()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        lr_params = [
            {'params': [self._mano_scale], 'lr': training_args.mano_scale_lr, "name": "m_scale"},
            {'params': [self._mano_rotation], 'lr': training_args.mano_rot_lr, "name": "m_rotation"},
            {'params': [self._mano_transl], 'lr': training_args.mano_transl_lr, "name": "transl"},

            {'params': [self._vertices_enlargement], 'lr': training_args.vertices_enlargement_lr,
             "name": "vertices_enlargement"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},

            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._scales], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},

            {'params': self.adjustment_net.parameters(), 'lr': training_args.adjustment_net_lr,
             "name": "adjustment_net"},
            {'params': [self._adjustment_alpha], 'lr': training_args.update_alpha_lr, "name": "update_alpha"},
        ]

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step."""
        pass

    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        mano_additional_attrs = [
            '_mano_scale', '_mano_rotation', '_mano_transl',
            '_vertices_enlargement', 'faces',
            'alpha', 'point_cloud',
        ]

        save_dict = {}
        for attr_name in mano_additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_flame = path.replace('point_cloud.ply', 'mano_params.pt')
        torch.save(save_dict, path_flame)

    def load_ply(self, path):
        self._load_ply(path)
        path_flame = path.replace('point_cloud.ply', 'mano_params.pt')
        params = torch.load(path_flame)
        self._mano_scale = params['_mano_scale']
        self._mano_rotation = params['_mano_rotation']
        self._mano_transl = params['_mano_transl']
        self._vertices_enlargement = params['_vertices_enlargement']
        self.faces = params['faces']
        self.alpha = params['alpha']
        self.point_cloud = params['point_cloud']

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._mano_pose,
            self._mano_shape,
            self._mano_rotation,
            self._mano_transl,
            self._mano_scale,
            self.adjustment_net.state_dict()
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self._mano_pose,
         self._mano_shape,
         self._mano_rotation,
         self._mano_transl,
         self._mano_scale,
         adj_net_dict
         ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.adjustment_net.load_state_dict(adj_net_dict)

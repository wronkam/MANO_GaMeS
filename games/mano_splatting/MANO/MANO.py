""" # TODO: new licence
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
import json
import os

# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import numpy as np
import smplx
import torch
import torch.nn as nn
import pickle
from games.mano_splatting.MANO.config import ManoConfig
from smplx.lbs import lbs, batch_rodrigues, vertices2landmarks, find_dynamic_lmk_idx_and_bcoords
from smplx.utils import Struct, to_tensor, to_np, rot_mat_to_euler
from games.mano_splatting.utils.graphics_utils import get_vertices, transform_vertices


class MANO(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 3D facial landmarks
    """
    def __init__(self, config:ManoConfig):
        super(MANO, self).__init__()
        print("creating the MANO Decoder")
        if config.annot_path is not None:
            with open(config.annot_path,'r') as f_annot:
                annots = json.load(f_annot)
        else:
            annots = {'is_rhand':config.mano_is_rhand}
        mano_model_path = config.mano_rhand_path if annots['is_rhand'] else config.mano_lhand_path
        # TODO: make ManoModel, use pcd for transl and scale
        #  and compute PCA to get rotation (if no annots)
        #  or load some optional config file (done in annots)
        #  remember to checkpoint mano params separately
        self.mean_xyz = None
        self.mano_model = smplx.create(mano_model_path,
                                  model_type='mano',
                                  use_pca=(config.pca is not None),
                                  num_pca_comps = 6 if config.pca is None else config.pca,
                                  is_rhand=annots['is_rhand'],
                                  batch_size=config.batch_size,
                                  # flat_hand_mean=True,
                                  )
        self.annots = annots
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.faces = self.mano_model.faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.faces, dtype=np.int64),
                                       dtype=torch.long))
        # TODO:
        #  consider adding Learnable wrapper from render matching for smaller/all params
        if 'pose' in annots.keys():
            default_pose = torch.tensor([annots['pose']],
                                         dtype=self.dtype, requires_grad=False).repeat(config.batch_size,1)
        else:
            default_pose = torch.zeros([self.batch_size, 48],
                                        dtype=self.dtype, requires_grad=False)
        self.register_parameter('pose', nn.Parameter(default_pose,
                                                            requires_grad=False))

        if 'shape' in annots.keys():
            default_shape = torch.tensor([annots['shape']],
                                        dtype=self.dtype, requires_grad=False).repeat(config.batch_size,1)
        else:
            default_shape = torch.zeros([self.batch_size, 10],
                                        dtype=self.dtype, requires_grad=False)
        self.register_parameter('shape', nn.Parameter(default_shape,
                                                            requires_grad=False))

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        if 'trans' in annots.keys():
            default_transl = torch.tensor([annots['trans']],
                                          dtype=self.dtype, requires_grad=False).repeat(config.batch_size,1)
        else:
            default_transl = torch.zeros([self.batch_size, 3],
                                         dtype=self.dtype, requires_grad=False)
        self.register_parameter('transl', nn.Parameter(default_transl,
                                                       requires_grad=False))

        if 'rotation' in annots.keys():
            default_rotation = torch.tensor([annots['rotation']],
                                          dtype=self.dtype, requires_grad=False).repeat(config.batch_size,1)
        else:
            default_rotation = torch.zeros([self.batch_size, 3],
                                         dtype=self.dtype, requires_grad=False)
        self.register_parameter('rotation', nn.Parameter(default_rotation,
                                                       requires_grad=False))

        if 'scale' in annots.keys():
            default_scale = torch.tensor([annots['scale']],
                                          dtype=self.dtype, requires_grad=False).repeat(config.batch_size,1)
        else:
            default_scale = torch.ones([self.batch_size, 1],
                                         dtype=self.dtype, requires_grad=False)
        self.register_parameter('scale', nn.Parameter(default_scale,
                                                       requires_grad=False))

        self.scale_loss = config.scale_loss # not the prettiest bypass to train


    def forward(self, shape=None, pose=None, transl=None, scale = None, rotation = None):
        # TODO
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        shape    = (shape    if shape    is not None else self.shape   )
        pose     = (pose     if pose     is not None else self.pose    )
        root_pose = pose.view(-1, 3)[0].view(1, 3)
        hand_pose = pose.view(-1, 3)[1:, :].view(1, -1)

        if self.use_3D_translation:
            scale    = (scale    if scale    is not None else self.scale   )
            rotation = (rotation if rotation is not None else self.rotation)
            transl = (transl if transl is not None else self.transl)
        else:
            scale    = torch.ones_like(self.scale)
            rotation = torch.zeros_like(self.rotation)
            transl = torch.zeros_like(self.transl)

        vertices, mean = get_vertices(self.mano_model,hand_pose,root_pose,
                                rotation,scale,shape,transl,self.mean_xyz)
        if self.mean_xyz is None:
            self.mean_xyz = mean

        return vertices

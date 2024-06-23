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

from arguments import OptimizationParams
from arguments_games import OptimizationParamsMano

from scene.gaussian_model import GaussianModel
from games.mano_splatting.scene.gaussian_mano_model import GaussianManoModel

optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
    "gs_mano": OptimizationParamsMano,
}
gaussianModel = {
    "gs": GaussianModel,
    "gs_mano": GaussianManoModel,
}

gaussianModelRender = {
    "gs": GaussianModel,
    "gs_mano": GaussianManoModel,
}

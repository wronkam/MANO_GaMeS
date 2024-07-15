from typing import Dict

import torch
import torch.nn as nn
from games.mano_splatting.MANO import ManoConfig
from games.mano_splatting.scene.embeddings import PositionalEncoding, GFFT1D

activation = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'identity': nn.Identity,
}
embeddings = {
    'PE': lambda size: PositionalEncoding(mapping_size=size, learnable=False),
    'PE+': lambda size: PositionalEncoding(mapping_size=size, learnable=True),
    'GFF': lambda size: GFFT1D(mapping_size=size, learnable=False),
    'GFF+': lambda size: GFFT1D(mapping_size=size, learnable=True),
    'None': lambda _: nn.Identity(),
}


def get_net(net_config) -> nn.Module:
    in_size = net_config['in']
    out_size = net_config['out']
    width = net_config['width']
    depth = net_config['depth']

    layers = []
    embed_type, map_size = net_config['embed']
    layers.append(embeddings[embed_type](map_size))
    if map_size is not None:
        in_size = in_size * map_size

    in_sizes = [in_size] + [width for _ in range(depth - 1)]
    out_sizes = [width for _ in range(depth - 1)] + [out_size]
    for idx, (in_s, out_s) in enumerate(zip(in_sizes, out_sizes)):
        layers.append(nn.Linear(in_s, out_s, bias=True))
        # batch_norm -> activation -> dropout
        if idx == depth - 1:
            # do not add after last layer
            continue
        if net_config['bn']:
            layers.append(nn.BatchNorm1d(out_s))
        layers.append(activation[net_config['activation']]())
        if net_config['dropout'] > 0.0:
            layers.append(nn.Dropout(net_config['dropout']))
    return nn.Sequential(*layers)


class AdjustmentNet(nn.Module):
    def __init__(self, config: ManoConfig):
        super().__init__()
        nets = {}
        self.sizes = config.embed_size
        for name, net_config in config.adjustment_net.items():
            nets[name] = get_net(net_config)
        self.nets = nn.ModuleDict(nets)
        self.schema = config.adjustment_net

    def get_time_embedding(self, frame, mano_pose, mano_shape, scale, rotation, transl):
        frame_in = frame.unsqueeze(0)  # [1] -> [B=1, 1]
        frame_embedding = self.nets['frame_embedding'](frame_in)

        mano_geo_input = torch.cat([scale, rotation, transl], dim=-1)
        mano_geo_embed = self.nets['mano_geometric_embedding'](mano_geo_input)

        mano_ps_input = torch.cat([mano_shape, mano_pose], dim=-1)
        mano_ps_embed = self.nets['mano_pose_shape_embedding'](mano_ps_input)

        mano_input = torch.cat([mano_geo_embed, mano_ps_embed], dim=-1)
        mano_embedding = self.nets['mano_embedding'](mano_input)

        joint_embedding = torch.cat([mano_embedding, frame_embedding], dim=-1)
        time_embedding = self.nets['time_embedding'](joint_embedding)

        return time_embedding

    def get_mano_adjustments(self, time_embedding):
        mano_adjustment = self.nets['mano_adjustment'](time_embedding)

        scale_idx = self.sizes['m_scale']
        rotation_idx = scale_idx+self.sizes['m_rotation']
        translation_idx = rotation_idx+self.sizes['m_transl']
        shape_idx = translation_idx+self.sizes['m_shape']

        (scale_d, rotation_d, transl_d, shape_d, pose_d) = (
            mano_adjustment[..., :scale_idx],
            mano_adjustment[..., scale_idx:rotation_idx],
            mano_adjustment[..., rotation_idx:translation_idx],
            mano_adjustment[..., translation_idx:shape_idx],
            mano_adjustment[..., shape_idx:])

        scale_d = self.nets['mano_scale_net'](scale_d)
        rotation_d = self.nets['mano_rotation_net'](rotation_d)
        transl_d = self.nets['mano_transl_net'](transl_d)
        shape_d = self.nets['mano_shape_net'](shape_d)
        pose_d = self.nets['mano_pose_net'](pose_d)

        return {'scale': scale_d, 'rotation': rotation_d, 'transl': transl_d,
                'mano_shape': shape_d, 'mano_pose': pose_d}

    def forward(self, values: Dict[str,torch.Tensor], time_embedding):

        input_keys = self.schema['joint_embedding']['input']
        if 'alpha' in input_keys:
            values['alpha'] = torch.flatten(values['alpha'],0,1)  # has size face x id x 3 not [face x id ] x 3
        input_values = [values[key] for key in input_keys]
        input_values = torch.cat(input_values, dim=-1)  # cat on 1 tensor does nothing so it is ok
        joint_embedding = self.nets['joint_embedding'](input_values)

        time_embedding = time_embedding.repeat(joint_embedding.shape[0],1)
        main_input = torch.cat([joint_embedding, time_embedding], dim=-1)
        out = self.nets['main'](main_input)

        alpha_d, scale_d = out[..., :self.sizes['alpha']], out[..., self.sizes['alpha']:]

        alpha_d = self.nets['alpha_net'](alpha_d)
        scale_d = self.nets['scale_net'](scale_d)

        return {'alpha': alpha_d, 'scale': scale_d}

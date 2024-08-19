# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proposal network field.
"""


from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field

from nerfstudio.fields.neurodin_field import LaplaceDensity
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class HashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear=False,
        num_levels=8,
        max_res=1024,
        base_res=16,
        log2_hashmap_size=18,
        features_per_level=2,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if not self.use_linear:
            self.mlp_base = tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=1,
                encoding_config=config["encoding"],
                network_config=config["network"],
            )
        else:
            self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])
            self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)

    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        if not self.use_linear:
            density_before_activation = (
                self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)
            )
        else:
            x = self.encoding(positions_flat).to(positions)
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}

class HashMLPSDFField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 256,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear=False,
        num_levels=16,
        max_res=2048,
        base_res=32,
        log2_hashmap_size=19,
        features_per_level=2,
        beta_init_val=0.1
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        # self.mlp_base = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3,
        #     n_output_dims=257,
        #     encoding_config=config["encoding"],
        #     network_config=config["network"],
        # )
        self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])
        self.linear = torch.nn.Linear(self.encoding.n_output_dims, 257)

        self.laplace_density = LaplaceDensity(init_val=beta_init_val)
        self.beta = None

    def get_sdf(self, ray_samples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples, self.aabb)
        positions_flat = positions.view(-1, 3)

        # sdf = self.mlp_base(positions_flat).view(*ray_samples.shape[:-1], -1).to(positions)
        x = self.encoding(positions_flat).to(positions)
        sdf = self.linear(x).view(*ray_samples.shape[:-1], -1)[..., 0:1]

        return sdf
    
    def get_density(self, ray_samples, return_sdf=False):
        if return_sdf:
            with torch.enable_grad():
                ray_samples.requires_grad_(True)
                sdf = self.get_sdf(ray_samples)
            density = self.laplace_density(sdf.detach(), self.beta)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf, inputs=ray_samples, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            return density, F.normalize(gradients)
        sdf = self.get_sdf(ray_samples)
        density = self.laplace_density(sdf, self.beta)
        return density, None
    
    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
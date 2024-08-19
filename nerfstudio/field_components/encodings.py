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
Encoding functions
"""

from abc import abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.field_components.base_field_component import FieldComponent
from nerfstudio.utils.math import components_from_spherical_harmonics, expected_sin
from nerfstudio.utils.printing import print_tcnn_speed_warning

try:
    import tinycudann as tcnn

    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False


class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError


class Identity(Encoding):
    """Identity encoding (Does not modify input)"""

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return in_tensor


class ScalingAndOffset(Encoding):
    """Simple scaling and offet to input

    Args:
        in_dim: Input dimension of tensor
        scaling: Scaling applied to tensor.
        offset: Offset applied to tensor.
    """

    def __init__(self, in_dim: int, scaling: float = 1.0, offset: float = 0.0) -> None:
        super().__init__(in_dim)

        self.scaling = scaling
        self.offset = offset

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        return self.in_dim

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return self.scaling * in_tensor + self.offset


class NeRFEncoding(Encoding):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        off_axis: bool = False,
        shift: int = 0
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input
        self.shift = shift

        self.off_axis = off_axis

        self.P = torch.tensor(
            [
                [0.8506508, 0, 0.5257311],
                [0.809017, 0.5, 0.309017],
                [0.5257311, 0.8506508, 0],
                [1, 0, 0],
                [0.809017, 0.5, -0.309017],
                [0.8506508, 0, -0.5257311],
                [0.309017, 0.809017, -0.5],
                [0, 0.5257311, -0.8506508],
                [0.5, 0.309017, -0.809017],
                [0, 1, 0],
                [-0.5257311, 0.8506508, 0],
                [-0.309017, 0.809017, -0.5],
                [0, 0.5257311, 0.8506508],
                [-0.309017, 0.809017, 0.5],
                [0.309017, 0.809017, 0.5],
                [0.5, 0.309017, 0.809017],
                [0.5, -0.309017, 0.809017],
                [0, 0, 1],
                [-0.5, 0.309017, 0.809017],
                [-0.809017, 0.5, 0.309017],
                [-0.809017, 0.5, -0.309017],
            ]
        ).T

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2

        if self.off_axis:
            out_dim = self.P.shape[1] * self.num_frequencies * 2

        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: TensorType["bs":..., "input_dim"],
        covs: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        # TODO check scaling here but just comment it for now
        # in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** (torch.linspace(self.min_freq, self.max_freq, self.num_frequencies)+self.shift).to(in_tensor.device)            ##########
        # freqs = 2 ** (
        #    torch.sin(torch.linspace(self.min_freq, torch.pi / 2.0, self.num_frequencies)) * self.max_freq
        # ).to(in_tensor.device)
        # freqs = 2 ** (
        #     torch.linspace(self.min_freq, 1.0, self.num_frequencies).to(in_tensor.device) ** 0.2 * self.max_freq
        # )

        if self.off_axis:
            scaled_inputs = torch.matmul(in_tensor, self.P.to(in_tensor.device))[..., None] * freqs
        else:
            scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs


class RFFEncoding(Encoding):
    """Random Fourier Feature encoding. Supports integrated encodings.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoding frequencies
        scale: Std of Gaussian to sample frequencies. Must be greater than zero
        include_input: Append the input coordinate to the encoding
    """

    def __init__(self, in_dim: int, num_frequencies: int, scale: float, include_input: bool = False) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        if not scale > 0:
            raise ValueError("RFF encoding scale should be greater than zero")
        self.scale = scale
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        b_matrix = torch.normal(mean=0, std=self.scale, size=(self.in_dim, self.num_frequencies))
        self.register_buffer(name="b_matrix", tensor=b_matrix)
        self.include_input = include_input

    def get_out_dim(self) -> int:
        return self.num_frequencies * 2

    def forward(
        self,
        in_tensor: TensorType["bs":..., "input_dim"],
        covs: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates RFF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.

        Returns:
            Output values will be between -1 and 1
        """
        in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        scaled_inputs = in_tensor @ self.b_matrix  # [..., "num_frequencies"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.sum((covs @ self.b_matrix) * self.b_matrix, -2)
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )

        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)

        return encoded_inputs


class HashEncoding(Encoding):
    """Hash encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
        interpolation: Interpolation override for tcnn hashgrid. Not supported for torch unless linear.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:

        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_table_size = 2**log2_hashmap_size

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        self.tcnn_encoding = None
        if not TCNN_EXISTS and implementation == "tcnn":
            print_tcnn_speed_warning("HashEncoding")
        elif implementation == "tcnn":
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": growth_factor,
            }
            if interpolation is not None:
                encoding_config["interpolation"] = interpolation

            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config=encoding_config,
            )

        if not TCNN_EXISTS or self.tcnn_encoding is None:
            assert (
                interpolation is None or interpolation == "Linear"
            ), f"interpolation '{interpolation}' is not supported for torch encoding backend"

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: TensorType["bs":..., "num_levels", 3]) -> TensorType["bs":..., "num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # min_val = torch.min(in_tensor)
        # max_val = torch.max(in_tensor)
        # assert min_val >= 0.0
        # assert max_val <= 1.0

        in_tensor = in_tensor * torch.tensor([1, 2654435761, 805459861]).to(in_tensor.device)
        x = torch.bitwise_xor(in_tensor[..., 0], in_tensor[..., 1])
        x = torch.bitwise_xor(x, in_tensor[..., 2])
        x %= self.hash_table_size
        x += self.hash_offset.to(x.device)
        return x

    def pytorch_fwd(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        if TCNN_EXISTS and self.tcnn_encoding is not None:
            return self.tcnn_encoding(in_tensor)
        return self.pytorch_fwd(in_tensor)


class TensorCPEncoding(Encoding):
    """Learned CANDECOMP/PARFAC (CP) decomposition encoding used in TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    def __init__(self, resolution: int = 256, num_components: int = 24, init_scale: float = 0.1) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components

        # TODO Learning rates should be different for these
        self.line_coef = nn.Parameter(init_scale * torch.randn((3, num_components, resolution, 1)))

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        line_coord = torch.stack([torch.zeros_like(line_coord), line_coord], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        line_coord = line_coord.view(3, -1, 1, 2).detach()

        line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        features = torch.prod(line_features, dim=0)
        features = torch.moveaxis(features.view(self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """

        self.line_coef.data = F.interpolate(
            self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True
        )

        self.resolution = resolution


class TensorVMEncoding(Encoding):
    """Learned vector-matrix encoding proposed by TensoRF

    Args:
        resolution: Resolution of grid.
        num_components: Number of components per dimension.
        init_scale: Initialization scale.
    """

    plane_coef: TensorType[3, "num_components", "resolution", "resolution"]
    line_coef: TensorType[3, "num_components", "resolution", 1]

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
        smoothstep: bool = False,
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.smoothstep = smoothstep

        self.plane_coef = nn.Parameter(init_scale * torch.randn((3 * resolution * resolution, num_components)))
        self.line_coef = nn.Parameter(init_scale * torch.randn((3 * resolution, num_components)))

        self.n_output_dims = self.get_out_dim()

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def index_fn(self, x: torch.Tensor, y: torch.Tensor, width: int, height: int):
        y.clamp_max_(height - 1)
        x.clamp_max_(width - 1)

        if y.max() >= height or x.max() >= width:
            breakpoint()

        index = y * width + x
        feature_offset = width * height * torch.arange(3)
        index += feature_offset.to(x.device)[:, None, None]

        return index.long()

    def grid_sample_2d(self, feature, coord, type="plane"):
        if type == "plane":
            height, width = self.resolution, self.resolution
        else:
            height, width = self.resolution, 1

        scaled = coord * torch.tensor([width, height]).to(coord.device)[None, None]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)    # [3, num_pts, 2]

        offset = scaled - scaled_f

        # smooth version of offset
        if self.smoothstep:
            offset = offset * offset * (3.0 - 2.0 * offset)

        offset = offset[..., None, :]   # [3, num_pts, 1, 2]

        index_0 = self.index_fn(scaled_c[..., 0:1], scaled_c[..., 1:2], height, width)  # [..., num_levels]
        index_2 = self.index_fn(scaled_f[..., 0:1], scaled_f[..., 1:2], height, width)
        if type == "plane":
            index_1 = self.index_fn(scaled_c[..., 0:1], scaled_f[..., 1:2], height, width)
            index_3 = self.index_fn(scaled_f[..., 0:1], scaled_c[..., 1:2], height, width)

        # breakpoint()
        if type == "plane":
            f_0 = feature[index_0]  # [..., num_levels, features_per_level]
            f_1 = feature[index_1]
            f_2 = feature[index_2]
            f_3 = feature[index_3]

            f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
            f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])

            f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])

            return f0312
        else:
            f_0 = feature[index_0]  # [..., num_levels, features_per_level]
            f_2 = feature[index_2]
            f_02 = f_0 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
            return f_02

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]]).detach()  # [3,...,2]
        # line_coord = torch.stack([in_tensor[..., 2], in_tensor[..., 1], in_tensor[..., 0]])  # [3, ...]
        # line_coord = torch.stack([line_coord, torch.zeros_like(line_coord)], dim=-1)  # [3, ...., 2]

        # Stop gradients from going to sampler
        # plane_coord = plane_coord.view(3, -1, 1, 2).detach()
        # line_coord = line_coord.view(3, -1, 1, 2).detach()

        # plane_features = F.grid_sample(self.plane_coef, plane_coord, align_corners=True)  # [3, Components, -1, 1]
        # line_features = F.grid_sample(self.line_coef, line_coord, align_corners=True)  # [3, Components, -1, 1]

        # diff grid_sample

        plane_features = self.grid_sample_2d(self.plane_coef, plane_coord, type="plane")  # [3, -1, 1, Components]
        # line_features = self.grid_sample_2d(self.line_coef, line_coord, type="line")  # [3, -1, 1, Components]

        # features = plane_features * line_features  # [3, -1, 1, components]
        features = plane_features
        features = torch.moveaxis(features, 0, 1).reshape(-1, 3 * self.num_components)

        # features = torch.moveaxis(features.view(3 * self.num_components, *in_tensor.shape[:-1]), 0, -1)

        return features  # [..., 3 * Components]

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underyling feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )
        line_coef = F.interpolate(self.line_coef.data, size=(resolution, 1), mode="bilinear", align_corners=True)

        # TODO(ethan): are these torch.nn.Parameters needed?
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.resolution = resolution

###########
class MultiTensorVMEncoding(Encoding):
    """Multi-resolution tri-plane encoding. ----- By Yifan

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        features_per_level: Number of features per level.
    """

    def __init__(
        self,
        num_levels: int = 5,
        min_res: int = 128,
        features_per_level: int = 6,
        grows_multi: int = 2
    ) -> None:
        super().__init__(in_dim=3)  
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.res_list = [int(np.round(min_res * grows_multi**i)) for i in range(num_levels)]
        self.triplanes = []
        # for res in self.res_list:
            # self.__setattr__(f'res_{res}', TensorVMEncoding(resolution=res, num_components=features_per_level))
        self.encodings = nn.ModuleList([TensorVMEncoding(resolution=res, num_components=features_per_level) for res in self.res_list])

        self.n_output_dims = self.get_out_dim()
    
    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level * 3

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        # out_tensor = []
        # for res in self.res_list:
        #     out = self.__getattr__(f'res_{res}')(in_tensor)
        #     out_tensor.append(out)
        # out_tensor = torch.concat(out_tensor, dim=-1).to(in_tensor.device)
        out_tensor = torch.cat([encoding(in_tensor) for encoding in self.encodings], dim=-1).to(in_tensor.device)
        return out_tensor        
    
# from diffusers.models.vae import Decoder
# from diffusers.models.modeling_utils import ModelMixin
# from diffusers.models.resnet import ResnetBlock2D
# from diffusers.models.attention_processor import Attention
# from diffusers.utils import is_xformers_available

# def constant_init(module, val, bias=0):
#     if hasattr(module, 'weight') and module.weight is not None:
#         nn.init.constant_(module.weight, val)
#     if hasattr(module, 'bias') and module.bias is not None:
#         nn.init.constant_(module.bias, bias)

# class VAEDecoder(Decoder, ModelMixin):
#     def __init__(
#             self,
#             in_channels=12,
#             out_channels=24,
#             up_block_types=('UpDecoderBlock2D',),
#             block_out_channels=(64,),
#             layers_per_block=2,
#             norm_num_groups=32,
#             act_fn='silu',
#             norm_type='group',
#             zero_init_residual=True):
#         super(VAEDecoder, self).__init__(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             up_block_types=up_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=norm_num_groups,
#             act_fn=act_fn,
#             norm_type=norm_type)
#         if is_xformers_available():
#             self.enable_xformers_memory_efficient_attention()
#         self.zero_init_residual = zero_init_residual
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 # kaiming_init(m)
#                 torch.nn.init.kaiming_normal_(m.weight.data)
#             elif isinstance(m, nn.GroupNorm):
#                 constant_init(m, 1)

#         if self.zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, ResnetBlock2D):
#                     constant_init(m.conv2, 0)
#                 elif isinstance(m, Attention):
#                     constant_init(m.to_out[0], 0)

# class Tensorial2D(nn.Module):
#     noise: torch.Tensor

#     def __init__(self, noise_ch, out_ch, noise_res) -> None:
#         super().__init__()
#         self.noise_ch, self.out_ch, self.noise_res = noise_ch, out_ch, noise_res
#         self.upx = 16
#         self.register_buffer("noise", torch.randn(1, noise_ch, noise_res, noise_res))
#         self.net = VAEDecoder(
#             in_channels=noise_ch,
#             out_channels=out_ch,
#             up_block_types=('UpDecoderBlock2D',) * 5,
#             block_out_channels=(32, 64, 64, 128, 256),
#             layers_per_block=1
#         )

#     def get_output_shape(self):
#         return [self.out_ch, self.noise.size(-2) * self.upx, self.noise.size(-1) * self.upx]

#     def forward(self):
#         return self.net(self.noise)

def gaussian_filter(image, sigma=1.):
    '''
    image: [1, num_channels, H, W]
    kernel_size: [2*3*sigma+1, 2*3*sigma+1]
    '''
    kernel = create_gaussian_kernel(sigma, image.shape[1]).to(image.device)
    image = F.conv2d(image, kernel, padding=int(sigma//2))
    return image

def create_gaussian_kernel(sigma, num_channels):
    size = int(2 * 3 * sigma + 1)       # 3 sigma
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-3*sigma)**2+(y-3*sigma)**2)/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    kernel = torch.from_numpy(kernel).float()
    kernel = kernel.view(1, 1, size, size)
    kernel = kernel.repeat(num_channels, num_channels, 1, 1)
    return kernel

## copied from NeRFStudio
class TriplaneEncoding(Encoding):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product", "concat"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )        
        if False:
            if resolution == 1024:      # kernel_size = 3
                print('Smoothing triplane of resolution 1024 by 3x3 Gaussian kernel')
                self.gaussian_kernel = nn.Parameter(create_gaussian_kernel(0.333, self.num_components), requires_grad=False)
                self.padding = int(0.333//2)
            elif resolution == 2048:           # kernel_size = 5
                print('Smoothing triplane of resolution 2048 by 5x5 Gaussian kernel')
                self.gaussian_kernel = nn.Parameter(create_gaussian_kernel(1.2, self.num_components), requires_grad=False)
                self.padding = int(1.2//2)
            else:
                self.gaussian_kernel = None
        else:
            self.gaussian_kernel = None
            
        # ###### downsampler
        # max_downsample_times = 4
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(self.num_components, self.num_components, kernel_size=2**(max_downsample_times-i), stride=2**(max_downsample_times-i), padding=i//2) 
        #     for i in range(0, max_downsample_times)
        # ])
        # self.current_plane = self.plane_coef

    def get_out_dim(self) -> int:
        return self.num_components

    '''
    This impletement allows second order derivative of grid_sampler, from https://github.com/pytorch/pytorch/issues/34704
    '''
    def grid_sample(self, image, optical):
        
        N, C, IH, IW = image.shape
        _, H, W, _ = optical.shape

        ix = optical[..., 0]
        iy = optical[..., 1]

        ix = ((ix + 1) / 2) * (IW-1);
        iy = ((iy + 1) / 2) * (IH-1);
        with torch.no_grad():
            ix_nw = torch.floor(ix);
            iy_nw = torch.floor(iy);
            ix_ne = ix_nw + 1;
            iy_ne = iy_nw;
            ix_sw = ix_nw;
            iy_sw = iy_nw + 1;
            ix_se = ix_nw + 1;
            iy_se = iy_nw + 1;

        nw = (ix_se - ix)    * (iy_se - iy)
        ne = (ix    - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix)    * (iy    - iy_ne)
        se = (ix    - ix_nw) * (iy    - iy_nw)
        
        with torch.no_grad():
            torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
            torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

            torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
            torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
    
            torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
            torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
    
            torch.clamp(ix_se, 0, IW-1, out=ix_se)
            torch.clamp(iy_se, 0, IH-1, out=iy_se)

        image = image.view(N, C, IH * IW)

        nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
        ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
        sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
        se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

        out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
                ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
                sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
                se_val.view(N, C, H, W) * se.view(N, 1, H, W))

        return out_val


    def forward(self, in_tensor):
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # # Stop gradients from going to sampler
        # plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        # plane_features = F.grid_sample(
        #     self.plane_coef, plane_coord, align_corners=True
        # )  # [3, num_components, flattened_bs, 1]

        ## use new grid_sample
        plane_coord = plane_coord.view(3, -1, 1, 2)
        if self.gaussian_kernel is not None:
            plane_features = self.grid_sample(
                F.conv2d(self.plane_coef, self.gaussian_kernel, padding=self.padding), plane_coord
            )  # [3, num_components, flattened_bs, 1]
        else:
            plane_features = self.grid_sample(
                self.plane_coef, plane_coord
            )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        elif self.reduce == "sum":
            plane_features = plane_features.sum(0).squeeze(-1).T
        elif self.reduce == "concat":
            return plane_features.reshape(3*self.num_components, *original_shape[:-1]).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)


    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        if False:
            if resolution == 1024:      # kernel_size = 3
                print('Smoothing triplane of resolution 1024 by 3x3 Gaussian kernel')
                self.gaussian_kernel = nn.Parameter(create_gaussian_kernel(0.333, self.num_components), requires_grad=False)
                self.padding = int(0.333//2)
            elif resolution == 2048:           # kernel_size = 5
                print('Smoothing triplane of resolution 2048 by 5x5 Gaussian kernel')
                self.gaussian_kernel = nn.Parameter(create_gaussian_kernel(1.2, self.num_components), requires_grad=False)
                self.padding = int(1.2//2)
            else:
                self.gaussian_kernel = None

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution

    # def downsample_grid(self, level: int):
    #     self.current_plane = torch.nn.Parameter(self.convs[level](self.plane_coef))
    #     print('Current plane resolution:', self.current_plane.shape)

class MultiTriplane(Encoding):
    """Multi-resolution tri-plane encoding. ----- By Yifan

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        features_per_level: Number of features per level.
    """

    def __init__(
        self,
        num_levels: int = 5,
        min_res: int = 128,
        features_per_level: int = 6,
        grows_multi: int = 2
    ) -> None:
        super().__init__(in_dim=3)  
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.res_list = [int(np.round(min_res * grows_multi**i)) for i in range(num_levels)]
        self.triplanes = []
        # for res in self.res_list:
            # self.__setattr__(f'res_{res}', TensorVMEncoding(resolution=res, num_components=features_per_level))
        self.encodings = nn.ModuleList([TriplaneEncoding(resolution=res, num_components=features_per_level, reduce='concat') for res in self.res_list])
        # self.encodings = nn.ModuleList([TriplaneEncoding(resolution=res, num_components=features_per_level, reduce='sum') for res in self.res_list])

        self.n_output_dims = self.get_out_dim()
    
    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level * 3
        # return self.num_levels * self.features_per_level

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        # out_tensor = []
        # for res in self.res_list:
        #     out = self.__getattr__(f'res_{res}')(in_tensor)
        #     out_tensor.append(out)
        # out_tensor = torch.concat(out_tensor, dim=-1).to(in_tensor.device)
        out_tensor = torch.cat([encoding(in_tensor) for encoding in self.encodings], dim=-1).to(in_tensor.device)
        return out_tensor        
    
    def upsample_grid(self, scale_factor: float):
        new_res_list = []
        for idx, res in enumerate(self.res_list):
            new_res = int(res * scale_factor)
            self.encodings[idx].upsample_grid(new_res)
            new_res_list.append(new_res)
        self.res_list = new_res_list
        print('New multi-triplane resolution:', new_res_list)   

    # def downsample_grid(self, level: int):
    #     for encoding in self.encodings:
    #         encoding.downsample_grid(level)

##############
class Triplane(Encoding):

    def __init__(
        self,
        resolution: int = 128,
        num_components: int = 24,
        init_scale: float = 0.1,
        smoothstep: bool = False,
        enable_deep_prior: bool = False
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.smoothstep = smoothstep
        self.enable_deep_prior = enable_deep_prior

        if enable_deep_prior:
            noise_channels = 8
            noise_resolution = 20
            self.noise = nn.Parameter(init_scale * torch.randn((1, noise_channels, noise_resolution, noise_resolution)), requires_grad=True)
            # self.generator = nn.Sequential(
            #     nn.Conv2d(8, 128, 3, padding=1),  
            #     nn.Upsample(scale_factor=2, mode='nearest'),  # (1, 16, 64, 64)
            #     nn.ReLU(),
            #     nn.Conv2d(128, 128, 3, padding=1),  
            #     nn.Upsample(scale_factor=2, mode='nearest'),  # (1, 32, 128, 128)
            #     nn.ReLU(),
            #     nn.Conv2d(128, 128, 3, padding=1),  
            #     nn.Upsample(scale_factor=2, mode='nearest'),  # (1, 64, 256, 256)
            #     nn.ReLU(),
            #     nn.Conv2d(128, 128, 3, padding=1),  
            #     nn.Upsample(scale_factor=2, mode='nearest'),  # (1, 128, 512, 512) 
            #     nn.ReLU(),
            #     nn.Conv2d(128, 128, 3, padding=1),  
            #     nn.Upsample(scale_factor=2, mode='nearest'),  # (1, 256, 1024, 1024)
            #     nn.ReLU(),
            #     nn.Conv2d(128, 3*self.num_components, 3, padding=1),  # (1, 3xnum_components, 1024, 1024)
            # )
            tensor_config = ['xy', 'yz', 'zx']
            self.generator = nn.ModuleList([
                # Tensorial2D(noise_channels, self.num_components, noise_resolution) for sub in tensor_config
            ])
        else:
            self.plane_coef = nn.Parameter(init_scale * torch.randn((3 * resolution * resolution, num_components)))

        self.n_output_dims = self.get_out_dim()

    def get_out_dim(self) -> int:
        return self.num_components * 3

    def index_fn(self, x: torch.Tensor, y: torch.Tensor, width: int, height: int):
        y.clamp_max_(height - 1)
        x.clamp_max_(width - 1)

        if y.max() >= height or x.max() >= width:
            breakpoint()

        index = y * width + x
        feature_offset = width * height * torch.arange(3)
        index += feature_offset.to(x.device)[:, None, None]

        return index.long()

    def grid_sample_2d(self, feature, coord, type="plane"):
        # height, width = self.resolution, self.resolution
        height = width = int((feature.shape[0]//3)**(0.5))

        scaled = coord * torch.tensor([width, height]).to(coord.device)[None, None]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        # smooth version of offset
        if self.smoothstep:
            offset = offset * offset * (3.0 - 2.0 * offset)

        offset = offset[..., None, :]

        index_0 = self.index_fn(scaled_c[..., 0:1], scaled_c[..., 1:2], height, width)  # [..., num_levels]
        index_2 = self.index_fn(scaled_f[..., 0:1], scaled_f[..., 1:2], height, width)
        index_1 = self.index_fn(scaled_c[..., 0:1], scaled_f[..., 1:2], height, width)
        index_3 = self.index_fn(scaled_f[..., 0:1], scaled_c[..., 1:2], height, width)

        f_0 = feature[index_0]  # [..., num_levels, features_per_level]
        f_1 = feature[index_1]
        f_2 = feature[index_2]
        f_3 = feature[index_3]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])

        return f0312

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Compute encoding for each position in in_positions

        Args:
            in_tensor: position inside bounds in range [-1,1],

        Returns: Encoded position
        """
        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]])  # [3,...,2]

        if self.enable_deep_prior:
            r = []
            for sub in self.generator:
                sub_out = sub()
                expected = list(sub.get_output_shape())
                assert list(sub_out.shape[1:]) == expected, [sub_out.shape[1:], expected]
                r.append(sub_out.reshape(expected[0], expected[1]*expected[2]).permute(1, 0))
            self.plane_coef = torch.cat(r, 0)

        plane_features = self.grid_sample_2d(self.plane_coef, plane_coord, type="plane")  # [3, -1, 1, Components]

        features = plane_features
        features = torch.moveaxis(features, 0, 1).reshape(-1, 3 * self.num_components)

        return features  # [..., 3 * Components]

class HybridEncoding(Encoding):
    """
    Hybrid representation of encoding. A low resolution multi-triplane with a high resolution sparse hash grid. --Yifan
    """
    def __init__(self, multi_triplane_dict: dict, hash_dict: dict):
        super().__init__(in_dim=3)

        self.mult_triplane = MultiTriplane(**multi_triplane_dict)
        self.hash = tcnn.Encoding(**hash_dict)

        self.n_output_dims = self.mult_triplane.get_out_dim() + hash_dict['encoding_config']['n_levels'] * hash_dict['encoding_config']['n_features_per_level']

    def get_out_dim(self) -> int:
        return self.n_output_dims
    
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        mt = self.mult_triplane(in_tensor)
        hash = self.hash(in_tensor)
        return torch.concat([mt, hash], dim=1)


class SHEncoding(Encoding):
    """Spherical harmonic encoding

    Args:
        levels: Number of spherical hamonic levels to encode.
    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__(in_dim=3)

        if levels <= 0 or levels > 4:
            raise ValueError(f"Spherical harmonic encoding only suports 1 to 4 levels, requested {levels}")

        self.levels = levels

    def get_out_dim(self) -> int:
        return self.levels**2

    @torch.no_grad()
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return components_from_spherical_harmonics(levels=self.levels, directions=in_tensor)


# '''
# Tri-MipRF encoding: https://github.com/wbhu/Tri-MipRF/blob/main/neural_field/encoding/tri_mip.py
# '''
# import nvdiffrast.torch
# from nerfstudio.cameras.rays import RayBundle, RaySamples
# import math

# class TriMipEncoding(Encoding):
#     def __init__(
#         self,
#         n_levels: int = 8,
#         plane_size: int = 512,
#         feature_dim: int = 16,
#         include_xyz: bool = False,
#         max_radius: float = 0.0
#     ):
#         super().__init__(in_dim=3)
#         self.n_levels = n_levels
#         self.plane_size = plane_size
#         self.feature_dim = feature_dim
#         self.include_xyz = include_xyz
#         self.log2_feat_grid_resolution = math.log2(plane_size)
#         self.max_radius = max_radius

#         self.register_parameter(
#             "fm",
#             nn.Parameter(torch.zeros(3, plane_size, plane_size, feature_dim)),
#         )
#         self.init_parameters()
#         self.n_output_dims = (
#             self.feature_dim * 3 + 3 if include_xyz else self.feature_dim * 3
#         )

#     def init_parameters(self) -> None:
#         # Important for performance
#         nn.init.uniform_(self.fm, -1e-2, 1e-2)

#     def get_out_dim(self) -> int:
#         return self.n_output_dims
    
#     def compute_ball_radius(self, distances, radius, cos):
#         inverse_cos = 1.0 / cos
#         tmp = (inverse_cos * inverse_cos - 1).sqrt() - radius
#         tmp = tmp[:, None, :]
#         sample_ball_radius = distances * radius[:, None, :] * cos[:, None, :] / (tmp * tmp + 1.0).sqrt()
#         return sample_ball_radius

#     def get_levels(self, ray_samples: RaySamples, ray_cos):
#         frustums = ray_samples.frustums
#         # positions = frustums.get_positions()
#         # Make sure the tcnn gets inputs between 0 and 1.

#         distances = (frustums.starts + frustums.ends) / 2
#         cone_radius = torch.sqrt(frustums.pixel_area[:, 0, :]) / 1.7724538509055159
#         sample_ball_radius = self.compute_ball_radius(distances, cone_radius, ray_cos)
#         # levels = torch.log2(sample_ball_radius / self.max_radius)
#         levels = torch.log2(sample_ball_radius / 0.003)     ## change it to large scale scene
#         levels += self.log2_feat_grid_resolution
#         return levels.reshape(-1, 1)

#     def forward(self, x, level):
#         # x in [0,1], level in [0,max_level]
#         # x is Nx3, level is Nx1
#         if 0 == x.shape[0]:
#             return torch.zeros([x.shape[0], self.feature_dim * 3]).to(x)
#         decomposed_x = torch.stack(
#             [
#                 x[:, None, [1, 2]],
#                 x[:, None, [0, 2]],
#                 x[:, None, [0, 1]],
#             ],
#             dim=0,
#         )  # 3xNx1x2
#         if 0 == self.n_levels:
#             level = None
#         else:
#             # assert level.shape[0] > 0, [level.shape, x.shape]
#             torch.stack([level, level, level], dim=0)
#             level = torch.broadcast_to(
#                 level, decomposed_x.shape[:3]
#             ).contiguous()
#         enc = nvdiffrast.torch.texture(
#             self.fm,
#             decomposed_x,
#             mip_level_bias=level,
#             boundary_mode="clamp",
#             max_mip_level=self.n_levels - 1,
#         )  # 3xNx1xC
#         enc = (
#             enc.permute(1, 2, 0, 3)
#             .contiguous()
#             .view(
#                 x.shape[0],
#                 self.feature_dim * 3,
#             )
#         )  # Nx(3C)
#         if self.include_xyz:
#             enc = torch.cat([x, enc], dim=-1)
#         return enc


class PeriodicVolumeEncoding(Encoding):
    """Periodic Volume encoding

    Args:
        num_levels: Number of feature grids.
        min_res: Resolution of smallest feature grid.
        max_res: Resolution of largest feature grid.
        log2_hashmap_size: Size of hash map is 2^log2_hashmap_size.
        features_per_level: Number of features per level.
        hash_init_scale: Value to initialize hash grid.
        implementation: Implementation of hash encoding. Fallback to torch if tcnn not available.
    """

    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 1024,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 0.001,
        smoothstep: bool = False,
    ) -> None:

        super().__init__(in_dim=3)
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        assert log2_hashmap_size % 3 == 0
        self.hash_table_size = 2**log2_hashmap_size
        self.n_output_dims = num_levels * features_per_level
        self.smoothstep = smoothstep

        levels = torch.arange(num_levels)
        growth_factor = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.scalings = torch.floor(min_res * growth_factor**levels)

        self.periodic_volume_resolution = 2 ** (log2_hashmap_size // 3)
        # self.periodic_resolution = torch.minimum(torch.floor(self.scalings), periodic_volume_resolution)

        self.hash_offset = levels * self.hash_table_size
        self.hash_table = torch.rand(size=(self.hash_table_size * num_levels, features_per_level)) * 2 - 1
        self.hash_table *= hash_init_scale
        self.hash_table = nn.Parameter(self.hash_table)

        # TODO weight loss by level?
        self.per_level_weights = 1.0

    def get_out_dim(self) -> int:
        return self.num_levels * self.features_per_level

    def hash_fn(self, in_tensor: TensorType["bs":..., "num_levels", 3]) -> TensorType["bs":..., "num_levels"]:
        """Returns hash tensor using method described in Instant-NGP

        Args:
            in_tensor: Tensor to be hashed
        """

        # round to make it perioidic
        x = in_tensor
        x %= self.periodic_volume_resolution
        # xyz to index
        x = (
            x[..., 0] * (self.periodic_volume_resolution**2)
            + x[..., 1] * (self.periodic_volume_resolution)
            + x[..., 2]
        )
        # offset by feature levels
        x += self.hash_offset.to(x.device)

        return x.long()

    def pytorch_fwd(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Forward pass using pytorch. Significantly slower than TCNN implementation."""

        assert in_tensor.shape[-1] == 3
        in_tensor = in_tensor[..., None, :]  # [..., 1, 3]
        scaled = in_tensor * self.scalings.view(-1, 1).to(in_tensor.device)  # [..., L, 3]
        scaled_c = torch.ceil(scaled).type(torch.int32)
        scaled_f = torch.floor(scaled).type(torch.int32)

        offset = scaled - scaled_f

        if self.smoothstep:
            offset = offset * offset * (3.0 - 2.0 * offset)

        hashed_0 = self.hash_fn(scaled_c)  # [..., num_levels]
        hashed_1 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_2 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_f[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_3 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_c[..., 2:3]], dim=-1))
        hashed_4 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_5 = self.hash_fn(torch.cat([scaled_c[..., 0:1], scaled_f[..., 1:2], scaled_f[..., 2:3]], dim=-1))
        hashed_6 = self.hash_fn(scaled_f)
        hashed_7 = self.hash_fn(torch.cat([scaled_f[..., 0:1], scaled_c[..., 1:2], scaled_f[..., 2:3]], dim=-1))

        f_0 = self.hash_table[hashed_0]  # [..., num_levels, features_per_level]
        f_1 = self.hash_table[hashed_1]
        f_2 = self.hash_table[hashed_2]
        f_3 = self.hash_table[hashed_3]
        f_4 = self.hash_table[hashed_4]
        f_5 = self.hash_table[hashed_5]
        f_6 = self.hash_table[hashed_6]
        f_7 = self.hash_table[hashed_7]

        f_03 = f_0 * offset[..., 0:1] + f_3 * (1 - offset[..., 0:1])
        f_12 = f_1 * offset[..., 0:1] + f_2 * (1 - offset[..., 0:1])
        f_56 = f_5 * offset[..., 0:1] + f_6 * (1 - offset[..., 0:1])
        f_47 = f_4 * offset[..., 0:1] + f_7 * (1 - offset[..., 0:1])

        f0312 = f_03 * offset[..., 1:2] + f_12 * (1 - offset[..., 1:2])
        f4756 = f_47 * offset[..., 1:2] + f_56 * (1 - offset[..., 1:2])

        encoded_value = f0312 * offset[..., 2:3] + f4756 * (
            1 - offset[..., 2:3]
        )  # [..., num_levels, features_per_level]

        return torch.flatten(encoded_value, start_dim=-2, end_dim=-1)  # [..., num_levels * features_per_level]

    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        return self.pytorch_fwd(in_tensor)

    def get_total_variation_loss(self):
        """Compute the total variation loss for the feature volume."""
        feature_volume = self.hash_table.reshape(
            self.num_levels,
            self.periodic_volume_resolution,
            self.periodic_volume_resolution,
            self.periodic_volume_resolution,
            self.features_per_level,
        )
        diffx = feature_volume[:, 1:, :, :, :] - feature_volume[:, :-1, :, :, :]
        diffy = feature_volume[:, :, 1:, :, :] - feature_volume[:, :, :-1, :, :]
        diffz = feature_volume[:, :, :, 1:, :] - feature_volume[:, :, :, :-1, :]

        # TODO how to sum here or should we use mask?
        resx = diffx.abs().mean(dim=(1, 2, 3, 4))
        resy = diffy.abs().mean(dim=(1, 2, 3, 4))
        resz = diffz.abs().mean(dim=(1, 2, 3, 4))

        return ((resx + resy + resz) * self.per_level_weights).mean()

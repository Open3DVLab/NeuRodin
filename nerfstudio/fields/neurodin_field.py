"""
Implementation of NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction.
Project Page: https://open3dvlab.github.io/NeuRodin/
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    SHEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
    MultiTensorVMEncoding,
    Triplane,
    MultiTriplane,
    HybridEncoding,
    # TriMipEncoding
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass

class GeometryNet(nn.Module):
    def __init__(self, dims, num_layers, skip_in, geometric_init=True, inside_outside=True, bias=0.5, weight_norm=True):
        super().__init__()
        self.num_layers = num_layers
        self.softplus = nn.Softplus(beta=100)
        self.skip_in = skip_in
        self.out_dim = dims[-1]

        for l in range(0, num_layers - 1):
            if l + 1 in skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "glin" + str(l), lin)
        
    def forward(self, inputs):
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x
    
    
class AppearanceNet(nn.Module):
    def __init__(self, dims, weight_norm=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.num_layers_color = len(dims)
        self.dims = dims
        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            # torch.nn.init.kaiming_uniform_(lin.weight.data)
            # torch.nn.init.zeros_(lin.bias.data)                   ##### comment out

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)

    def forward(self, inputs):
        h = inputs
        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        return self.sigmoid(h) 
    

class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))
        self.schedule_beta = None

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if self.schedule_beta is not None:
            if beta is None:
                beta = self.schedule_beta + self.beta_min
            else:
                beta[beta > self.schedule_beta] = self.schedule_beta

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))
        self.enable_variance_schedule = False

    def forward(self, x):
        """Returns current variance value"""
        if self.enable_scheduled_variance:
            return torch.ones([len(x), 1], device=x.device) * self.variance
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        if self.enable_variance_schedule:
            return self.variance.clip(1e-6, 1e6)
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class NeuRodinFieldConfig(FieldConfig):
    """NeuRodin Model Config"""

    _target: Type = field(default_factory=lambda: NeuRodinField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Dimension of appearance embedding"""
    bias: float = 0.5
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.3
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm", "triplane", "multi-triplane", "hybrid", "tri-mip"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    grad_taps: int = 4
    """numerical gradients taps"""
    use_random_taps: bool = False
    """numerical gradients taps"""
    random_taps: float = 0.1
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16                                                          # neuralangelo, neus-instantngp set 32. default 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19                                                 # neuralangelo set 22,          default 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2                                            # neuralangelo set 8,   default 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True                                            ##### defalut: True (doesn't matter?)
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""
    direction_encoding_type: Literal["nerf", "sh"] = "nerf"
    """direction encoding"""
    background_model: Literal["grid", "mlp", "none"] = "mlp"
    """background models, make sure to be consistent with `background_model` from model config"""
    grows_multi: int = 2
    """multi-triplane growing factor, default 2"""
    enable_deep_prior: bool = False                 #### should debug
    """Triplane's deep prior"""

    ### spatial-varying variance
    enable_spatial_varying_variance: bool = False
    spatial_varying_variance_scale: float = 0.1

    use_laplace_density: bool = False
    use_estimated_sdf_for_laplace: bool = False

    use_unbias_for_laplace: bool = False
    """TUVR's modeling, scale the sdf before calculating density"""
    enable_scheduled_unbias_for_laplace: bool = False

    freq_shift: int = 0         ## positive value to increase the freqency

    use_pseudo_normal: bool = False

    input_normal: bool = True

    ### mipnerf-360 unbounding setting
    enable_unbounding: bool = False


class NeuRodinField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: NeuRodinFieldConfig

    def __init__(
        self,
        config: NeuRodinFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.aabb = aabb
        self.max_radius = (self.aabb[1, :] - self.aabb[0, :]).max() / 2
        self.outside_val = 1000. * (-1 if self.config.inside_outside else 1)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        if self.config.use_appearance_embedding:
            self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)

        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor
        self.use_position_encoding = self.config.use_position_encoding

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res 
        self.base_res = self.config.base_res 
        self.log2_hashmap_size = self.config.log2_hashmap_size 
        self.features_per_level = self.config.hash_features_per_level 
        use_hash = True
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        if self.use_grid_feature:
            if self.config.encoding_type == "hash":
                print("using hash encoding")
                # feature encoding
                self.encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid" if use_hash else "DenseGrid",
                        "n_levels": self.num_levels,
                        "n_features_per_level": self.features_per_level,
                        "log2_hashmap_size": self.log2_hashmap_size,
                        "base_resolution": self.base_res,
                        "per_level_scale": self.growth_factor,
                        "interpolation": "Smoothstep" if smoothstep else "Linear",
                    },
                )
                self.hash_encoding_mask = torch.ones(
                    self.num_levels * self.features_per_level,
                    dtype=torch.float32,
                )
                self.resolutions = []
                for lv in range(0, self.num_levels):
                    size = np.floor(self.base_res * self.growth_factor ** lv).astype(int) + 1
                    self.resolutions.append(size)
                print('Hash feature resolution:', self.resolutions)
            elif self.config.encoding_type == "periodic":
                print("using periodic encoding")
                self.encoding = PeriodicVolumeEncoding(
                    num_levels=self.num_levels,
                    min_res=self.base_res,
                    max_res=self.max_res,
                    log2_hashmap_size=self.log2_hashmap_size,  # 64 ** 3 = 2^18
                    # log2_hashmap_size=18,  # 64 ** 3 = 2^18
                    features_per_level=self.features_per_level,
                    smoothstep=smoothstep,
                )     
                self.hash_encoding_mask = torch.ones(
                    self.encoding.get_out_dim(),
                    dtype=torch.float32,
                )
            elif self.config.encoding_type == "tensorf_vm":
                print("using tensor vm")
                self.encoding = TensorVMEncoding(self.base_res, self.features_per_level, smoothstep=smoothstep)            # default 128, 32
                self.hash_encoding_mask = torch.ones(
                    self.encoding.get_out_dim(),
                    dtype=torch.float32,
                )
            elif self.config.encoding_type == "triplane":
                print("using triplane")
                self.encoding = Triplane(self.base_res, self.features_per_level, smoothstep=smoothstep, enable_deep_prior=self.config.enable_deep_prior)            # default 128, 32
                self.hash_encoding_mask = torch.ones(
                    self.encoding.get_out_dim(),
                    dtype=torch.float32,
                )
            elif self.config.encoding_type == "multi-triplane":
                print("using multi-resolution triplane")
                self.encoding = MultiTriplane(self.num_levels, self.base_res, self.features_per_level, grows_multi=self.growth_factor)       #####
                self.hash_encoding_mask = torch.ones(
                    self.encoding.get_out_dim(),
                    dtype=torch.float32,
                )
                print('Multi-triplane resolution:', self.encoding.res_list)
            elif self.config.encoding_type == "hybrid":
                print("using hybrid encoding")
                ### A low resolution multi-triplane to recover layout
                mt_dict = dict(
                    num_levels=self.num_levels,
                    min_res=self.base_res,
                    features_per_level=self.features_per_level,
                    grows_multi=self.growth_factor
                )
                ### A sparse but high resolution hash table to recover details
                hash_dict = dict(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 3,
                        "n_features_per_level": 8,
                        "log2_hashmap_size": 19,
                        "base_resolution": 512,
                        "per_level_scale": 2.0,
                        "interpolation": "Smoothstep" if smoothstep else "Linear",
                    },
                )
                self.encoding = HybridEncoding(mt_dict, hash_dict)
                self.hash_encoding_mask = torch.ones(
                    self.encoding.get_out_dim(),
                    dtype=torch.float32,
                )

                print('encoding_mask shape', self.hash_encoding_mask.shape)
            elif self.config.encoding_type == 'tri-mip':
                raise NotImplementedError

        # we concat inputs position ourselves
        if self.use_position_encoding:
            if not self.use_grid_feature:
                self.position_encoding = NeRFEncoding(
                    in_dim=3,
                    num_frequencies=self.config.position_encoding_max_degree,
                    min_freq_exp=0.0,
                    max_freq_exp=self.config.position_encoding_max_degree - 1,
                    include_input=False,
                    off_axis=self.config.off_axis,
                    shift=self.config.freq_shift
                )

            if self.config.direction_encoding_type == "nerf":
                self.direction_encoding = NeRFEncoding(
                    in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
                )
            elif self.config.direction_encoding_type == "sh":
                self.direction_encoding = SHEncoding(levels=4)

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        if self.use_grid_feature:
            in_dim = 3 + self.encoding.n_output_dims
        elif self.use_position_encoding:
            in_dim = 3 + self.position_encoding.get_out_dim()
        else:
            in_dim = 3
        if self.config.enable_spatial_varying_variance:
            dims = [in_dim] + dims + [2 + self.config.geo_feat_dim]
        else:
            dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]

        num_layers = len(dims)
        skip_in = [4]
        self.geometry_net = GeometryNet(dims, num_layers, skip_in, geometric_init=self.config.geometric_init, inside_outside=self.config.inside_outside, bias=self.config.bias, weight_norm=self.config.weight_norm)

        # laplace function for transform sdf to density from VolSDF
        if self.config.use_laplace_density:
            self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)             ##############

        # diffuse and specular tint layer
        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        # view dependent color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]

        if self.config.use_appearance_embedding:
            if self.config.use_diffuse_color:
                in_dim = (
                    +self.direction_encoding.get_out_dim()
                    + self.config.geo_feat_dim
                    + self.embedding_appearance.get_out_dim()
                )
            else:
                in_dim = (
                    3       # point
                    +self.direction_encoding.get_out_dim()
                    +3      # normal
                    + self.config.geo_feat_dim
                    + self.embedding_appearance.get_out_dim()
                )
                if not self.config.input_normal:
                    in_dim -= 3
        else:
            if self.config.use_diffuse_color:
                in_dim = (
                    +self.direction_encoding.get_out_dim()
                    + self.config.geo_feat_dim
                )
            else:
                in_dim = (
                    3       # point
                    +self.direction_encoding.get_out_dim()
                    +3      # normal
                    + self.config.geo_feat_dim
                )
                if not self.config.input_normal:
                    in_dim -= 3
        
        if self.config.use_n_dot_v:
            in_dim += 1

        dims = [in_dim] + dims + [3]

        self.appearance_net = AppearanceNet(dims, weight_norm=self.config.weight_norm)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        # self.numerical_gradients_delta = 0.0001
        self.numerical_gradients_delta = self.config.random_taps

        self.step = 0

        if self.config.enable_scheduled_unbias_for_laplace:
            self.unbias_laplace_ratio = 0.0
            
    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level:] = 0

    def get_zero_level_set(self, ray_samples: RaySamples, sdf=None, enable_grad=True, return_z_vals=False, return_mask=False):
        if sdf is None:
            sdf = self.get_sdf(ray_samples)

        if not enable_grad:
            torch.set_grad_enabled(False)

        [batch_size, n_samples] = ray_samples.frustums.shape
        rays_o = ray_samples.frustums.origins
        rays_d = ray_samples.frustums.directions
        z_vals = ray_samples.frustums.starts.reshape(batch_size, n_samples)
        mid_z_vals = ((ray_samples.frustums.starts + ray_samples.frustums.ends) * 0.5).reshape(batch_size, n_samples)

        sdf_d = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
        tmp = torch.einsum("ab,b->ab", (sign, idx))
        prev_idx = torch.argmax(tmp, 1, keepdim=True)
        next_idx = prev_idx + 1

        mask = (sdf_d.min(dim=-1)[0] < 0) & (sdf_d[:, 0] > 0)

        sdf1 = torch.gather(sdf_d, 1, prev_idx)
        sdf2 = torch.gather(sdf_d, 1, next_idx)

        z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
        z_vals2 = torch.gather(mid_z_vals, 1, next_idx)
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        max_z_val = torch.max(z_vals)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        pts_sdf0 = rays_o.mean(dim=1, keepdims=True) + rays_d.mean(dim=1, keepdims=True) * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]

        if not enable_grad:
            torch.set_grad_enabled(True)

        if return_z_vals:
            if return_mask:
                return z_vals_sdf0, pts_sdf0, mask
            return z_vals_sdf0, pts_sdf0
        
        if return_mask:
            return pts_sdf0, mask
        return pts_sdf0
    
    def forward_geonetwork(self, inputs):
        """forward the geonetwork, inputs: (num_pts, 3)"""
        if self.use_grid_feature:
            if self.config.enable_unbounding:
                positions = (inputs + 2.0) / 4.0                    # with mip-nerf360 unbounding setting
            else:
                positions = (inputs + 1.0) / 2.0                  # without mip-nerf360 unbounding setting
            if self.config.encoding_type == 'tri-mip':
                levels = self.encoding.get_levels(self.ray_samples, self.ray_bundle.metadata['ray_cos'])
                feature = self.encoding(positions, levels)
            else:
                feature = self.encoding(positions)
            # mask feature
            feature = feature * self.hash_encoding_mask.to(feature.device)

        if self.use_position_encoding and not self.use_grid_feature:
            pe = self.position_encoding(inputs)
        
        if self.use_grid_feature:
            inputs = torch.cat((inputs, feature), dim=-1)
        elif self.use_position_encoding:
            inputs = torch.cat((inputs, pe), dim=-1)
        else:
            pass

        return self.geometry_net(inputs)

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        if self.config.enable_spatial_varying_variance:
            sdf, _ = torch.split(h, [1, self.config.geo_feat_dim+1], dim=-1)
        else:
            sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False, use_numerical_gradients=False, use_random_taps=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if use_numerical_gradients:
            if self.config.grad_taps == 6:
                # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
                delta = self.numerical_gradients_delta
                if use_random_taps:
                    # delta = torch.rand((x.shape[0])).to(x)*0.1*(self.laplace_density.get_beta().detach()/0.1)**0.1
                    delta = torch.rand((x.shape[0])).to(x)*self.config.random_taps
                    self.delta = delta
                    delta[delta < 1. / (4. * self.config.max_res)] = 1. / (4. * self.config.max_res)
                    shift_x = torch.zeros_like(x)
                    shift_x[:, 0] = delta
                    shift_y = torch.zeros_like(x)
                    shift_y[:, 1] = delta
                    shift_z = torch.zeros_like(x)
                    shift_z[:, 2] = delta
                    points = torch.stack(
                        [
                            x + shift_x,
                            x - shift_x,
                            x + shift_y,
                            x - shift_y,
                            x + shift_z,
                            x - shift_z,
                        ],
                        dim=0,
                    )
                else:
                    points = torch.stack(
                        [
                            x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                            x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                            x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                            x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                            x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                            x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                        ],
                        dim=0,
                    )

                points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(6, *x.shape[:-1])
                gradients = torch.stack(
                    [
                        0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                        0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                        0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                    ],
                    dim=-1,
                )
            elif self.config.grad_taps == 4:
                eps = self.numerical_gradients_delta / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                points = torch.stack(
                    [
                        x + k1*eps,  # [3]
                        x + k2*eps,  # [3]
                        x + k3*eps,  # [3]
                        x + k4*eps  # [3]
                    ],
                    dim=0
                )

                points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(4, *x.shape[:-1])
                gradients = (k1*points_sdf[0, ..., None] + k2*points_sdf[1, ..., None] + k3*points_sdf[2, ..., None] + k4*points_sdf[3, ..., None]) / (4.0 * eps)
            else:
                raise NotImplementedError
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            if use_numerical_gradients:
                return gradients, points_sdf
            else:
                return gradients, y

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        if self.config.enable_spatial_varying_variance:
            sdf, _, geo_feature = torch.split(h, [1, 1, self.config.geo_feat_dim], dim=-1)
        else:
            sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None, var=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                if self.config.enable_spatial_varying_variance:
                    sdf, _, _ = torch.split(h, [1, 1, self.config.geo_feat_dim], dim=-1)
                else:
                    sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        if self.config.enable_spatial_varying_variance:
            ## should not be the zero level set points!
            ## TODO: change it to point-wise variance instead of ray-wise.
            inv_s = var.reshape(*ray_samples.frustums.shape, 1)
            if self.deviation_network.enable_variance_schedule:
                sche_inv_s = self.deviation_network.get_variance()
                inv_s[inv_s < sche_inv_s] = sche_inv_s
        else:
            inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(self, points, directions, gradients, geo_features, camera_indices, return_uncertainty=False):
        """compute colors"""

        # diffuse color and specular tint
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        if gradients is not None:
            normals = F.normalize(gradients, p=2, dim=-1)

        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        # appearance
        if self.training:
            if self.config.use_appearance_embedding:
                embedded_appearance = self.embedding_appearance(camera_indices)
        else:
             if self.config.use_appearance_embedding:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                    )

        if self.config.use_appearance_embedding:
            if self.config.use_diffuse_color:
                h = [
                    d,
                    geo_features.view(-1, self.config.geo_feat_dim),
                    embedded_appearance.view(-1, self.config.appearance_embedding_dim),
                ]
            else:
                if self.config.input_normal:
                    h = [
                        points,
                        d,
                        gradients,
                        geo_features.view(-1, self.config.geo_feat_dim),
                        embedded_appearance.view(-1, self.config.appearance_embedding_dim),
                    ]
                else:
                    h = [
                        points,
                        d,
                        geo_features.view(-1, self.config.geo_feat_dim),
                        embedded_appearance.view(-1, self.config.appearance_embedding_dim),
                    ]
        else:
            if self.config.use_diffuse_color:
                h = [
                    d,
                    geo_features.view(-1, self.config.geo_feat_dim),
                ]
            else:
                if self.config.input_normal:
                    h = [
                        points,
                        d,
                        gradients,
                        geo_features.view(-1, self.config.geo_feat_dim),
                    ]           
                else:
                    h = [
                        points,
                        d,
                        geo_features.view(-1, self.config.geo_feat_dim),
                    ]           

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)

        h = torch.cat(h, dim=-1)

        out = self.appearance_net(h)
        rgb = out[..., :3]

        if self.config.use_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # TODO linear to srgb?
            # Combine specular and diffuse components and tone map to sRGB.
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding

        if return_uncertainty:
            return rgb, out[..., -3], out[..., -2], out[..., -1]          

        return rgb

    def get_outputs(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        self.ray_samples = ray_samples

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        if self.config.use_laplace_density:
            inputs = ray_samples.frustums.get_start_positions()                    
        else:
            inputs = ray_samples.frustums.get_positions()                 
        inputs = inputs.view(-1, 3) 

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)
        with torch.enable_grad():
            h = self.forward_geonetwork(inputs)
            if self.config.enable_spatial_varying_variance:
                sdf, var, geo_feature = torch.split(h, [1, 1, self.config.geo_feat_dim], dim=-1)
                sdf = sdf.clone()               #######
                var = 1. / (self.sigmoid(var)*self.config.spatial_varying_variance_scale+2e-4)              ###############
                if self.config.use_laplace_density and self.laplace_density.schedule_beta is not None:
                    var[var < 1. / self.laplace_density.schedule_beta] = 1. / self.laplace_density.schedule_beta
                outputs.update({"var": var})
            else:
                sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
                sdf = sdf.clone()               #######
            if self.config.use_pseudo_normal:
                pseudo_normal = self.sigmoid(h[..., 1:4])
        if hasattr(self.ray_bundle, 'outside') and self.config.background_model != 'none' and not self.config.enable_unbounding:
            with torch.no_grad():
                sdf[self.ray_bundle.outside.expand_as(sdf.reshape(*ray_samples.frustums.directions.shape[:-1])).reshape(-1, 1)] = self.outside_val
        # if self.config.input_normal or self.is_evaluating_image:
        if self.config.use_numerical_gradients:
            gradients, sampled_sdf = self.gradient(
                inputs,
                skip_spatial_distortion=True,
                return_sdf=True,
                use_numerical_gradients=self.config.use_numerical_gradients,
                use_random_taps=self.config.use_random_taps
            )
            sampled_sdf = sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            sampled_sdf = None

        if self.config.input_normal:
            rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices, return_uncertainty=False)
        else:
            rgb = self.get_colors(inputs, directions_flat, None, geo_feature, camera_indices, return_uncertainty=False)

        if self.config.use_laplace_density:
            if self.config.use_estimated_sdf_for_laplace:
                true_cos = (ray_samples.frustums.directions.reshape(-1, 3) * gradients).sum(-1, keepdim=True)

                # anneal as NeuS
                cos_anneal_ratio = self._cos_anneal_ratio

                # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
                # the cos value "not dead" at the beginning training iterations, for better convergence.
                iter_cos = -(
                    F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
                )  # always non-positive

                # Estimate signed distances at section points
                estimated_next_sdf = sdf + iter_cos * ray_samples.deltas.view(-1, 1)  * 0.5
                estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas.view(-1, 1)  * 0.5
                density = self.laplace_density((estimated_next_sdf + estimated_prev_sdf) / 2)
            else:
                if self.config.use_unbias_for_laplace:
                    v = ray_samples.frustums.directions.reshape(-1, 3).unsqueeze(2)
                    denom = torch.bmm(gradients.unsqueeze(1), v).squeeze(-1) + 1e-8
                    if self.config.enable_spatial_varying_variance:
                        if self.config.enable_scheduled_unbias_for_laplace:
                            density = self.laplace_density(sdf/torch.abs(denom)**self.unbias_laplace_ratio, 1. / var)
                        else:
                            density = self.laplace_density(sdf/torch.abs(denom), 1. / var)
                    else:
                        if self.config.enable_scheduled_unbias_for_laplace:
                            density = self.laplace_density(sdf/torch.abs(denom)**self.unbias_laplace_ratio)
                        else:
                            density = self.laplace_density(sdf/torch.abs(denom))
                elif self.config.enable_spatial_varying_variance:
                    density = self.laplace_density(sdf, 1. / var) 
                else:
                    density = self.laplace_density(sdf)           ###########
            density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
            outputs.update({FieldHeadNames.DENSITY: density})

        outputs.update({'eik_grad': None})

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)
        
        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                "sampled_sdf": sampled_sdf,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            if self.config.enable_spatial_varying_variance:
                alphas = self.get_alpha(ray_samples, sdf, gradients, var=var)
            else:
                alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        if self.config.use_pseudo_normal:
            outputs.update({'pseudo_normal': pseudo_normal.view(*ray_samples.frustums.directions.shape[:-1], -1)})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy)
        return field_outputs

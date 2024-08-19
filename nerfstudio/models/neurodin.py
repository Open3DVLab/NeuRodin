"""
Implementation of NeuRodin: A Two-stage Framework for High-Fidelity Neural Surface Reconstruction.
Project Page: https://open3dvlab.github.io/NeuRodin/
"""

from __future__ import annotations
import torch
import numpy as np
import math

from dataclasses import dataclass, field
from typing import Dict, List, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import NeuSSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig
from nerfstudio.fields.neurodin_field import NeuRodinFieldConfig
import torch.nn.functional as F
import torch.nn as nn
from nerfstudio.utils.math import map_range_val
from nerfstudio.fields.density_fields import HashMLPDensityField, HashMLPSDFField
from nerfstudio.model_components.losses import interlevel_loss, distortion_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from typing import Dict, List, Tuple, Type
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.fields.nerfacto_field import TCNNNerfactoField
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from torch.nn import Parameter
from nerfstudio.model_components.scene_colliders import (
    AABBBoxCollider,
    NearFarCollider,
    SphereCollider,
    SphereColliderWithBackground
)

from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    MultiViewLoss,
    ScaleAndShiftInvariantLoss,
    SensorDepthLoss,
    compute_scale_and_shift,
    monosdf_normal_loss,
    S3IM,
)
from nerfstudio.field_components.spatial_distortions import SceneContraction


@dataclass
class NeuRodinModelConfig(SurfaceModelConfig):
    """NeuS Model Config"""

    _target: Type = field(default_factory=lambda: NeuRodinModel)
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    num_up_sample_steps: int = 4
    """number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 64
    """fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """use to use perturb for the sampled points"""
    sdf_field: NeuRodinFieldConfig = NeuRodinFieldConfig() 
    """neus sdf network config ---Yifan"""
    eikonal_loss_mult: float = 0.033              # default 0.1

    ### numerical gradients schedule
    enable_numerical_gradients_schedule: bool = False
    """numerical gradients, together with `sdf_field.use_numerical_gradients`"""
    ## curvature loss
    enable_curvature_loss_schedule: bool = False
    """curvature loss --Yifan"""
    curvature_loss_multi: float = 0.0
    """curvature loss weight, default=5e-4"""
    enable_progressive_hash_encoding: bool = False
    """Neuralangelo's progressive multi-resolution hash"""
    level_init: int = 4
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 5000
    """steps per level of multi-resolution hash encoding"""
    warm_up_end: int = 5000
    curvature_loss_warmup_steps: int = 5000
    """curvature loss warmup steps"""

    proj_consistent_multi: float = 0.0
    """"`Towards Better Gradient ...`'s loss"""

    #### schedule variance
    enable_variance_schedule: bool = False
    scheduled_variance_start_num_iters: int = 0
    """scheduled linear decay for SDF to density (transimittance) function. borrowed from PermutoSDF (or BakedSDF)"""
    scheduled_variance_max_num_iters: int = 250000
    """Max num iterations for scheduling variance of sigmoid function."""
    inv_std_init: float = 20
    """init value of scheduling variance of sigmoid function. 0.3 is for var = 1 / exp(10 * 0.3) ≈ 1 / 20 ≈ 0.04978"""
    inv_std_end: float = 1200
    """end value of scheduling variance of sigmoid function. 0.8 is for var = 1 / exp(10 * 0.8) ≈ 1 / 2981 ≈ 0.000335"""

    tangent_normal_consistent_multi: float = 0.0
    """curvature loss from PermutoSDF"""

    #### anneal eikonal loss
    use_anneal_eikonal_weight: bool = False
    """whether to use annealing for eikonal loss weight"""
    eikonal_anneal_max_num_iters: int = 250000
    """Max num iterations for the annealing of beta in laplacian density."""
    use_spatial_varying_eikonal_loss: bool = False
    """whether to use different weight of eikonal loss based the points norm, farway points have large weights"""
    eikonal_loss_mult_start: float = 0.01
    eikonal_loss_mult_end: float = 0.1
    eikonal_loss_mult_slop: float = 2.0

    #### proposal network
    use_proposal_network_sampler: bool = False
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_neus_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    """Arguments for the proposal density fields."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    dist_loss_weight: float = 0.0
    """proposal network regularization"""

    ### photometric_consistency loss
    photometric_consistency_loss_mult: float = 0.0
    enable_photometric_consistency_loss_schedule: bool = False
    
    ### align rendered depth and zero level set
    unbias_depth_loss_mult: float = 0.0
    unbias_loss_mask_coeff: float = 0.01    # 0.01 for scannetpp, 0.001 for tanks and temples

    ### to nerf
    sdf_as_density: bool = False

    ### proposal network normal loss
    use_sdf_proposal_network: bool = False

    spatial_variance_smooth_multi: float = 0.0
    spatial_sdf_smooth_multi: float = 0.0

    ## from `Critical Regularizations for Neural Surface Reconstruction in the Wild`. Should be smaller than 5e-3 to avoid failure optimization.
    minimal_surface_loss_multi: float = 0.0

    enable_unbias_loss_schedule: bool = False


class NeuRodinModel(SurfaceModel):
    """NeuS model

    Args:
        config: NeuS configuration to instantiate model
    """

    config: NeuRodinModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.scene_contraction_norm == "inf":
            order = float("inf")
        elif self.config.scene_contraction_norm == "l2":
            order = None
        else:
            raise ValueError("Invalid scene contraction norm")

        if self.config.sdf_field.enable_unbounding:
            self.scene_contraction = SceneContraction(order=order)
        else:
            self.scene_contraction = None

        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=self.scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
        )

        # Collider
        if self.scene_box.collider_type == "near_far":
            self.collider = NearFarCollider(near_plane=self.scene_box.near, far_plane=self.scene_box.far)
        elif self.scene_box.collider_type == "box":
            self.collider = AABBBoxCollider(self.scene_box, near_plane=self.scene_box.near)
        elif self.scene_box.collider_type == "sphere":
            self.collider = SphereColliderWithBackground(radius=self.scene_box.radius)
        else:
            raise NotImplementedError

        # command line near and far has highest priority
        if self.config.overwrite_near_far_plane:
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # background model
        if self.config.background_model == "grid":
            self.field_background = TCNNNerfactoField(
                self.scene_box.aabb,
                spatial_distortion=self.scene_contraction,
                num_images=self.num_train_data,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            )
        elif self.config.background_model == "mlp":
            position_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=9.0, include_input=True
            )
            direction_encoding = NeRFEncoding(
                in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
            )

            self.field_background = NeRFField(
                position_encoding=position_encoding,
                direction_encoding=direction_encoding,
                spatial_distortion=self.scene_contraction,
                use_appearance_embedding=self.field.config.use_appearance_embedding,
                num_images=self.num_train_data,
                appearance_embedding_dim=self.field.config.appearance_embedding_dim
            )
        else:
            # dummy background model
            self.field_background = Parameter(torch.ones(1), requires_grad=False)


        self.curvature_loss_multi_factor = 1.0

        if self.config.enable_variance_schedule:
            self.field.deviation_network.enable_variance_schedule = True

        if self.config.use_proposal_network_sampler:
            self.density_fns = []
            num_prop_nets = self.config.num_proposal_iterations
            # Build the proposal network(s)
            self.proposal_networks = torch.nn.ModuleList()
            if self.config.use_same_proposal_network:
                assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
                prop_net_args = self.config.proposal_net_args_list[0]
                network = HashMLPDensityField(
                    self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args
                )
                self.proposal_networks.append(network)
                self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
            else:
                for i in range(num_prop_nets):
                    prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                    network = HashMLPDensityField(
                        self.scene_box.aabb,
                        spatial_distortion=self.scene_contraction,
                        **prop_net_args,
                    )
                  
                    self.proposal_networks.append(network)
                self.density_fns.extend([network.density_fn for network in self.proposal_networks])

            # update proposal network every iterations
            update_schedule = lambda step: -1

            self.proposal_sampler = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                use_uniform_sampler=False,
                single_jitter=self.config.use_single_jitter,
                update_sched=update_schedule,
                use_sdf=self.config.use_sdf_proposal_network
            )
        else:
            self.sampler = NeuSSampler(
                num_samples=self.config.num_samples,
                num_samples_importance=self.config.num_samples_importance,
                num_samples_outside=self.config.num_samples_outside,
                num_upsample_steps=self.config.num_up_sample_steps,
                base_variance=self.config.base_variance,
            )

        if self.field.config.use_laplace_density:
            self.warm_up_end = self.config.warm_up_end

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # read the hash encoding parameters from field
        level_init = self.config.level_init
        # schedule the delta in numerical gradients computation
        num_levels = self.field.num_levels
        max_res = self.field.max_res
        base_res = self.field.base_res
        growth_factor = self.field.growth_factor
        steps_per_level = self.config.steps_per_level

        # anneal for cos in NeuS
        if self.field.config.use_laplace_density and self.warm_up_end > 0:

            def set_anneal(step):
                anneal = min([1.0, step / self.warm_up_end])
                self.field.set_cos_anneal_ratio(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )

        """
        Neuralangelo callbacks
        """
        init_delta = 1. / base_res
        end_delta = 1. / max_res
        
        # compute the delta based on level
        if self.config.enable_numerical_gradients_schedule:
            def set_delta(step):
                delta = 1. / (base_res * growth_factor ** ( step / steps_per_level))
                delta = max(1. / (4. * max_res), delta)
                self.field.set_numerical_gradients_delta(delta * 4.) # TODO because we divide 4 to normalize points to [0, 1]
                

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_delta,
                )
            )
        
        # schedule the current level of multi-resolution hash encoding
        if self.config.enable_progressive_hash_encoding:
            def set_mask(step):
                #TODO make this consistent with delta schedule
                level = int(step / steps_per_level) + 1
                level = max(level, level_init)
                self.field.update_mask(level)
    
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_mask,
                )
            )
        # schedule the curvature loss weight
        # linear warmup for 5000 steps to 5e-4 and then decay as delta
        if self.config.enable_curvature_loss_schedule:
            def set_curvature_loss_mult_factor(step):
                if step < self.config.curvature_loss_warmup_steps:
                    factor = step / self.config.curvature_loss_warmup_steps
                else:
                    delta = 1. / (base_res * growth_factor ** ( (step - self.config.curvature_loss_warmup_steps) / steps_per_level))
                    delta = max(1. / (max_res * 10.), delta)
                    factor = delta / init_delta

                self.curvature_loss_multi_factor = factor
            
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_curvature_loss_mult_factor,
                )
            )
        
        
        """
        Others
        """
        if self.config.use_anneal_eikonal_weight:
            # anneal the beta of volsdf before each training iterations
            K = self.config.eikonal_anneal_max_num_iters
            weight_init = self.config.eikonal_loss_mult_start
            weight_end = self.config.eikonal_loss_mult_end

            def set_eikonal_weight(step):
                # bakedsdf's beta schedule
                train_frac = np.clip(step / K, 0, 1)
                mult = weight_end / (1 + (weight_end - weight_init) / weight_init * ((1.0 - train_frac) ** 10))
                self.config.eikonal_loss_mult = mult

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_eikonal_weight,
                )
            )
        
        if self.config.use_proposal_network_sampler and self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )



        if self.config.enable_variance_schedule:
            M = self.config.scheduled_variance_max_num_iters
            var_init = self.config.inv_std_init+1e-2
            # var_init = self.config.inv_std_init
            var_end = self.config.inv_std_end

            def set_variance(step):
                # variance = map_range_val(step, self.config.scheduled_variance_start_num_iters, M, var_init, var_end, type='log', factor=1.)
                # variance = map_range_val(step, self.config.scheduled_variance_start_num_iters, M, var_init, var_end, type='exp', factor=1.)             #######
                if M == -1:         ## hard code for variance scheduler, can be adjusted.
                    if step < 200000:
                        variance = map_range_val(step, 0, 200000, 0.01, 500, type='exp', factor=4.)             #######
                    else:
                        variance = map_range_val(step, 200000, 300000, 500, 3000, type='exp', factor=1.)             #######
                else:
                    variance = map_range_val(step, self.config.scheduled_variance_start_num_iters, M, var_init, var_end, type='log', factor=1.)

                if self.field.config.use_laplace_density:
                    self.field.laplace_density.schedule_beta = 1. / variance
                else:
                    self.field.deviation_network.variance.data[...] = variance
            
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_variance,
                )
            )


        if self.config.enable_unbias_loss_schedule:
            unbias_loss_mult = self.config.unbias_depth_loss_mult
            def enbale_unbias_loss(step):
                self.config.unbias_depth_loss_mult = map_range_val(step, 0, 10000, 0.001, unbias_loss_mult, type='log', factor=1.)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,   
                    func=enbale_unbias_loss,
                )
            )

        if self.config.enable_photometric_consistency_loss_schedule:
            pass
            def disable_photometric_consistency_loss(step):
                if step > 200000:
                    self.config.photometric_consistency_loss_mult = 0.0

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,   
                    func=disable_photometric_consistency_loss,
                )
            )

        return callbacks
    
    def compute_LNCC(self, ref_gray, src_grays):
        # ref_gray: [1, batch_size, 121, 1]
        # src_grays: [nsrc, batch_size, 121, 1]
        ref_gray = ref_gray.permute(1, 0, 3, 2)  # [batch_size, 1, 1, 121]
        src_grays = src_grays.permute(1, 0, 3, 2)  # [batch_size, nsrc, 1, 121]

        ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

        bs, nsrc, nc, npatch = src_grays.shape
        patch_size = int(np.sqrt(npatch))
        ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
        src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
        ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

        ref_sq = ref_gray.pow(2)
        src_sq = src_grays.pow(2)

        filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
        padding = patch_size // 2

        ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

        u_ref = ref_sum / npatch
        u_src = src_sum / npatch

        cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
        ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
        src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

        cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, 1, npatch]
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        # ncc, _ = torch.topk(ncc, 4, dim=1, largest=False)
        ncc = torch.mean(ncc, dim=1, keepdim=True)
        mask = ((1-ncc).abs() > 0.01) & (ncc < 1.0)
        return ncc, mask

    def update_patch_size(self, h_patch_size, device):
        offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
        return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2
    
    def patch_homography(self, H, uv):
        # H: [batch_size, nsrc, 3, 3]
        # uv: [batch_size, 121, 2]
        N, Npx = uv.shape[:2]
        H = H.permute(1, 0, 2, 3)
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
        hom_uv = torch.cat((uv, ones), dim=-1)
 
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv)
        tmp = tmp.reshape(Nsrc, -1, 3)

        grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)

        return grid

    def get_photometric_consistency(self, ray_samples: RaySamples, cameras: Cameras, images):
        device = images.device
        batch_size = ray_samples.frustums.directions.shape[0]
        
        intrinsics = cameras.get_intrinsics_matrices().to(device)
        intrinsics_inv = torch.inverse(intrinsics)
        poses = cameras.camera_to_worlds.to(device)

        # convert camera to opencv format
        poses[:, :3, 1:3] *= -1

        # >> TODO: filter out the patches that are outside the boarder of image
        # >> TODO: filter out warping error points

        pts_sdf0, mask = self.field.get_zero_level_set(ray_samples, sdf=None, enable_grad=True, return_mask=True)
        gradients_sdf0 = self.field.gradient(pts_sdf0.view(-1, 3)).reshape(-1, 1, 3)
        gradients_sdf0 = gradients_sdf0 / torch.linalg.norm(gradients_sdf0, ord=2, dim=-1, keepdim=True)
        gradients_sdf0 = torch.matmul(poses[0, :3, :3].permute(1, 0)[None, ...], gradients_sdf0.permute(0, 2, 1)).permute(0, 2, 1).detach()

        project_xyz = torch.matmul(poses[0, :3, :3].permute(1, 0), pts_sdf0.permute(0, 2, 1))
        t = - torch.matmul(poses[0, :3, :3].permute(1, 0), poses[0, :3, 3, None])
        project_xyz = project_xyz + t
        pts_sdf0_ref = project_xyz
        project_xyz = torch.matmul(intrinsics[0, :3, :3], project_xyz)  # [batch_size, 3, 1]
        # depth_sdf = project_xyz[:, 2, 0] * mid_inside_sphere.squeeze(1)
        disp_sdf0 = torch.matmul(gradients_sdf0, pts_sdf0_ref)

        K_ref_inv = intrinsics_inv[0, :3, :3]
        K_src = intrinsics[1:, :3, :3]
        num_src = K_src.shape[0]
        R_ref_inv = poses[0, :3, :3]
        R_src = poses[1:, :3, :3].permute(0, 2, 1)
        C_ref = poses[0, :3, 3]
        C_src = poses[1:, :3, 3]
        R_relative = torch.matmul(R_src, R_ref_inv)
        C_relative = C_ref[None, ...] - C_src
        tmp = torch.matmul(R_src, C_relative[..., None])
        tmp = torch.matmul(tmp[None, ...].expand(batch_size, num_src, 3, 1), gradients_sdf0.expand(batch_size, num_src, 3)[..., None].permute(0, 1, 3, 2))  # [Batch_size, num_src, 3, 1]
        tmp = R_relative[None, ...].expand(batch_size, num_src, 3, 3) + tmp / (disp_sdf0[..., None] + 1e-10)
        tmp = torch.matmul(K_src[None, ...].expand(batch_size, num_src, 3, 3), tmp)
        Hom = torch.matmul(tmp, K_ref_inv[None, None, ...])

        pixels_x = project_xyz[:, 0, 0] / (project_xyz[:, 2, 0] + 1e-8)
        pixels_y = project_xyz[:, 1, 0] / (project_xyz[:, 2, 0] + 1e-8)
        pixels = torch.stack([pixels_x, pixels_y], dim=-1).float()
        patch_size = 3      # 5, larger patch will result in larger warpping error
        total_size = (patch_size * 2 + 1) ** 2
        offsets = self.update_patch_size(patch_size, gradients_sdf0.device)  # [1, 121, 2]
        pixels_patch = pixels.view(batch_size, 1, 2) + offsets.float()  # [batch_size, 121, 2]

        ref_image = images[0, :, :]
        src_images = images[1:, :, :]
        h, w = ref_image.shape

        grid = self.patch_homography(Hom, pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
        sampled_gray_val = F.grid_sample(src_images.unsqueeze(1), grid.view(num_src, -1, 1, 2), align_corners=True)
        sampled_gray_val = sampled_gray_val.view(num_src, batch_size, total_size, 1)  # [nsrc, batch_size, 121, 1]
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
        grid = pixels_patch.detach()
        ref_gray_val = F.grid_sample(ref_image[None, None, ...], grid.view(1, -1, 1, 2), align_corners=True)
        ref_gray_val = ref_gray_val.view(1, batch_size, total_size, 1)
        ncc, ncc_mask = self.compute_LNCC(ref_gray_val, sampled_gray_val)
        ncc_mask = ncc_mask.reshape(-1)
        ncc = ncc*ncc_mask

        # ncc = ncc * mid_inside_sphere

        return ncc*mask[:, None]


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = [n_p[1] for n_p in filter(lambda n_p:"encoding" not in n_p[0], self.field.named_parameters())]
        
        if self.field.config.use_grid_feature:
            param_groups["encoding"] = list(self.field.encoding.parameters())

        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)

        if self.config.use_proposal_network_sampler:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
            
        return param_groups

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        self.field.is_evaluating_image = self.is_evaluating_image
        if self.config.use_proposal_network_sampler:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        else:
            ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf, use_laplace_density=self.field.config.use_laplace_density)
        self.field.ray_bundle = ray_bundle              
        field_outputs = self.field(ray_samples, return_alphas=(not self.field.config.use_laplace_density))
        if self.config.sdf_as_density:
            weights, transmittance = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.SDF])
        else:
            if self.field.config.use_laplace_density:
                weights, transmittance = ray_samples.get_weights_and_transmittance(field_outputs[FieldHeadNames.DENSITY])
            else:
                weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
                    field_outputs[FieldHeadNames.ALPHA]
                )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }

        if self.config.use_proposal_network_sampler:
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
            samples_and_field_outputs["weights_list"] = weights_list
            samples_and_field_outputs["ray_samples_list"] = ray_samples_list

        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            if not self.field.config.enable_spatial_varying_variance:
                if self.field.config.use_laplace_density:
                    if self.config.enable_variance_schedule:
                        beta = self.field.laplace_density.schedule_beta
                    else:
                        beta = self.field.laplace_density.get_beta()
                    metrics_dict["std"] = beta
                    metrics_dict["inv_std"] = 1.0 / beta
                else:
                    metrics_dict["inv_std"] = self.field.deviation_network.get_variance().item()
                    metrics_dict["std"] = 1.0 / self.field.deviation_network.get_variance().item()
            elif self.config.enable_variance_schedule:
                if self.field.config.use_laplace_density:
                    beta = self.field.laplace_density.schedule_beta
                    metrics_dict["std"] = beta
                    metrics_dict["inv_std"] = 1.0 / beta
                else:
                    inv_s = self.field.deviation_network.get_variance() 
                    metrics_dict["std"] = 1. / inv_s
                    metrics_dict["inv_std"] = inv_s
            if self.config.enable_curvature_loss_schedule:
                metrics_dict["curvature_loss_multi"] = self.curvature_loss_multi_factor * self.config.curvature_loss_multi
                metrics_dict["curvature_loss_factor"] = self.curvature_loss_multi_factor
            if self.config.use_anneal_eikonal_weight:
                metrics_dict["eik_loss_multi"] = self.config.eikonal_loss_mult
            if self.config.enable_numerical_gradients_schedule:
                metrics_dict["numerical_gradients_delta"] = self.field.numerical_gradients_delta
            if self.config.enable_unbias_loss_schedule:
                metrics_dict["unbias_depth_loss_mult"] = self.config.unbias_depth_loss_mult
            grad_theta = outputs["eik_grad"]
            metrics_dict["eik_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean()

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"]) * self.config.rgb_loss_mult
        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            loss_dict["eikonal_loss"] = ((grad_theta.norm(2, dim=-1) - 1) ** 2).mean() * self.config.eikonal_loss_mult

            # s3im loss
            if self.config.s3im_loss_mult > 0:
                loss_dict["s3im_loss"] = self.s3im_loss(image, outputs["rgb"]) * self.config.s3im_loss_mult
            # foreground mask loss
            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                weights_sum = outputs["weights"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    F.binary_cross_entropy(weights_sum, fg_label) * self.config.fg_mask_loss_mult
                )
                loss_dict["rgb_loss"] = self.rgb_loss(image*fg_label, outputs["rgb"])

            if self.config.curvature_loss_multi > 0.0:
                delta = self.field.numerical_gradients_delta
                centered_sdf = outputs['field_outputs'][FieldHeadNames.SDF]
                sourounding_sdf = outputs['field_outputs']["sampled_sdf"]
                
                if self.field.config.grad_taps == 6:
                    sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
                    
                    # (a - b)/d - (b -c)/d = (a + c - 2b)/d
                    # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
                    curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
                elif self.field.config.grad_taps == 4:
                    curvature = (sourounding_sdf.sum(dim=-1, keepdim=True) / 2.0 - 2 * centered_sdf) / (delta * delta * 3)
                loss_dict["curvature_loss"] = torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            
            if self.config.proj_consistent_multi > 0.0:
                pts = outputs['ray_samples'].frustums.get_positions() 
                with torch.enable_grad():
                    pts_moved = pts - outputs['field_outputs'][FieldHeadNames.NORMAL] * outputs['field_outputs'][FieldHeadNames.SDF]
                    gradient_moved = self.field.gradient(pts_moved.view(-1, 3))
                    gradient_moved_norm = F.normalize(gradient_moved, dim=-1)
                    consis_constraint = 1 - F.cosine_similarity(gradient_moved_norm, outputs['field_outputs'][FieldHeadNames.NORMAL].reshape(-1, 3), dim=-1)
                    weight_moved = torch.exp(-10 * torch.abs(outputs['field_outputs'][FieldHeadNames.SDF])).reshape(-1) 

                loss_dict['proj_consistent_loss'] = self.config.proj_consistent_multi * torch.mean(consis_constraint * weight_moved)

            if self.config.tangent_normal_consistent_multi > 0.0:
                # epsilon=1e-4
                epsilon=1e-2
                pts = outputs['ray_samples'].frustums.get_positions().reshape(-1, 3)
                rand_directions=torch.randn_like(pts)
                rand_directions=F.normalize(rand_directions,dim=-1)
                #instead of random direction we take the normals at these points, and calculate a random vector that is orthogonal 
                tangent=torch.cross(outputs['field_outputs'][FieldHeadNames.NORMAL].reshape(-1, 3), rand_directions)
                rand_directions=tangent #set the random moving direction to be the tangent direction now
                points_shifted=pts.clone()+rand_directions*epsilon
                sdf_gradients_shifted = self.field.gradient(points_shifted)

                normals_shifted=F.normalize(sdf_gradients_shifted,dim=-1)

                dot=(outputs['field_outputs'][FieldHeadNames.NORMAL].reshape(-1, 3)*normals_shifted).sum(dim=-1, keepdim=True)
                ##the dot would assign low weight importance to normals that are almost the same, and increasing error the more they deviate. So it's something like and L2 loss. But we want a L1 loss so we get the angle, and then we map it to range [0,1]
                angle=torch.acos(torch.clamp(dot, -1.0+1e-6, 1.0-1e-6)) #goes to range 0 when the angle is the same and pi when is opposite

                curvature=angle/math.pi #map to [0,1 range]

                loss_dict['tangent_normal_consistent_loss'] = self.config.tangent_normal_consistent_multi * curvature.mean()

            if self.config.use_proposal_network_sampler:
                # eikonal loss
                grad_theta = outputs["eik_grad"]
                if self.config.use_spatial_varying_eikonal_loss:
                    ## TODO: use variance to downweight eikonal loss.
                    var = outputs['field_outputs']['var']
                    mask = torch.ones(grad_theta.shape[0], grad_theta.shape[1]).to(var.device)
                    if var.max() > 100:
                        mask[(var < 200).reshape(mask.shape)] = 0.1
                    loss_dict["eikonal_loss"] = (((grad_theta.norm(2, dim=-1) - 1) ** 2) * mask).mean() * self.config.eikonal_loss_mult
                else:
                    loss_dict["eikonal_loss"] = (
                        (grad_theta.norm(2, dim=-1) - 1) ** 2
                    ).mean() * self.config.eikonal_loss_mult

                loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )

                if self.config.dist_loss_weight > 0.0:
                    loss_dict["dist_loss"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"]) * self.config.dist_loss_weight


            if self.config.photometric_consistency_loss_mult > 0.0:
               loss_dict['photometric_consistency_loss'] = outputs['ncc'].mean() * self.config.photometric_consistency_loss_mult

            if self.config.unbias_depth_loss_mult > 0.0:
                weights = outputs['weights'].squeeze(-1)
                max_weights_idx = torch.argmax(weights, dim=-1)[:, None]
                mask = outputs['accumulation'] > 0.999

                ## let maximum weight points's back points sdf negative
                eps = 5e-4
                max_weights_pts = torch.gather(outputs['ray_samples'].frustums.get_positions(), 1, max_weights_idx[..., None].expand(-1, -1, 3))
                back_max_weights_pts = max_weights_pts + outputs['ray_samples'].frustums.directions[:, :1, :] * eps

                #####
                back_max_weights_pts_for_mask = max_weights_pts + outputs['ray_samples'].frustums.directions[:, :1, :] * self.config.unbias_loss_mask_coeff                   
                batch_pts = torch.cat([back_max_weights_pts.reshape(-1, 3), back_max_weights_pts_for_mask.reshape(-1, 3)], dim=0)
                batch_sdf = self.field.forward_geonetwork(batch_pts)[:, 0]
                back_max_weights_sdf, back_max_weights_sdf_for_mask = torch.split(batch_sdf, [batch_pts.shape[0]//2, batch_pts.shape[0]//2], dim=0)
                if self.field.config.inside_outside:
                    loss_dict["back_pts_negative_loss"] = ((outputs['field_outputs'][FieldHeadNames.SDF].min(dim=1)[0] < 0)*F.relu(back_max_weights_sdf+eps)*(back_max_weights_sdf_for_mask>0)).mean() * self.config.unbias_depth_loss_mult
                else:
                    loss_dict["back_pts_negative_loss"] = (mask*(outputs['field_outputs'][FieldHeadNames.SDF].min(dim=1)[0] < 0)*F.relu(back_max_weights_sdf+eps)*(back_max_weights_sdf_for_mask>0)).mean() * self.config.unbias_depth_loss_mult 


            if self.config.spatial_variance_smooth_multi > 0.0 and self.field.config.enable_spatial_varying_variance:
                var = outputs['field_outputs']['var']
                epsilon=1e-4
                pts = outputs['ray_samples'].frustums.get_positions().reshape(-1, 3)
                rand_directions=torch.randn_like(pts)
                rand_directions=F.normalize(rand_directions,dim=-1)
                points_shifted=pts.clone()+rand_directions*epsilon
                h_shifted = self.field.forward_geonetwork(points_shifted.reshape(-1, 3))
                var_shifted = 1 / (self.field.sigmoid(h_shifted[:, 1])*0.1)
                loss_dict["spatial_variance_smooth_loss"] = F.l1_loss(torch.log1p(var.squeeze(1)), torch.log1p(var_shifted)) * self.config.spatial_variance_smooth_multi

            if self.config.spatial_sdf_smooth_multi > 0.0:
                try:
                    sdf_shifted = h_shifted[:, 0]
                except:
                    epsilon=1e-4
                    pts = outputs['ray_samples'].frustums.get_positions().reshape(-1, 3)
                    rand_directions=torch.randn_like(pts)
                    rand_directions=F.normalize(rand_directions,dim=-1)
                    points_shifted=pts.clone()+rand_directions*epsilon
                    h_shifted = self.field.forward_geonetwork(points_shifted.reshape(-1, 3))
                    sdf_shifted = h_shifted[:, 0]

                sdf = outputs['field_outputs'][FieldHeadNames.SDF]
                loss_dict["spatial_sdf_smooth_loss"] = F.l1_loss(sdf.reshape(-1), sdf_shifted) * self.config.spatial_variance_smooth_multi

            if self.config.minimal_surface_loss_multi > 0.0:
                reg_sdf_eps = 0.5
                reg_sdf_loss = torch.mean(reg_sdf_eps / (outputs['field_outputs'][FieldHeadNames.SDF] ** 2 + reg_sdf_eps ** 2))
                loss_dict["minimal_surface_loss"] = reg_sdf_loss * self.config.minimal_surface_loss_multi

        return loss_dict


    def get_outputs(self, ray_bundle: RayBundle) -> Dict:
        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # Shotscuts
        field_outputs = samples_and_field_outputs["field_outputs"]
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.directions_norm

        # remove the rays that don't intersect with the surface
        # hit = (field_outputs[FieldHeadNames.SDF] > 0.0).any(dim=1) & (field_outputs[FieldHeadNames.SDF] < 0).any(dim=1)
        # depth[~hit] = 10000.0

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMAL], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        # TODO add a flat to control how the background model are combined with foreground sdf field
        # background model
        if self.config.background_model != "none" and "bg_transmittance" in samples_and_field_outputs:
            bg_transmittance = samples_and_field_outputs["bg_transmittance"]

            # sample inversely from far to 1000 and points and forward the bg model
            ray_bundle.nears = ray_bundle.fars
            ray_bundle.fars = torch.ones_like(ray_bundle.fars) * self.config.far_plane_bg

            ray_samples_bg = self.sampler_bg(ray_bundle)
            # use the same background model for both density field and occupancy field
            field_outputs_bg = self.field_background(ray_samples_bg)
            weights_bg = ray_samples_bg.get_weights(field_outputs_bg[FieldHeadNames.DENSITY])

            rgb_bg = self.renderer_rgb(rgb=field_outputs_bg[FieldHeadNames.RGB], weights=weights_bg)

            # merge background color to forgound color
            rgb = rgb + bg_transmittance * rgb_bg

            # rgb = self.renderer_rgb(rgb=torch.cat([field_outputs[FieldHeadNames.RGB], field_outputs_bg[FieldHeadNames.RGB]], dim=1), weights=torch.cat([weights, weights_bg], dim=1))
        else:
            weights_bg = None
            # rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            "ray_points": self.scene_contraction(
                ray_samples.frustums.get_start_positions()
            ) if self.scene_contraction is not None else ray_samples.frustums.get_start_positions(),  # used for creating visiblity mask
            "directions_norm": ray_bundle.directions_norm,  # used to scale z_vals for free space and sdf loss
            "weights_bg": weights_bg
        }

        if self.field.config.enable_spatial_varying_variance:
            var = samples_and_field_outputs['field_outputs']['var'].reshape(*ray_samples.frustums.directions.shape[:-1], 1)
            render_var = torch.sum(weights * var, dim=-2) / (torch.sum(weights, -2) + 1e-10)
            outputs['render_var'] = render_var

        if self.training:
            if field_outputs['eik_grad'] is None:
                grad_points = field_outputs[FieldHeadNames.GRADIENT]
            else:
                grad_points = field_outputs['eik_grad']
            points_norm = field_outputs["points_norm"]
            outputs.update({"eik_grad": grad_points, "points_norm": points_norm})

            outputs.update(samples_and_field_outputs)

        # TODO how can we move it to neus_facto without out of memory
        if "weights_list" in samples_and_field_outputs:
            weights_list = samples_and_field_outputs["weights_list"]
            ray_samples_list = samples_and_field_outputs["ray_samples_list"]

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs
    
    def get_outputs_flexible(self, ray_bundle: RayBundle, additional_inputs: Dict[str, TensorType]) -> Dict:
        """run the model with additional inputs such as warping or rendering from unseen rays
        Args:
            ray_bundle: containing all the information needed to render that ray latents included
            additional_inputs: addtional inputs such as images, src_idx, src_cameras

        Returns:
            dict: information needed for compute gradients
        """
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        outputs = self.get_outputs(ray_bundle)

        ray_samples = outputs["ray_samples"]
        field_outputs = outputs["field_outputs"]

        if self.config.photometric_consistency_loss_mult > 0.0:
            ncc = self.get_photometric_consistency(ray_samples[-1024:],         ###### test hybrid sampler
                additional_inputs["src_cameras"],
                additional_inputs["src_imgs"],
                )
            outputs.update({"ncc": ncc})

        return outputs
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
Implementation of Neuralangelo model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.utils import colormaps


@dataclass
class NeuralangeloModelConfig(NeuSModelConfig):
    """Neuralangelo Model Config"""

    _target: Type = field(default_factory=lambda: NeuralangeloModel)
    #TODO move to base model config since it can be used in all models
    enable_progressive_hash_encoding: bool = True
    """whether to use progressive hash encoding"""
    enable_numerical_gradients_schedule: bool = True
    """whether to use numerical gradients delta schedule"""
    enable_curvature_loss_schedule: bool = True
    """whether to use curvature loss weight schedule"""
    curvature_loss_multi: float = 5e-4
    """curvature loss weight"""
    curvature_loss_warmup_steps: int = 5000
    """curvature loss warmup steps"""
    level_init: int = 4
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 5000
    """steps per level of multi-resolution hash encoding"""

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
    use_sdf_proposal_network: bool = False

class NeuralangeloModel(NeuSModel):
    """Neuralangelo model

    Args:
        config: Neuralangelo configuration to instantiate model
    """

    config: NeuralangeloModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.curvature_loss_multi_factor = 1.0

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
                    if i == num_prop_nets - 1 and self.config.use_sdf_proposal_network:
                          network = HashMLPSDFField(                  ##########
                            self.scene_box.aabb,
                            spatial_distortion=self.scene_contraction,
                            **prop_net_args,
                        )
                    else:
                        network = HashMLPDensityField(
                            self.scene_box.aabb,
                            spatial_distortion=self.scene_contraction,
                            **prop_net_args,
                        )
                  
                    self.proposal_networks.append(network)
                if self.config.use_sdf_proposal_network:
                    self.density_fns.extend([network.density_fn for network in self.proposal_networks[:-1]])
                    self.density_fns.append(self.proposal_networks[-1].get_density)
                else:
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
        
        
        init_delta = 1. / base_res
        end_delta = 1. / max_res
        
        # compute the delta based on level
        if self.config.enable_numerical_gradients_schedule:
            def set_delta(step):
                delta = 1. / (base_res * growth_factor ** ( step / steps_per_level))
                delta = max(1. / max_res, delta)
                self.field.set_numerical_gradients_delta(delta * 2.) # TODO because we divide 4 to normalize points to [0, 1]
                
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
                    delta = max(1. / max_res, delta)
                    factor = delta / init_delta

                self.curvature_loss_multi_factor = factor
            
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_curvature_loss_mult_factor,
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

        
        
        #TODO switch to analytic gradients after delta is small enough?
        
        return callbacks
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)

        if self.config.use_proposal_network_sampler:
            param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
            
        return param_groups
    
    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        
        if self.training:
            # training statics
            metrics_dict["activated_encoding"] = self.field.hash_encoding_mask.mean().item()
            metrics_dict["numerical_gradients_delta"] = self.field.numerical_gradients_delta
            metrics_dict["curvature_loss_multi"] = self.curvature_loss_multi_factor * self.config.curvature_loss_multi
            
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # curvature loss
        if self.training and self.config.curvature_loss_multi > 0.0:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs['field_outputs'][FieldHeadNames.SDF]
            sourounding_sdf = outputs['field_outputs']["sampled_sdf"]
            
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            
            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
        
        if self.training and self.config.use_proposal_network_sampler:
                # eikonal loss
                grad_theta = outputs["eik_grad"]
                loss_dict["eikonal_loss"] = (
                    (grad_theta.norm(2, dim=-1) - 1) ** 2
                ).mean() * self.config.eikonal_loss_mult
                # if self.field.config.sample_eikonal_points and self.field.config.sample_eikonal_type == 'proj':
                #     loss_dict['proj_loss'] = outputs['field_outputs']['proj'].mean() * self.config.eikonal_loss_mult * self.curvature_loss_multi_factor    #############

                loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )

                if self.config.dist_loss_weight > 0.0:
                    loss_dict["dist_loss"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"]) * self.config.dist_loss_weight

        return loss_dict

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        if self.config.use_proposal_network_sampler:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        else:
            ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)

        # save_points("a.ply", ray_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())
        field_outputs = self.field(ray_samples, return_alphas=True)
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
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict

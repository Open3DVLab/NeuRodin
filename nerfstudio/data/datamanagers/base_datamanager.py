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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import tyro
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.heritage_dataparser import HeritageDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nerfstudio.data.dataparsers.monosdf_dataparser import MonoSDFDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import GeneralizedDataset, InputDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.misc import IterableWrapper

from nerfstudio.data.dataparsers.scannet_dataparser import ScannetParserConfig          ######### by Yifan
from nerfstudio.data.dataparsers.dtu_dataparser import DTUParserConfig              ######### by Yifan
from nerfstudio.data.dataparsers.tnt_dataparser import TNTParserConfig              ######### by Yifan
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig    ######### by Yifan
from nerfstudio.data.dataparsers.tnt_advance_dataparser import TNTAdvanceParserConfig    ######### by Yifan

from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
import numpy as np
import os
import json
from rich.progress import Console, track

CONSOLE = Console(width=120)

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[  # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "nerfstudio-data": NerfstudioDataParserConfig(),
            "mipnerf360-data": Mipnerf360DataParserConfig(),
            "blender-data": BlenderDataParserConfig(),
            "friends-data": FriendsDataParserConfig(),
            "instant-ngp-data": InstantNGPDataParserConfig(),
            # "nuscenes-data": NuScenesDataParserConfig(),
            "record3d-data": Record3DDataParserConfig(),
            "dnerf-data": DNeRFDataParserConfig(),
            "phototourism-data": PhototourismDataParserConfig(),
            "monosdf-data": MonoSDFDataParserConfig(),
            "sdfstudio-data": SDFStudioDataParserConfig(),
            "heritage-data": HeritageDataParserConfig(),
            "scannet-data": ScannetParserConfig(),                          ########## by Yifan
            "dtu-data": DTUParserConfig(),                                  ########## by Yifan
            "tnt-data": TNTParserConfig(),                                  ########## by Yifan
            "colmap-data": ColmapDataParserConfig(),                        ########## by Yifan
            "tnt-advance-data": TNTAdvanceParserConfig(),                        ########## by Yifan
        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]
"""Union over possible dataparser types, annotated with metadata for tyro. This is the
same as the vanilla union, but results in shorter subcommand names."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:
        1. A Raybundle: This will contain the rays we are sampling, with latents and
            conditionals attached (everything needed at inference)
        2. A "batch" of auxilury information: This will contain the mask, the ground truth
            pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def iter_train(self):
        """The __iter__ function for the train iterator.

        This only exists to assist the get_train_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.train_count = 0

    def iter_eval(self):
        """The __iter__ function for the eval iterator.

        This only exists to assist the get_eval_iterable function, since we need to pass
        in an __iter__ function for our trivial iterable that we are making."""
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_train and next_train functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_train_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        """Gets a trivial pythonic iterator that will use the iter_eval and next_eval functions
        as __iter__ and __next__ methods respectivley.

        This basically is just a little utility if you want to do something like:
        |    for ray_bundle, batch in datamanager.get_eval_iterable():
        |        <eval code here>
        since the returned IterableWrapper is just an iterator with the __iter__ and __next__
        methods (methods bound to our DataManager instance in this case) specified in the constructor.
        """
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    eval_camera_res_scale_factor: float = 1.0
    num_images_to_sample_pixel_from: int = -1
    """for align monocular depth or warping faster. by Yifan"""             ##############
    downsample_factor: int = 1
    """for downsampling pixels. by Yifan"""             ##############
    enable_schedule_downsample_factor: bool = False
    """for scheduling downsampling pixels. by Yifan"""             ##############


class VanillaDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: VanillaDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser = self.config.dataparser.setup()

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor*self.config.eval_camera_res_scale_factor,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self, dataset: InputDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.train_camera_optimizer,  # should be shared between train and eval.
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        # batch = self.train_pixel_sampler.sample(image_batch)
        batch = self.train_pixel_sampler.sample(image_batch, self.config.num_images_to_sample_pixel_from)           ###########
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            if isinstance(batch["image"], BasicImages):  # If this is a generalized dataset, we need to get image tensor
                batch["image"] = batch["image"].images[0]
                camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch
        raise ValueError("No more eval images")

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@dataclass
class FlexibleDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: FlexibleDataManager)
    """Target class to instantiate."""
    train_num_images_to_sample_from: int = 1
    """Number of images to sample during training iteration."""
    num_images_to_sample_pixel_from: int = -1
    """for align monocular depth or warping faster. by Yifan"""             ##############
    downsample_factor: int = 1
    """for downsampling pixels. by Yifan"""             ##############
    enable_schedule_downsample_factor: bool = False
    """for scheduling downsampling pixels. by Yifan"""             ##############
    include_pair_images: bool = False               ########### for warping
    train_num_rays_per_batch_for_pairs: int = 1024 
    neighbors_num: int = 8
    neighbors_sample_num: int = 4
    neighbors_shuffle: bool = True
    enable_schedule_pixel_sampler: bool = False

class FlexibleDataManager(VanillaDataManager):
    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs
        )
        if self.config.include_pair_images:
            self.train_pixel_sampler_for_pairs = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch_for_pairs)
            if hasattr(self.dataparser.config, 'scene_name'):
                pair_path = self.dataparser.config.data / self.dataparser.config.scene_name / f'{self.dataparser.config.scene_name}_pairs.json'
            else:
                pair_path = self.dataparser.config.data / f'pairs.json'
            if os.path.exists(str(pair_path)):
                with open(str(pair_path), 'r') as f:
                    nearest_id = json.load(f)
                self.nearest_id = torch.tensor(nearest_id)
            else:
                nearest_id = []
                for cur_cam in track(
                    self.train_dataset._dataparser_outputs.cameras, transient=True, description="Calculating pairs for cross-view constraints"
                ):
                    distances = []
                    angles = []
                    cur_poses = cur_cam.camera_to_worlds
                    for id_n, cam in enumerate(self.train_dataset._dataparser_outputs.cameras):
                        poses = cam.camera_to_worlds
                        dis = torch.norm(cur_poses[:3, 3] - poses[:3, 3])
                        angle = torch.arccos(
                            ((torch.inverse(poses[:3, :3]) @ cur_poses[:3, :3] @ torch.tensor([0, 0, 1.])) * torch.tensor(
                                [0, 0, 1.])).sum())
                        distances.append(dis)
                        angles.append(angle)
                    distances = np.array(distances)
                    angles = np.array(angles)
                    sorted_indices = np.lexsort((angles, distances))
                    nearest_id.append(list(sorted_indices[:self.config.neighbors_num+1]))

                with open(str(pair_path), 'w') as f:
                    json.dump(nearest_id, f, ensure_ascii=False, default=default_dump)
                self.nearest_id = torch.tensor(nearest_id)

    def get_nearby_images(self, image_batch, batch, neighbors_num=10, neighbors_sample_num=4, neighbors_shuffle=True):
        device = image_batch['image_idx'].device
        all_imgs = image_batch['image']
        indices = torch.tensor(list(set(batch['indices'][:, 0].tolist())))
        src_idx = self.nearest_id.index_select(0, indices)
        if neighbors_shuffle:
            perm_idx = torch.randperm(neighbors_num) + 1
            src_idx = torch.cat([src_idx[:, 0:1], src_idx[:, perm_idx[:neighbors_sample_num]]], dim=-1)
        selected_imgs = torch.index_select(all_imgs, 0, src_idx.reshape(-1)).view(*src_idx.shape, *all_imgs.shape[1:])
        selected_imgs = 0.2989 * selected_imgs[...,0] + 0.5870 * selected_imgs[...,1] + 0.1140 * selected_imgs[...,2]
        image_batch['src_imgs'] = selected_imgs.to(device)
        image_batch['src_idxs'] = src_idx.to(device)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch, num_images_to_sample_pixel_from=self.config.num_images_to_sample_pixel_from, downsample_factor=self.config.downsample_factor)
        if self.config.include_pair_images:
            # assert self.config.num_images_to_sample_pixel_from == 1
            # self.get_nearby_images(image_batch, batch, neighbors_num=self.config.neighbors_num, neighbors_shuffle=self.config.neighbors_shuffle, neighbors_sample_num=self.config.neighbors_sample_num)                 
            #### test hybrid sampler
            batch_for_pairs = self.train_pixel_sampler_for_pairs.sample(image_batch, num_images_to_sample_pixel_from=1, downsample_factor=self.config.downsample_factor)
            self.get_nearby_images(image_batch, batch_for_pairs, neighbors_num=self.config.neighbors_num, neighbors_shuffle=self.config.neighbors_shuffle, neighbors_sample_num=self.config.neighbors_sample_num) 
            batch['image'] = torch.cat([batch['image'], batch_for_pairs['image']], dim=0)
            batch['indices'] = torch.cat([batch['indices'], batch_for_pairs['indices']], dim=0)
            ####
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        additional_output = {}
        if "src_imgs" in image_batch.keys():
            ray_indices = ray_indices.to(image_batch["src_idxs"].device)
            # assert (ray_indices[:, 0] == image_batch["image_idx"]).all()
            additional_output["uv"] = ray_indices[:, 1:]
            additional_output["src_idxs"] = image_batch["src_idxs"][0]
            additional_output["src_imgs"] = image_batch["src_imgs"][0]
            additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
                image_batch["src_idxs"][0].cpu()
            ]
        return ray_bundle, batch, additional_output


    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        callbacks = []

        if self.config.enable_schedule_downsample_factor:
            def set_downsample_factor(step):
                if step < 10000:
                    self.config.downsample_factor = 8
                elif step < 40000:
                    self.config.downsample_factor = 4
                elif step < 120000:
                    self.config.downsample_factor = 2
                else:
                    self.config.downsample_factor = 1

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=10000,
                    func=set_downsample_factor,
                )
            )

        if self.config.enable_schedule_pixel_sampler:
            def set_pixel_sampler(step):
                if step > 200000:
                    if self.config.include_pair_images:
                        self.config.include_pair_images = False
                        # self.config.num_images_to_sample_pixel_from = -1
                        del self.train_image_dataloader.cached_collated_batch['src_imgs']
                        del self.train_image_dataloader.cached_collated_batch['src_idxs']
                        # self.train_pixel_sampler.num_rays_per_batch *= 4
                
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_pixel_sampler,
                )
            )

        return callbacks


# class NearbyDataManager(VanillaDataManager):
#     def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
#         """Returns the next batch of data from the train dataloader."""
#         self.train_count += 1
#         ## make sure to cache all images
#         image_batch = next(self.iter_train_image_dataloader)
#         batch = self.train_pixel_sampler.sample(image_batch)
#         ray_indices = batch["indices"]
#         ray_bundle = self.train_ray_generator(ray_indices)
#         additional_output = {}
#         if "src_imgs" in image_batch.keys():
#             ray_indices = ray_indices.to(image_batch["src_idxs"].device)
#             assert (ray_indices[:, 0] == image_batch["image_idx"]).all()
#             ######
#             # indices = torch.tensor([torch.nonzero(image_batch["image_idx"] == num) for num in ray_indices[:, 0]]).squeeze()
#             # additional_output["uv"] = ray_indices[:, 1:]
#             # additional_output["src_idxs"] = image_batch["src_idxs"][indices]
#             # additional_output["src_imgs"] = image_batch["src_imgs"][indices]
#             # additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
#             #     image_batch["src_idxs"][indices]
#             # ]
#             #####
#             additional_output["uv"] = ray_indices[:, 1:]
#             additional_output["src_idxs"] = image_batch["src_idxs"][0]
#             additional_output["src_imgs"] = image_batch["src_imgs"][0]
#             additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
#                 image_batch["src_idxs"][0]
#             ]
#         return ray_bundle, batch, additional_output
    

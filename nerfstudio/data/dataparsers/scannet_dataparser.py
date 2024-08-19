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

"""Data parser for ScanNet dataset ---Yifan"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type
from typing_extensions import Literal
from skimage.transform import resize

import os
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType
import json
from skimage.transform import resize


from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.io import load_from_json
import imageio


CONSOLE = Console()

def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}

def get_colmap_depths(image_idx: int, colmap_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    colmap_depth = colmap_depths[image_idx]

    return {"colmap_depth": colmap_depth}

def get_normals(image_idx: int, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # normal
    normal = normals[image_idx]

    return {"normal": normal}


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_src_from_pairs(
    ref_idx, all_imgs, neighbors_num=10, neighbors_sample_num=4, neighbors_shuffle=True
) -> Dict[str, TensorType]:
    # src_idx[0] is ref img
    tmp_idx = torch.tensor(list(range(neighbors_num//2))) + 1
    left_idx = (ref_idx - tmp_idx) % len(all_imgs)
    right_idx = (ref_idx + tmp_idx) % len(all_imgs)
    src_idx = torch.concat([torch.tensor([ref_idx]), left_idx, right_idx], dim=0)
    # randomly sample neighbors
    if neighbors_shuffle:
        perm_idx = torch.randperm(len(src_idx) - 1) + 1
        src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_sample_num]]])
    src_idx = src_idx.to(all_imgs.device)
    return {"src_imgs": all_imgs[src_idx].cuda(), "src_idxs": src_idx}

def get_image(image_filename, alpha_color=None, downscale=2) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    if downscale is not None:
        np_image = resize(np_image, (np_image.shape[0]//downscale, np_image.shape[1]//downscale), preserve_range=True)
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


@dataclass
class ScannetParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: Scannet)
    """target class to instantiate"""
    data: Path = Path("./data/ScanNet/rgbd")
    """Directory specifying location of data."""
    scene_name: str = "scene0050_00"
    """Scene name of ScanNet dataset"""
    frame_interval: int = 10
    """Interval of frames' sampling"""
    skip_path: Path = Path("./data/scannet/invalid_list.json")
    """Path of invaild poses list"""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    mono_normal_path: Path = Path("")
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: float = 1.
    """!!How much to downscale images. If not set, images are chosen such that the max dimension is <1600px. Intrinsic will be motivate in `base_dataset.py`. We don't need this param here.!! Now it is for normal"""
    orientation_method: Literal["up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: int = 10
    neighbors_shuffle: bool = True
    pairs_sorted_ascending: bool = True
    """if src image pairs are sorted in ascending order by similarity i.e. 
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""
    use_key_frames: bool = False
    """whether to use key frames, if true, `frame_interval` doesn't work"""
    enable_partial_frames: bool = False    #### for debug
    partial_start: int = 0
    partial_end: int = 40
    partial_frames_scale: float = 0.0

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


@dataclass
class Scannet(DataParser):
    """Scannet Dataset"""

    config: ScannetParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        if self.config.use_key_frames:
            idxs = json.load(open(f'/mnt/workspace/wangyifan/code/KeyFramesSelector/{self.config.scene_name}_keyframes.json', 'r'))
            idxs = [int(i) for i in idxs]
        else:
            idxs = [int(idx.split('.')[0]) for idx in os.listdir(self.config.data / self.config.scene_name / 'color')]
            idxs.sort()
            idxs = idxs[::self.config.frame_interval]

        indices = list(range(len(idxs)))

        skip_list = json.load(open(self.config.skip_path, 'r'))

        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        colmap_depth_images = []
        foreground_mask_images = []
        sfm_points = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []

        intrinsic = torch.tensor(np.loadtxt(self.config.data / self.config.scene_name / 'intrinsic' / 'intrinsic_color.txt').reshape([4, 4])).float()

        valid_idxs = []

        for i, idx in enumerate(idxs):
            if [self.config.scene_name, idx] in skip_list:
                print(f'Skipping frame {idx} from {self.config.scene_name}')
                indices.remove(indices[-1])
                continue
            valid_idxs.append(idx)
            image_filename = self.config.data / self.config.scene_name / 'color' / f'{idx}.jpg'

            camtoworld = torch.tensor(np.loadtxt(self.config.data / self.config.scene_name / 'pose' / f'{idx}.txt')).float()

            if self.config.include_mono_prior:
                # raise NotImplementedError
                # assert meta["has_mono_prior"]
                # # load mono depth
                # depth = np.load(self.config.data / frame["mono_depth_path"])
                # depth_images.append(torch.from_numpy(depth).float())

                # load mono normal (DSINE, different operation from ominidata)
                normal = - np.load(self.config.mono_normal_path / f'{idx}.npy')

                # transform normal to world coordinate system
                # normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                normal = torch.from_numpy(normal).float()

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)



            if self.config.include_sensor_depth:
                # assert meta["has_sensor_depth"]
                # # load sensor depth
                # sensor_depth = np.load(self.config.data / frame["sensor_depth_path"])
                # sensor_depth_images.append(torch.from_numpy(sensor_depth).float())
                sensor_depth = imageio.imread(self.config.data / self.config.scene_name / 'depth' / f'{idx}.png') / 1000.
                sensor_depth = resize(sensor_depth, (484, 648), preserve_range=True)
                sensor_depth_images.append(torch.from_numpy(sensor_depth).float())

            if self.config.include_foreground_mask:
                raise NotImplementedError
                # assert meta["has_foreground_mask"]
                # # load foreground mask
                # if self.config.load_dtu_highres:
                #     # filenames format is 000.png
                #     foreground_mask = np.array(
                #         Image.open(
                #             self.config.data / "mask" / frame["foreground_mask"].replace("_foreground_mask", "")[-7:]
                #         ),
                #         dtype="uint8",
                #     )
                # else:
                #     # filenames format is 000000_foreground_mask.png
                #     foreground_mask = np.array(Image.open(self.config.data / frame["foreground_mask"]), dtype="uint8")
                # foreground_mask = foreground_mask[..., :1]
                # foreground_mask_images.append(torch.from_numpy(foreground_mask).float() / 255.0)

            if self.config.include_sfm_points:
                raise NotImplementedError
                # assert meta["has_sparse_sfm_points"]
                # # load sparse sfm points
                # sfm_points_view = np.loadtxt(self.config.data / frame["sfm_sparse_points_view"])
                # sfm_points.append(torch.from_numpy(sfm_points_view).float())
            
            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsic[0, 0])
            fy.append(intrinsic[1, 1])
            cx.append(intrinsic[0, 2])
            cy.append(intrinsic[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            orientation_method = self.config.orientation_method

            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=orientation_method,
                center_poses=False,
            )

            # we should also transform normal accordingly
            if self.config.include_mono_prior:
                normal_images_aligned = []
                for normal_image in normal_images:
                    h, w, _ = normal_image.shape
                    normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                    normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                    normal_images_aligned.append(normal_image)
                normal_images = normal_images_aligned

        if self.config.center_poses:
            # print(torch.mean(camera_to_worlds[:, :3, 3], dim=0))
            cam_pos_list = camera_to_worlds[:, :3, 3]
            cam_center = (cam_pos_list.max(dim=0)[0] + cam_pos_list.min(dim=0)[0]) / 2
            camera_to_worlds[:, :3, 3] -= cam_center

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            # scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))       ## Yifan: for single object
            scale_factor /= float(torch.norm(camera_to_worlds[:, :3, 3], dim=1).max()*1.1)      ## Yifan: for indoor scene
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor
        if self.config.include_sensor_depth:
            sensor_depth_images = [img * scale_factor for img in sensor_depth_images]

        print('cam_center: ', cam_center)
        print('scale: ', scale_factor)

        # scene box from meta data
        aabb = torch.tensor([[-1., -1, -1], [1, 1, 1]], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=0.05,
            far=2.5,
            radius=1.0,
            collider_type="box",
        )

        if self.config.enable_partial_frames:
            indices = indices[self.config.partial_start:self.config.partial_end]                ##########
            print(f'#### ATTENTION ####: Cropped frames from [{self.config.partial_start}, {self.config.partial_end}]!')           ########
            if self.config.partial_frames_scale > 0.0:
                partial_cam_pos_list = camera_to_worlds[indices, :3, 3]
                partial_cam_center = (partial_cam_pos_list.max(dim=0)[0] + partial_cam_pos_list.min(dim=0)[0]) / 2
                camera_to_worlds[indices, :3, 3] -= partial_cam_center
                partial_scale_factor = 1.0
                partial_scale_factor /= float(torch.norm(camera_to_worlds[indices, :3, 3], dim=1).max()*1.1)
                partial_scale_factor *= self.config.partial_frames_scale
                camera_to_worlds[indices, :3, 3] *= partial_scale_factor
        
        height, width = 968, 1296
        cameras = Cameras(
            fx=fx[indices],
            fy=fy[indices],
            cx=cx[indices],
            cy=cy[indices],
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[indices, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        if self.config.include_mono_prior:
            additional_inputs_dict = {
                "cues": {
                    "func": get_depths_and_normals,
                    "kwargs": {
                        "depths": filter_list(depth_images, indices),
                        "normals": filter_list(normal_images, indices),
                    },
                }
            }
        else:
            additional_inputs_dict = {}

        if self.config.include_sensor_depth:
            additional_inputs_dict["sensor_depth"] = {
                "func": get_sensor_depths,
                "kwargs": {"sensor_depths": filter_list(sensor_depth_images, indices)},
            }

        if self.config.include_foreground_mask:
            raise NotImplementedError
            # additional_inputs_dict["foreground_masks"] = {
            #     "func": get_foreground_masks,
            #     "kwargs": {"fg_masks": filter_list(foreground_mask_images, indices)},
            # }

        if self.config.include_sfm_points:
            raise NotImplementedError
            # additional_inputs_dict["sfm_points"] = {
            #     "func": get_sparse_sfm_points,
            #     "kwargs": {"sfm_points": filter_list(sfm_points, indices)},
            # }

        # load pair information
        if split == "train" and self.config.load_pairs:
            # TODO: check correctness of sorting
            all_imgs = torch.stack([get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0)[
                indices
            ]

            additional_inputs_dict["pairs"] = {
                "func": get_src_from_pairs,
                "kwargs": {
                    "all_imgs": all_imgs,
                    "neighbors_num": self.config.neighbors_num,
                    "neighbors_shuffle": self.config.neighbors_shuffle,
                },
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=filter_list(image_filenames, indices),
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=filter_list(depth_images, indices),
            normals=filter_list(normal_images, indices),
        )
        return dataparser_outputs

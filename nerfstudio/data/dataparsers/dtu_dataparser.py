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

"""Data parser for DTU dataset ---Yifan"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type
from typing_extensions import Literal

import os
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType
import json
from glob import glob
import imageio
import skimage
import cv2

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

CONSOLE = Console()

@dataclass
class DTUParserConfig(DataParserConfig):
    """DTU (IDR) dataset parser config"""

    _target: Type = field(default_factory=lambda: DTU)
    """target class to instantiate"""
    data: Path = Path("/mnt/data/oss_beijing/DTU/DTU_IDR")
    """Directory specifying location of data."""
    scan_id: int = 24
    """scan id of DTU dataset"""
    include_mono_prior: bool = False
    """whether or not to load monocular depth and normal """
    include_sensor_depth: bool = False
    """whether or not to load sensor depth"""
    include_foreground_mask: bool = False
    """whether or not to load foreground mask"""
    include_sfm_points: bool = False
    """whether or not to load sfm points"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    # downscale_factor: float = 1.
    # """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px. Intrinsic will be motivate in `base_dataset.py`. We don't need this param here."""
    orientation_method: Literal["up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = False
    """Whether to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: Optional[int] = None
    neighbors_shuffle: Optional[bool] = False
    pairs_sorted_ascending: Optional[bool] = True
    """if src image pairs are sorted in ascending order by similarity i.e. 
    the last element is the most similar to the first (ref)"""
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    train_val_no_overlap: bool = False
    """remove selected / sampled validation images from training set"""
    auto_orient: bool = False
    """automatically orient the scene such that the up direction is the same as the viewer's up direction"""


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    # object_mask = alpha > 127.5
    object_mask = alpha / 255.

    return object_mask

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


@dataclass
class DTU(DataParser):
    """DTU Dataset"""

    config: DTUParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements

        instance_dir = os.path.join('../data', str(self.config.data), 'scan{0}'.format(self.config.scan_id))

        image_dir = '{0}/image'.format(instance_dir)
        image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/mask'.format(instance_dir)
        mask_paths = sorted(glob_imgs(mask_dir))

        cam_file = '{0}/cameras.npz'.format(instance_dir)
        n_images = len(image_paths)

        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

        camera_to_worlds = []
        fx = []
        fy = []
        cx = []
        cy = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            intrinsics = torch.tensor(intrinsics).float()
            camera_to_worlds.append(torch.from_numpy(pose).float())
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        if self.config.include_foreground_mask:
            foreground_mask_images = []
            for path in mask_paths:
                object_mask = load_mask(path)
                # object_mask = object_mask.reshape(-1)
                foreground_mask_images.append(torch.from_numpy(object_mask))
            foreground_mask_images = torch.stack(foreground_mask_images)


        indices = list(range(len(camera_to_worlds)))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]

        image_filenames = image_paths
        depth_images = []
        normal_images = []

        if self.config.include_mono_prior:
            raise NotImplementedError
            # assert meta["has_mono_prior"]
            # # load mono depth
            # depth = np.load(self.config.data / frame["mono_depth_path"])
            # depth_images.append(torch.from_numpy(depth).float())

            # # load mono normal
            # normal = np.load(self.config.data / frame["mono_normal_path"])

            # # transform normal to world coordinate system
            # normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
            # normal = torch.from_numpy(normal).float()

            # rot = camtoworld[:3, :3]

            # normal_map = normal.reshape(3, -1)
            # normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

            # normal_map = rot @ normal_map
            # normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
            # normal_images.append(normal_map)

        if self.config.include_sensor_depth:
            raise NotImplementedError
            # assert meta["has_sensor_depth"]
            # # load sensor depth
            # sensor_depth = np.load(self.config.data / frame["sensor_depth_path"])
            # sensor_depth_images.append(torch.from_numpy(sensor_depth).float())

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
            # normal_images_aligned = []
            # for normal_image in normal_images:
            #     h, w, _ = normal_image.shape
            #     normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
            #     normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
            #     normal_images_aligned.append(normal_image)
            # normal_images = normal_images_aligned

        if self.config.center_poses:
            # print(torch.mean(camera_to_worlds[:, :3, 3], dim=0))
            cam_pos_list = camera_to_worlds[:, :3, 3]
            cam_center = (cam_pos_list.max(dim=0)[0] + cam_pos_list.min(dim=0)[0]) / 2
            camera_to_worlds[:, :3, 3] -= cam_center

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))       ## Yifan: for single object
            # scale_factor /= float(torch.norm(camera_to_worlds[:, :3, 3], dim=1).max()*1.1)      ## Yifan: for indoor scene
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        # scene box from meta data
        aabb = torch.tensor([[-1., -1, -1], [1, 1, 1]], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=0.05,
            far=2.5,
            radius=1.0,
            collider_type="sphere",
        )

        height, width = 1200, 1600
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
            raise NotImplementedError
            # additional_inputs_dict = {
            #     "cues": {
            #         "func": get_depths_and_normals,
            #         "kwargs": {
            #             "depths": filter_list(depth_images, indices),
            #             "normals": filter_list(normal_images, indices),
            #         },
            #     }
            # }
        else:
            additional_inputs_dict = {}

        if self.config.include_sensor_depth:
            raise NotImplementedError
            # additional_inputs_dict["sensor_depth"] = {
            #     "func": get_sensor_depths,
            #     "kwargs": {"sensor_depths": filter_list(sensor_depth_images, indices)},
            # }

        if self.config.include_foreground_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": filter_list(foreground_mask_images, indices)},
            }

        if self.config.include_sfm_points:
            raise NotImplementedError
            # additional_inputs_dict["sfm_points"] = {
            #     "func": get_sparse_sfm_points,
            #     "kwargs": {"sfm_points": filter_list(sfm_points, indices)},
            # }
        # load pair information
        pairs_path = self.config.data / "pairs.txt"
        if pairs_path.exists() and split == "train" and self.config.load_pairs:
            raise NotImplementedError
            # with open(pairs_path, "r") as f:
            #     pairs = f.readlines()
            # split_ext = lambda x: x.split(".")[0]
            # pairs_srcs = []
            # for sources_line in pairs:
            #     sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
            #     if self.config.pairs_sorted_ascending:
            #         # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
            #         sources_array = [sources_array[0]] + sources_array[:1:-1]
            #     pairs_srcs.append(sources_array)
            # pairs_srcs = torch.tensor(pairs_srcs)
            # # TODO: check correctness of sorting
            # all_imgs = torch.stack([get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0)[
            #     indices
            # ].cuda()

            # additional_inputs_dict["pairs"] = {
            #     "func": get_src_from_pairs,
            #     "kwargs": {
            #         "all_imgs": all_imgs,
            #         "pairs_srcs": pairs_srcs,
            #         "neighbors_num": self.config.neighbors_num,
            #         "neighbors_shuffle": self.config.neighbors_shuffle,
            #     },
            # }

        dataparser_outputs = DataparserOutputs(
            image_filenames=filter_list(image_filenames, indices),
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=filter_list(depth_images, indices),
            normals=filter_list(normal_images, indices),
        )
        return dataparser_outputs

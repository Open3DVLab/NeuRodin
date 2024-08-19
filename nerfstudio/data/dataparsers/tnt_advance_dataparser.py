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

"""Data parser for tnt advance dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type
from typing_extensions import Literal

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

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
from glob import glob
import os

CONSOLE = Console()

scale_factor_dict = {
    'Ballroom': 0.8,
    'Museum': 0.3,
    'Auditorium': 0.8,
    'Courtroom': 0.7
}

@dataclass
class TNTAdvanceParserConfig(DataParserConfig):
    """TNT (Tanks And Temples) dataset parser config"""

    _target: Type = field(default_factory=lambda: TNTAdvance)
    """target class to instantiate"""
    data: Path = Path("./TanksAndTemples/advanced")
    """Directory specifying location of data."""
    scene_name: str = "Meetingroom"
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
    # TODO supports downsample
    # downscale_factor: Optional[int] = None
    # """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    orientation_method: Literal["up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = False
    """Whether to center the poses."""
    auto_scale_poses: bool = True
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
    load_dtu_highres: bool = False
    """load high resolution images from DTU dataset, should only be used for the preprocessed DTU dataset"""

    frame_interval: int = 1


def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


@dataclass
class TNTAdvance(DataParser):
    """TNT Dataset"""

    config: TNTAdvanceParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        # meta = load_from_json(self.config.data / self.config.scene_name / "transforms.json")
        # frames = meta['frames'][::self.config.frame_interval]

        # indices = list(range(len(frames)))

        # # subsample to avoid out-of-memory for validation set
        # if split != "train" and self.config.skip_every_for_val_split >= 1:
        #     indices = indices[:: self.config.skip_every_for_val_split]
        # else:
        #     # if you use this option, training set should not contain any image in validation set
        #     if self.config.train_val_no_overlap:
        #         indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]
        # print('Using frames:', indices)

        def glob_imgs(path):
            imgs = []
            for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
                imgs.extend(glob(os.path.join(path, ext)))
            return imgs

        def glob_txt(path):
            imgs = []
            for ext in ['*.txt']:
                imgs.extend(glob(os.path.join(path, ext)))
            return imgs
        
        def load_cam(file: str):
            """ read camera txt file """
            cam = np.zeros((2, 4, 4))
            with open(file) as f:
                words = f.read().split()
            # read extrinsic
            for i in range(0, 4):
                for j in range(0, 4):
                    extrinsic_index = 4 * i + j + 1
                    cam[0][i][j] = words[extrinsic_index]

            # read intrinsic
            for i in range(0, 3):
                for j in range(0, 3):
                    intrinsic_index = 3 * i + j + 18
                    cam[1][i][j] = words[intrinsic_index]

            return cam

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        sfm_points = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        image_paths = glob_imgs(self.config.data / self.config.scene_name / 'images')
        image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
        camera_paths = glob_txt(self.config.data / self.config.scene_name / 'cams')
        camera_paths = sorted(camera_paths, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[0]))

        indices = list(range(len(image_paths)))

        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]
        print('Using frames:', indices)

        for i in range(len(image_paths)):
            # intrinsic, camtoworld = get_camera(i)
            cam = load_cam(camera_paths[i])
            intrinsic = torch.from_numpy(cam[1][:3, :3]).float()
            extrinsic = cam[0]
            camtoworld = torch.from_numpy(np.linalg.inv(extrinsic)).float()
            image_filename = image_paths[i]

            if self.config.include_mono_prior:
                raise NotImplementedError

            if self.config.include_sensor_depth:
                raise NotImplementedError

            if self.config.include_foreground_mask:
                raise NotImplementedError

            if self.config.include_sfm_points:
                raise NotImplementedError

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
        self.config.scale_factor = scale_factor_dict[self.config.scene_name]
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            if self.config.scene_name == 'Meetingroom':
                scale_factor /= float(torch.norm(camera_to_worlds[:, :3, 3], dim=1).max()*1.1)      ## Yifan: for indoor scene
            else:
                scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))       ## Yifan: for single object
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor

        # scene box from meta data
        # aabb = torch.tensor([[-1., -1, -1], [1, 1, 1]], dtype=torch.float32)
        # scene_box = SceneBox(
        #     aabb=aabb,
        #     near=0.05,
        #     far=2.5,
        #     radius=1.0,
        #     # collider_type="box",
        #     collider_type="sphere",
        # )
        scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32), near=0)         #### change the near plane from 0.1 to 0.01


        height, width = 835, 1500 
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

        try:
            tmp_img = Image.open(str(image_filenames[0])[:-4]+'.JPG')
            if str(image_filenames[0])[-3:] == 'jpg':
                image_filenames = [Path(str(fname)[:-4]+'.JPG') for fname in image_filenames]
        except:
            tmp_img = Image.open(str(image_filenames[0])[:-4]+'.jpg')
            if str(image_filenames[0])[-3:] == 'JPG':
                image_filenames = [Path(str(fname)[:-4]+'.jpg') for fname in image_filenames]
        (cameras.height, cameras.width, _) = np.array(tmp_img).shape    # revise the shape, or it will be wrong due to the round operator

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

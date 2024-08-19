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

"""Data parser for friends dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type
from typing_extensions import Literal
from skimage.transform import resize

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
import cv2

CONSOLE = Console()


# def get_src_from_pairs(
#     ref_idx, all_imgs, pairs_srcs, neighbors_num=None, neighbors_shuffle=False
# ) -> Dict[str, TensorType]:
#     # src_idx[0] is ref img
#     src_idx = pairs_srcs[ref_idx]
#     # randomly sample neighbors
#     if neighbors_num and neighbors_num > -1 and neighbors_num < len(src_idx) - 1:
#         if neighbors_shuffle:
#             perm_idx = torch.randperm(len(src_idx) - 1) + 1
#             src_idx = torch.cat([src_idx[[0]], src_idx[perm_idx[:neighbors_num]]])
#         else:
#             src_idx = src_idx[: neighbors_num + 1]
#     src_idx = src_idx.to(all_imgs.device)
#     return {"src_imgs": all_imgs[src_idx], "src_idxs": src_idx}

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


# def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
#     """Returns a 3 channel image.

#     Args:
#         image_idx: The image index in the dataset.
#     """
#     pil_image = Image.open(image_filename)
#     np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
#     assert len(np_image.shape) == 3
#     assert np_image.dtype == np.uint8
#     assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
#     image = torch.from_numpy(np_image.astype("float32") / 255.0)
#     if alpha_color is not None and image.shape[-1] == 4:
#         assert image.shape[-1] == 4
#         image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
#     else:
#         image = image[:, :, :3]
#     return image

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


def get_sensor_depths(image_idx: int, sensor_depths):
    """function to process additional sensor depths

    Args:
        image_idx: specific image index to work with
        sensor_depths: semantics data
    """

    # sensor depth
    sensor_depth = sensor_depths[image_idx]

    return {"sensor_depth": sensor_depth}


def get_foreground_masks(image_idx: int, fg_masks):
    """function to process additional foreground_masks

    Args:
        image_idx: specific image index to work with
        fg_masks: foreground_masks
    """

    # sensor depth
    fg_mask = fg_masks[image_idx]

    return {"fg_mask": fg_mask}


def get_sparse_sfm_points(image_idx: int, sfm_points):
    """function to process additional sparse sfm points

    Args:
        image_idx: specific image index to work with
        sfm_points: sparse sfm points
    """

    # sfm points
    sparse_sfm_points = sfm_points[image_idx]
    sparse_sfm_points = BasicImages([sparse_sfm_points])
    return {"sparse_sfm_points": sparse_sfm_points}


@dataclass
class SDFStudioDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: SDFStudio)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
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
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    load_pairs: bool = False
    """whether to load pairs for multi-view consistency"""
    neighbors_num: int = 10
    neighbors_shuffle: bool = True
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
    include_instance_mask: bool = False
    enable_partial_frames: bool = False    #### for debug
    partial_start: int = 0
    partial_end: int = 40
    frames_interval: int = 1
    partial_frames_scale: float = 0.0
    include_test_normal: bool = False

    ablation: bool = False

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []


@dataclass
class SDFStudio(DataParser):
    """SDFStudio Dataset"""

    config: SDFStudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        meta = load_from_json(self.config.data / "meta_data.json")

        indices = list(range(len(meta["frames"])))[::self.config.frames_interval]

        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]
        else:
            # if you use this option, training set should not contain any image in validation set
            if self.config.train_val_no_overlap:
                indices = [i for i in indices if i % self.config.skip_every_for_val_split != 0]
        # print(split, indices)

        if self.config.ablation and split != "train":
            indices = [28]

        image_filenames = []
        depth_images = []
        normal_images = []
        sensor_depth_images = []
        foreground_mask_images = []
        ins_mask_images = []
        sfm_points = []
        fx = []
        fy = []
        cx = []
        cy = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            image_filename = self.config.data / frame["rgb_path"]

            intrinsics = torch.tensor(frame["intrinsics"])
            camtoworld = torch.tensor(frame["camtoworld"])

            # here is hard coded for DTU high-res images
            if self.config.load_dtu_highres:
                image_filename = self.config.data / "image" / frame["rgb_path"].replace("_rgb", "")
                intrinsics[:2, :] *= 1200 / 384.0
                intrinsics[0, 2] += 200
                height, width = 1200, 1600
                meta["height"], meta["width"] = height, width

            if self.config.include_mono_prior:
                assert meta["has_mono_prior"]
                # load mono depth
                depth = np.load(self.config.data / frame["mono_depth_path"])
                # depth = np.zeros((384, 384))            ########
                depth_images.append(torch.from_numpy(depth).float())

                # load mono normal
                normal = np.load(self.config.data / frame["mono_normal_path"])

                # transform normal to world coordinate system
                normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                normal = torch.from_numpy(normal).float()

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)

            if self.config.include_sensor_depth:
                assert meta["has_sensor_depth"]
                # load sensor depth
                sensor_depth = np.load(self.config.data / frame["sensor_depth_path"])
                sensor_depth_images.append(torch.from_numpy(sensor_depth).float())

            if self.config.include_test_normal:
                # normal = np.load(f'plots/pure-neus-0e75f3c4d9-mt5x6-mixeik-smooth-pure-neus-2024-03-25_110347/normal/{i}.npy')         ###### 21d970d8de multi-tirplane
                # normal = np.load(f'plots/pure-neus-21d970d8de-hash-eik0.0-nonormal-pure-neus-2024-03-22_173542/normal/{i}.npy')             ######### self
                # normal = np.random.randn(*normal.shape)          ######## random
                # normal = np.load(f'plots/pure-neus-0050-mt5x6-128b-mixeik-pure-neus-2024-03-24_010127/normal/{i}.npy')             ######### 0050 multi-triplane
                # normal = np.load(f'plots/pure-neus-0e75f3c4d9-mt5x6-mixeik-smooth-pure-neus-2024-03-25_110347/normal/{i}.npy')             ######### 0e75f3c4d9 multi-triplane
                normal = - np.load(self.config.data / frame["mono_normal_path"])
                # normal = cv2.resize(normal.transpose(1, 2, 0), (876, 584)).transpose(2, 0, 1)         ##########

                # transform normal to world coordinate system
                normal = torch.from_numpy(normal).float()

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)

                # normal_images.append(normal)

                # depth = np.load(f'plots/pure-neus-0e75f3c4d9-mt5x6-mixeik-smooth-pure-neus-2024-03-25_110347/depth/{i}.npy')[..., 0]       ###### 21d970d8de multi-tirplane
                # depth = np.load(f'plots/pure-neus-21d970d8de-hash-eik0.0-nonormal-pure-neus-2024-03-22_173542/depth/{i}.npy')[..., 0]           ######### self depth
                # depth = np.random.rand(*normal_map.shape[:-1])          ######## random
                # depth = np.load(f'plots/pure-neus-0050-mt5x6-128b-mixeik-pure-neus-2024-03-24_010127/depth/{i}.npy')[..., 0]           ######### 0050 multi-triplane
                # depth = np.load(f'plots/pure-neus-0e75f3c4d9-mt5x6-mixeik-smooth-pure-neus-2024-03-25_110347/depth/{i}.npy')[..., 0]           ######### 0050 multi-triplane
                depth = np.load(self.config.data / frame["mono_depth_path"])
                # depth = cv2.resize(depth, (876, 584))

                depth_images.append(torch.from_numpy(depth).float())
                # depth_images.append(torch.zeros(normal.shape[:2]).float())

            if self.config.include_foreground_mask:
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
                ins_mask_load = np.array(Image.open(f'/mnt/data/oss_beijing/wangyifan/objectsdf++/scannet/scan1/segs/{frame["rgb_path"].replace("rgb", "segs")}'))
                ins_mask = np.zeros_like(ins_mask_load)
                ins_mask[ins_mask_load==5] = 1
                foreground_mask_images.append(ins_mask[..., None])

            if self.config.include_sfm_points:
                assert meta["has_sparse_sfm_points"]
                # load sparse sfm points
                sfm_points_view = np.loadtxt(self.config.data / frame["sfm_sparse_points_view"])
                sfm_points.append(torch.from_numpy(sfm_points_view).float())

            if self.config.include_instance_mask:
                ins_mask_load = np.array(Image.open(f'/mnt/data/oss_beijing/wangyifan/objectsdf++/scannet/scan1/segs/{frame["rgb_path"].replace("rgb", "segs")}'))
                ins_mask = np.zeros_like(ins_mask_load)
                ins_mask[ins_mask_load==2] = 1
                ins_mask_images.append(ins_mask)

            # append data
            image_filenames.append(image_filename)
            fx.append(intrinsics[0, 0])
            fy.append(intrinsics[1, 1])
            cx.append(intrinsics[0, 2])
            cy.append(intrinsics[1, 2])
            camera_to_worlds.append(camtoworld)

        fx = torch.stack(fx)
        fy = torch.stack(fy)
        cx = torch.stack(cx)
        cy = torch.stack(cy)
        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            if "orientation_override" in meta:
                orientation_method = meta["orientation_override"]
                CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
            else:
                orientation_method = self.config.orientation_method

            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method=orientation_method,
                center_poses=self.config.center_poses,
            )

            # we should also transform normal accordingly
            normal_images_aligned = []
            for normal_image in normal_images:
                h, w, _ = normal_image.shape
                normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(camera_to_worlds[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        camera_to_worlds[:, :3, 3] *= scale_factor
        self.scale_factor = scale_factor

        # scene box from meta data
        meta_scene_box = meta["scene_box"]
        aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox(
            aabb=aabb,
            near=meta_scene_box["near"],
            far=meta_scene_box["far"],
            radius=meta_scene_box["radius"],
            collider_type=meta_scene_box["collider_type"],
        )

        if self.config.enable_partial_frames:
            indices = indices[self.config.partial_start:self.config.partial_end]                ##########
            print(f'#### ATTENTION ####: Test cropped frames from [{self.config.partial_start}, {self.config.partial_end}]!')           ########
            if self.config.partial_frames_scale > 0.0:
                partial_cam_pos_list = camera_to_worlds[indices, :3, 3]
                partial_cam_center = (partial_cam_pos_list.max(dim=0)[0] + partial_cam_pos_list.min(dim=0)[0]) / 2
                camera_to_worlds[indices, :3, 3] -= partial_cam_center
                partial_scale_factor = 1.0
                partial_scale_factor /= float(torch.norm(camera_to_worlds[indices, :3, 3], dim=1).max()*1.1)
                partial_scale_factor *= self.config.partial_frames_scale
                camera_to_worlds[indices, :3, 3] *= partial_scale_factor
                camera_to_worlds[indices, :3, 3] += torch.tensor([0.2, -0.2, 0.2])[None, :]

        height, width = meta["height"], meta["width"]
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

        if self.config.include_mono_prior or self.config.include_test_normal:
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
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"fg_masks": filter_list(foreground_mask_images, indices)},
            }

        if self.config.include_instance_mask:
            additional_inputs_dict["foreground_masks"] = {
                "func": get_foreground_masks,
                "kwargs": {"ins_masks": filter_list(ins_mask_images, indices)},
            }

        if self.config.include_sfm_points:
            additional_inputs_dict["sfm_points"] = {
                "func": get_sparse_sfm_points,
                "kwargs": {"sfm_points": filter_list(sfm_points, indices)},
            }
        # # load pair information
        # pairs_path = self.config.data / "pairs.txt"
        # if pairs_path.exists() and split == "train" and self.config.load_pairs:
        #     with open(pairs_path, "r") as f:
        #         pairs = f.readlines()
        #     split_ext = lambda x: x.split(".")[0]
        #     pairs_srcs = []
        #     for sources_line in pairs:
        #         sources_array = [int(split_ext(img_name)) for img_name in sources_line.split(" ")]
        #         if self.config.pairs_sorted_ascending:
        #             # invert (flip) the source elements s.t. the most similar source is in index 1 (index 0 is reference)
        #             sources_array = [sources_array[0]] + sources_array[:1:-1]
        #         pairs_srcs.append(sources_array)
        #     pairs_srcs = torch.tensor(pairs_srcs)
        #     # TODO: check correctness of sorting
        #     all_imgs = torch.stack([get_image(image_filename) for image_filename in sorted(image_filenames)], axis=0)[
        #         indices
        #     ].cuda()

            # additional_inputs_dict["pairs"] = {
            #     "func": get_src_from_pairs,
            #     "kwargs": {
            #         "all_imgs": all_imgs,
            #         "pairs_srcs": pairs_srcs,
            #         "neighbors_num": self.config.neighbors_num,
            #         "neighbors_shuffle": self.config.neighbors_shuffle,
            #     },
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

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
Implementation of patch warping for multi-view consistency loss.
"""

import numpy as np
import torch
from torch import nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples


def get_intersection_points(
    ray_samples: RaySamples, sdf: torch.Tensor, normal: torch.Tensor, in_image_mask: torch.Tensor
):
    """compute intersection points

    Args:
        ray_samples (RaySamples): _description_
        sdf (torch.Tensor): _description_
        normal (torch.Tensor): _description_
        in_image_mask (torch.Tensor): we only use the rays in the range of [half_patch:h-half_path, half_patch:w-half_path]
    Returns:
        _type_: _description_
    """
    # TODO we should support different ways to compute intersections

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    n_rays, n_samples = ray_samples.shape
    starts = ray_samples.frustums.starts
    sign_matrix = torch.cat([torch.sign(sdf[:, :-1, 0] * sdf[:, 1:, 0]), torch.ones(n_rays, 1).to(sdf.device)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_samples, 0, -1).float().to(sdf.device)

    # Get first sign change and mask for values where a.) a sign changed
    # occurred and b.) no a neg to pos sign change occurred (meaning from
    # inside surface to outside)
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_pos_to_neg = sdf[torch.arange(n_rays), indices, 0] > 0

    # Define mask where a valid depth value is found
    mask = mask_sign_change & mask_pos_to_neg & in_image_mask

    # Get depth values and function values for the interval
    d_low = starts[torch.arange(n_rays), indices, 0][mask]
    v_low = sdf[torch.arange(n_rays), indices, 0][mask]
    n_low = normal[torch.arange(n_rays), indices, :][mask]

    indices = torch.clamp(indices + 1, max=n_samples - 1)
    d_high = starts[torch.arange(n_rays), indices, 0][mask]
    v_high = sdf[torch.arange(n_rays), indices, 0][mask]
    n_high = normal[torch.arange(n_rays), indices, :][mask]

    # linear-interpolations or run secant method to refine depth
    z = (v_low * d_high - v_high * d_low) / (v_low - v_high)

    # make this simpler
    origins = ray_samples.frustums.origins[torch.arange(n_rays), indices, :][mask]
    directions = ray_samples.frustums.directions[torch.arange(n_rays), indices, :][mask]

    intersection_points = origins + directions * z[..., None]

    # interpolate normal for simplicity so we don't need to call the model again
    points_normal = (v_low[..., None] * n_high - v_high[..., None] * n_low) / (v_low[..., None] - v_high[..., None])

    points_normal = torch.nn.functional.normalize(points_normal, dim=-1, p=2)

    # filter normals that are perpendicular to view directions
    valid = (points_normal * directions).sum(dim=-1).abs() > 0.1
    intersection_points = intersection_points[valid]
    points_normal = points_normal[valid]
    new_mask = mask.clone()
    new_mask[mask] &= valid

    return intersection_points, points_normal, new_mask


def get_homography(
    intersection_points: torch.Tensor, normal: torch.Tensor, cameras: Cameras, valid_angle_thres: float = 0.3
):
    """get homography

    Args:
        intersection_points (torch.Tensor): _description_
        normal (torch.Tensor): _description_
        cameras (Cameras): _description_
    """
    device = intersection_points.device

    # construct homography
    c2w = cameras.camera_to_worlds.to(device)
    K = cameras.get_intrinsics_matrices().to(device)
    K_inv = torch.linalg.inv(K)

    # convert camera to opencv format
    c2w[:, :3, 1:3] *= -1

    w2c_r = c2w[:, :3, :3].transpose(1, 2)
    w2c_t = -w2c_r @ c2w[:, :3, 3:]
    w2c = torch.cat([w2c_r, w2c_t], dim=-1)

    R_rel = w2c[:, :3, :3] @ c2w[:1, :3, :3]  # [N, 3, 3]
    t_rel = w2c[:, :3, :3] @ c2w[:1, :3, 3:] + w2c[:1, :3, 3:]  # [N, 3, 1]

    p_ref = w2c[0, :3, :3] @ intersection_points.transpose(1, 0) + w2c[0, :3, 3:]  # [3, n_pts]
    n_ref = w2c[0, :3, :3] @ normal.transpose(1, 0)  # [3, n_pts]

    d = torch.sum(n_ref * p_ref, dim=0, keepdims=True)
    # TODO make this clear
    H = R_rel[:, None, :, :] + t_rel[:, None, :, :] @ n_ref.transpose(1, 0)[None, :, None, :] / d[..., None, None]

    H = K[:, None] @ H @ K_inv[None, :1]  # [n_cameras, n_pts, 3, 3]

    # compute valid mask for homograpy, we should filter normal that are prependicular to source viewing ray directions
    dir_src = torch.nn.functional.normalize(c2w[:, None, :, 3] - intersection_points[None], dim=-1)
    valid = (dir_src * normal[None]).sum(dim=-1) > valid_angle_thres

    # point should be in front of cameras
    p_src = w2c[:, :3, :3] @ intersection_points.transpose(1, 0) + w2c[:, :3, 3:]  # [:, 3, n_pts]
    valid_2 = p_src[:, 2, :] > 0.01

    return H, valid & valid_2


class PatchWarping(nn.Module):
    """Standard patch warping."""

    def __init__(self, patch_size: int = 31, pixel_offset: float = 0.5, valid_angle_thres: float = 0.3):
        super().__init__()

        self.patch_size = patch_size
        half_size = patch_size // 2
        self.valid_angle_thres = valid_angle_thres

        # generate pattern
        patch_coords = torch.meshgrid(
            torch.arange(-half_size, half_size + 1), torch.arange(-half_size, half_size + 1), indexing="xy"
        )

        patch_coords = torch.stack(patch_coords, dim=-1) + pixel_offset  # stored as (y, x) coordinates
        self.patch_coords = torch.cat([patch_coords, torch.zeros_like(patch_coords[..., :1])], dim=-1)

    def forward(
        self,
        ray_samples: RaySamples,
        sdf: torch.Tensor,
        normal: torch.Tensor,
        cameras: Cameras,
        images: torch.Tensor,
        pix_indices: torch.Tensor,
    ):

        device = sdf.device

        cameras = cameras.to(device)

        # filter out the patches that are outside the boarder of image
        in_image_mask = (
            (pix_indices[:, 0] > self.patch_size // 2)
            & (pix_indices[:, 1] > self.patch_size // 2)
            & (pix_indices[:, 0] < (cameras.image_height[0] - self.patch_size // 2 - 1))
            & (pix_indices[:, 1] < (cameras.image_width[0] - self.patch_size // 2 - 1))
        )
        # [n_imgs, n_rays_valid, patch_h*patch_w]

        # find intersection points and normals
        intersection_points, normal, mask = get_intersection_points(ray_samples, sdf, normal, in_image_mask)

        # Attention: we construct homography with OPENCV coordinate system
        H, H_valid_mask = get_homography(intersection_points, normal, cameras, self.valid_angle_thres)

        # Attention uv is (y, x) and we should change to (x, y) for homography
        pix_indices = torch.flip(pix_indices, dims=[-1])[mask].float()
        pix_indices = torch.cat([pix_indices, torch.ones(pix_indices.shape[0], 1).to(device)], dim=-1)  # [n_pts, 3]

        pix_indices = pix_indices[:, None, None, :] + self.patch_coords[None].to(device)  # [n_pts, patch_h, patch_w, 3]
        pix_indices = pix_indices.permute(0, 3, 1, 2).reshape(
            1, -1, 3, self.patch_size**2
        )  # [1, n_pts, 3, patch_h*patch_w]

        warped_indices = H @ pix_indices

        # patches after warping should have positive depth
        positive_depth_mask = warped_indices[:, :, 2, :] >= 0.2
        warped_indices[:, :, 2, :] *= positive_depth_mask
        warped_indices = warped_indices[:, :, :2, :] / (warped_indices[:, :, 2:, :] + 1e-6)

        pix_coords = warped_indices.permute(0, 1, 3, 2).contiguous()  # [..., :2]
        pix_coords[..., 0:1] /= cameras.image_width[:, None, None].to(device) - 1
        pix_coords[..., 1:2] /= cameras.image_height[:, None, None].to(device) - 1
        pix_coords = (pix_coords - 0.5) * 2

        # valid
        valid = (
            (pix_coords[..., 0] > -1.0)
            & (pix_coords[..., 0] < 1.0)
            & (pix_coords[..., 1] > -1.0)
            & (pix_coords[..., 1] < 1.0)
        )  # [n_imgs, n_rays_valid, patch_h*patch_w]

        # combine valid with H
        valid = valid & H_valid_mask[..., None] & positive_depth_mask

        rgb = torch.nn.functional.grid_sample(
            images.permute(0, 3, 1, 2).to(sdf.device),
            pix_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        rgb = rgb.permute(0, 2, 3, 1)  # [n_imgs, n_rays_valid, patch_h*patch_w, 3]

        # save as visualization
        if False:
            import cv2

            vis_patch_num = 60
            image = (
                rgb[:, :vis_patch_num, :, :]
                .reshape(-1, vis_patch_num, self.patch_size, self.patch_size, 3)
                .permute(1, 2, 0, 3, 4)
                .reshape(vis_patch_num * self.patch_size, -1, 3)
            )

            cv2.imwrite("vis.png", (image.detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1])
        return rgb, valid

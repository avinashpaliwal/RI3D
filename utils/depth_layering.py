# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def get_depth_bins(depth=None, disparity=None, num_bins=None, mask=None):
    """
    :param depth: [1, 1, H, W]
    :param disparity: [1, 1, H, W]
    :return: depth_bins
    """

    assert (disparity is not None) or (depth is not None)
    if disparity is None:
        depth = torch.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=1e-2)
        depth = torch.clamp(depth, min=1e-2)
        disparity = 1. / depth

    if depth is None:
        disparity = torch.nan_to_num(disparity, nan=0.1, posinf=10.0, neginf=1e-4)
        disparity = torch.clamp(disparity, min=1e-4)
        depth = 1. / disparity

    # Ensure tensors are finite
    depth = torch.nan_to_num(depth, nan=10.0, posinf=10.0, neginf=1e-2)
    depth = torch.clamp(depth, min=1e-2)
    disparity = torch.nan_to_num(disparity, nan=0.1, posinf=10.0, neginf=1e-4)
    disparity = torch.clamp(disparity, min=1e-4)

    assert depth.shape[:2] == (1, 1) and disparity.shape[:2] == (1, 1)
    disparity_max = disparity.max().item()
    disparity_min = disparity.min().item()
    disparity_feat = disparity[:, :, ::10, ::10].reshape(-1, 1).cpu().numpy()
    
    if mask is not None:
        mask_np = mask[:, :, ::10, ::10].reshape(-1, 1)
        if isinstance(mask_np, torch.Tensor):
            mask_np = mask_np.cpu().numpy()
        nonzero_idx = np.nonzero(mask_np.reshape(-1))[0]
        if len(nonzero_idx) > 2:
            disparity_feat = disparity_feat[nonzero_idx].reshape(-1, 1)
            
    disparity_feat = np.nan_to_num(disparity_feat, nan=disparity_min, posinf=disparity_max, neginf=disparity_min)
    
    denom = disparity_max - disparity_min
    if denom < 1e-6:
        d_min = float(depth.min().item())
        d_max = float(depth.max().item())
        n = num_bins if num_bins is not None else 5
        return [float(x) for x in np.linspace(d_min - 1e-6, d_max + 1e-6, n + 1)]

    disparity_feat = (disparity_feat - disparity_min) / denom
    disparity_feat = np.nan_to_num(disparity_feat, nan=0.0, posinf=1.0, neginf=0.0)

    if num_bins is None:
        n_clusters = None
        distance_threshold = 5
    else:
        n_clusters = num_bins
        distance_threshold = None

    try:
        result = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold).fit(disparity_feat)
        num_bins = result.n_clusters_ if n_clusters is None else n_clusters
        depth_bins = [depth.min().item()]
        for i in range(num_bins):
            cluster_pts = disparity_feat[result.labels_ == i]
            if len(cluster_pts) > 0:
                th = cluster_pts.min()
                th = th * denom + disparity_min
                if th > 1e-6:
                    depth_bins.append(1. / th)

        depth_bins = sorted(set(depth_bins))
        while len(depth_bins) < (num_bins + 1):
            depth_bins.append(depth.max().item() + 1e-6 * (len(depth_bins) + 1))
        depth_bins = sorted(depth_bins)
        depth_bins[0] = depth.min().item() - 1e-6
        depth_bins[-1] = depth.max().item() + 1e-6
        return depth_bins
    except Exception:
        d_min = float(depth.min().item())
        d_max = float(depth.max().item())
        n = num_bins if num_bins is not None else 5
        return [float(x) for x in np.linspace(d_min - 1e-6, d_max + 1e-6, n + 1)]

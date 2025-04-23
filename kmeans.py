"""Implements KMeans using lists, numpy, and torch."""

import sys
from collections import defaultdict
from typing import Tuple, List
import numpy as np
import torch


def create_problem(
    n_items: int, dim: int
) -> Tuple[List[List[float]], np.array, torch.Tensor]:
    """Creates n_items vectors of size dim to be clusterd.

    It creates three versions of the data:
    (1) Lists of lists of floats
    (2) A 2D numpy array
    (3) A 2D torch tensor"""
    numpy_data = np.random.rand(n_items, dim)
    torch_data = torch.Tensor(numpy_data)
    data = numpy_data.tolist()
    return data, numpy_data, torch_data


def init_clusters(n_items, k: int) -> defaultdict:
    """Initializes the clusters (core python implementation)."""
    mapping = defaultdict(list)
    for i in range(n_items):
        mapping[i % k].append(i)
    return mapping


# ----- Computing cluster centers (3 implementations) -----
def compute_cluster_centers(
    data: List[List[float]], mapping: defaultdict, k: int
) -> List[List[float]]:
    """Computes the cluster centers."""
    n_items, dim = len(data), len(data[0])
    centers = [[0] * dim for _ in range(n_items)]
    for c in range(k):
        item_idxs = mapping[c]
        for idx in item_idxs:
            for j in range(dim):
                centers[c][j] += data[idx][j] / len(item_idxs)
    return centers


def compute_cluster_centers_with_numpy(
    data: np.array, mapping: defaultdict, k: int
) -> np.array:
    """Computes the cluster centers (using numpy)."""
    _, dim = data.shape
    centers = np.zeros((k, dim))
    for c in range(k):
        item_idxs = mapping[c]
        cluster_data = data[item_idxs]
        centers[c] = np.mean(cluster_data, axis=0)
    return centers


def compute_cluster_centers_with_torch(
    data: torch.tensor, mapping: defaultdict, k: int
) -> torch.tensor:
    """Computes the cluster centers (using torch)."""
    _, dim = data.shape
    centers = torch.zeros((k, dim))
    for c in range(k):
        item_idxs = mapping[c]
        cluster_data = data[item_idxs]
        centers[c] = torch.mean(cluster_data, axis=0)
    return centers


# ----- Computing cluster centers (3 implementations) -----
def find_new_clusters(
    data: List[List[float]], centers: List[List[float]], old_mapping: defaultdict
) -> Tuple[defaultdict, float, bool]:
    """Find the new assignments."""
    n_items, dim = len(data), len(data[0])
    mapping = defaultdict(list)
    sum_dist = 0.0
    change = False
    for i_idx in range(n_items):
        # which one is closer
        best_center, best_dist = -1, sys.maxsize
        for c_idx, c in enumerate(centers):
            dist = 0.0
            for x in range(dim):
                dist += (c[x] - data[i_idx][x]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_center = c_idx
        new_c = best_center
        sum_dist += best_dist
        mapping[new_c].append(i_idx)
        if i_idx not in old_mapping[new_c]:
            change = True
    mean_dist = sum_dist / n_items
    return mapping, mean_dist, change


def find_new_clusters_with_numpy(
    data: np.array, centers: np.array, old_mapping: defaultdict
) -> Tuple[defaultdict, float, bool]:
    """Find the new assignments (using numpy)."""
    n_items, _ = data.shape
    mapping = defaultdict(list)
    sum_dist = 0.0
    change = False
    for i in range(n_items):
        # which one is closer
        distances = np.sum((data[i] - centers) ** 2, axis=1)
        new_c = np.argmin(distances)
        sum_dist += distances[new_c]
        mapping[new_c].append(i)
        if i not in old_mapping[new_c]:
            change = True
    mean_dist = sum_dist / n_items
    return mapping, mean_dist, change


def find_new_clusters_with_torch(
    data: torch.tensor, centers: torch.tensor, old_mapping: defaultdict
) -> Tuple[defaultdict, float, bool]:
    """Find the new assignments (using numpy)."""
    n_items, _ = data.shape
    mapping = defaultdict(list)
    sum_dist = 0.0
    change = False
    for i in range(n_items):
        # which one is closer
        distances = torch.sum((data[i] - centers) ** 2, axis=1)
        new_c = torch.argmin(distances).item()
        sum_dist += distances[new_c].item()
        mapping[new_c].append(i)
        if i not in old_mapping[new_c]:
            change = True
    mean_dist = sum_dist / n_items
    return mapping, mean_dist, change


def kmeans(data, mapping):
    """Run the main kmeans loop"""
    itt = 0
    change = True
    while itt < 20 and change:
        if isinstance(data, list):
            centers = compute_cluster_centers(data, mapping, 3)
            mapping, mean_dist, change = find_new_clusters(data, centers, mapping)
        elif isinstance(data, np.ndarray):
            centers = compute_cluster_centers_with_numpy(data, mapping, 3)
            mapping, mean_dist, change = find_new_clusters_with_numpy(
                data, centers, mapping
            )
        else:
            centers = compute_cluster_centers_with_torch(data, mapping, 3)
            mapping, mean_dist, change = find_new_clusters_with_torch(
                data, centers, mapping
            )
        print(f"Itt: {itt} -- average distance: {mean_dist}")
        itt += 1


def main():
    """The main function."""
    # create the problem
    data, np_data, torch_data = create_problem(100, 10)
    # initializing the kmeans loop
    mapping = init_clusters(len(data), 3)
    # the main loop
    print ('== Solving with basic python')
    kmeans(data, mapping)
    print ('== Solving with numpy')
    kmeans(np_data, mapping)
    print ('== Solving with torch')
    kmeans(torch_data, mapping)

if __name__ == "__main__":
    main()

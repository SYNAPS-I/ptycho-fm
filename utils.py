import numpy as np
import torch
import matplotlib.pyplot as plt


def computedistancematrix(patch_size: int,
                          num_patches: int,
                          length: int)->np.array:
  distance_matrix = np.zeros((num_patches, num_patches))
  for i in range(num_patches):
    for j in range(num_patches):
      if i == j: 
        continue 
      xi, yi = i // length, i % length
      xj, yj = j // length, j % length
      distance_matrix[i, j] = patch_size * np.linalg.norm([xj - xi, yj - yi])
  return distance_matrix


def computemeanattentiondistance(patch_size: int, 
                                 attention_weights: np.array)->np.array:
  attention_weights = attention_weights[..., 1:, 1:]
  num_patches = attention_weights.shape[-1]
  length = int(np.sqrt(num_patches))
  distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
  h, w = distance_matrix.shape
  mean_distances = attention_weights * distance_matrix
  mean_distances = np.sum(mean_distances, axis=-1)
  mean_distances = np.mean(mean_distances, axis=-1)
  return mean_distances


def computemads(attention_scores: torch.tensor, 
                patch_size: int)->list:
  all_mads = [computemeanattentiondistance(patch_size, attention_weight.numpy()) for attention_weight in attention_scores]
  return all_mads


def visualize_mads(all_mads: np.array, 
                   save_dir:str="./")->None:
  fpath = save_dir + "mads.pdf"
  num_heads = len(all_mads)
  plt.figure(figsize=(6, 6))
  for idx in range(len(all_mads)):
      mean_distance = all_mads[f"block_{idx}_mean_dist"]
      x = [idx] * num_heads
      y = mean_distance[0, :]
      plt.scatter(x=x, y=y, label=f"attention_head_{idx}")
    plt.xlabel("Block Index")
    plt.ylabel("Mean Attention Distance")
    plt.legend(loc="lower right")
    plt.savefig(fpath, bbox_inches='tight')



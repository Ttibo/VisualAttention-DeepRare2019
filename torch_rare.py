import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class DeepRare(nn.Module):
    """
    DeepRare2019 Class.
    """

    def __init__(self):
        """
        Constructor for the DeepRare model.
        """
        super(DeepRare, self).__init__()

    @staticmethod
    def tensor_resize(tensor, size=(240, 240)):
        """
        Resize a tensor to the specified size using bilinear interpolation.

        Args:
            tensor (torch.Tensor): Input tensor.
            size (tuple): Desired output size (height, width).

        Returns:
            torch.Tensor: Resized tensor.
        """
        resize_transform = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)
        return resize_transform(tensor.unsqueeze(0)).squeeze(0)

    @staticmethod
    def normalize_tensor(tensor, min_val=0, max_val=1):
        """
        Normalize a tensor to the specified range [min_val, max_val].

        Args:
            tensor (torch.Tensor): Input tensor.
            min_val (float): Minimum value of the normalized range.
            max_val (float): Maximum value of the normalized range.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max - tensor_min == 0:
            return torch.zeros_like(tensor)
        return ((tensor - tensor_min) / (tensor_max - tensor_min)) * (max_val - min_val) + min_val
    
    def map_ponderation(self, tensor):
        """
        Apply weighting to a tensor map based on its rarity.

        Args:
            tensor (torch.Tensor): Input tensor map.

        Returns:
            torch.Tensor: Weighted tensor map.
        """
        map_max = tensor.max()
        map_mean = tensor.mean()
        map_weight = (map_max - map_mean) ** 2
        return self.normalize_tensor(tensor, min_val=0, max_val=1) * map_weight

    def fuse_itti(self, maps):
        """
        Perform Itti-like fusion of maps.

        Args:
            maps (list[torch.Tensor]): List of input maps to fuse.

        Returns:
            torch.Tensor: Fused map.
        """
        fused_map = torch.zeros_like(maps[0])
        for feature_map in maps:
            fused_map += self.map_ponderation(feature_map)
        return fused_map

    def rarity(self, channel, bins=6):
        """
        Compute the single-resolution rarity for a given channel.

        Args:
            channel (torch.Tensor): Input channel.
            bins (int): Number of bins for histogram computation.

        Returns:
            torch.Tensor: Rarity map.
        """
        a, b = channel.shape

        # Apply border padding
        channel[:1, :] = 0
        channel[:, :1] = 0
        channel[a - 1:, :] = 0
        channel[:, b - 1:] = 0

        # Histogram computation
        channel = self.normalize_tensor(channel, min_val=0, max_val=256)
        hist = torch.histc(channel, bins=bins, min=0, max=256)
        hist = hist / hist.sum()
        hist = -torch.log(hist + 1e-4)

        # Back-projection
        hist_idx = ((channel * bins - 1).long().clamp(0, bins - 1))
        dst = self.normalize_tensor(hist[hist_idx], min_val=0, max_val=1)
        return self.map_ponderation(dst)

    def apply_rarity(self, layer_output, threshold=0.2):
        """
        Apply rarity computation to all feature maps in a layer.

        Args:
            layer_output (torch.Tensor): Feature maps of shape [B, C, H, W].
            threshold (float): Threshold to filter low-rarity values.

        Returns:
            torch.Tensor: Processed feature map.
        """
        feature_maps = layer_output.permute(0, 2, 3, 1)
        _, _, _, num_maps = feature_maps.shape

        processed_map = self.map_ponderation(self.rarity(feature_maps[0, :, :, 0]))

        for i in range(1, num_maps):
            feature = self.rarity(feature_maps[0, :, :, i])
            feature[:1, :] = 0
            feature[:, :1] = 0
            feature[-1:, :] = 0
            feature[:, -1:] = 0
            processed_map += self.map_ponderation(feature)

        processed_map = self.normalize_tensor(processed_map, min_val=0, max_val=1)
        processed_map[processed_map < threshold] = 0
        return processed_map

    def forward(self, layer_output):
        """
        Forward pass to process feature maps.

        Args:
            layer_output (list[torch.Tensor]): List of feature maps from different layers.

        Returns:
            torch.Tensor: Fused saliency map.
            torch.Tensor: Stacked feature maps.
        """
        packs = []

        for layer in layer_output:
            added = next((pack for pack in packs if pack[0].shape[-2:] == layer.shape[-2:]), None)
            if added:
                added.append(layer)
            else:
                packs.append([layer])

        groups = torch.zeros((240, 240, len(packs)), device=layer_output[0].device)

        for i, pack in enumerate(packs):
            processed_layers = [
                self.tensor_resize(self.apply_rarity(features))
                for features in pack
            ]
            groups[:, :, i] = self.normalize_tensor(self.fuse_itti(processed_layers), min_val=0, max_val=256)

        return groups.sum(dim=-1), groups

import torch
import numpy as np
import cv2
from mmdet.models.task_modules import AnchorGenerator



def get_box_scales(boxes):
    # poly_boxe = obb2poly(boxes, version='le90')
    # a_square = (poly_boxe[:, 2] - poly_boxe[:, 0]) ** 2 + (poly_boxe[:, 3] - poly_boxe[:, 1]) ** 2
    # b_square = (poly_boxe[:, 4] - poly_boxe[:, 2]) ** 2 + (poly_boxe[:, 5] - poly_boxe[:, 3]) ** 2
    # return torch.pow(a_square * b_square, 0.25)
    return torch.sqrt(boxes.areas)


def get_anchor_center_min_dis(box_centers: torch.Tensor, anchor_centers: torch.Tensor):
    """
    Args:
        box_centers: [N, 2]
        anchor_centers: [M, 2]
    Returns:
        
    """
    N, _ = box_centers.size()
    M, _ = anchor_centers.size()
    if N == 0:
        return torch.ones_like(anchor_centers)[:, 0] * 99999, (torch.zeros_like(anchor_centers)[:, 0]).long()
    acenters = anchor_centers.view(-1, 1, 2)
    acenters = acenters.repeat(1, N, 1)
    bcenters = box_centers.view(1, -1, 2)
    bcenters = bcenters.repeat(M, 1, 1)

    dis = torch.sqrt(torch.sum((acenters - bcenters) ** 2, dim=2))

    mindis, minind = torch.min(input=dis, dim=1)

    return mindis, minind


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class QueryAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.
                 ):
        super(QueryAnchorGenerator, self).__init__(
            strides,
            ratios,
            scales,
            base_sizes,
            scale_major,
            octave_base_scale,
            scales_per_octave,
            centers,
        )
        self.center_offset = 0.5

    def single_level_get_center_and_anchor(self,
                                           featmap_size,
                                           level_idx,
                                           dtype=torch.float32,
                                           device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
                ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = torch.arange(self.center_offset, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(self.center_offset, feat_h, device=device).to(dtype) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        center = torch.stack([shift_xx, shift_yy], dim=1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors, center

    def get_center_and_anchor(self, featmap_sizes, dtype=torch.float32, device='cuda'):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            dtype (:obj:`torch.dtype`): Dtype of priors.
                Default: torch.float32.
            device (str): The device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        multi_level_anchors = []
        centers = []
        for i in range(len(featmap_sizes)):
            anchors, center = self.single_level_get_center_and_anchor(
                featmap_sizes[i], level_idx=i + 1, dtype=dtype, device=device)
            multi_level_anchors.append(anchors)
            centers.append(center.view(-1, 2))
        # return multi_level_anchors, centers
        return centers  # 将batch_img_metas删除省显存

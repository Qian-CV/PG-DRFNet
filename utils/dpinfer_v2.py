from typing import List
import torch
import torch.nn.functional as F
import spconv.pytorch as spconv


def permute_to_N_HWA_K(tensor, K):
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


class DynamicInfer(object):
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):

        self.anchor_num = anchor_num
        self.num_classes = num_classes
        self.score_th = score_th
        self.context = context

    def _split_feature(self, query_logits, last_ys, last_xs, inds, feature_value):
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]  # .float()
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:
            prob = torch.sigmoid_(query_logits).view(-1)[inds]
            pidxs = prob > self.score_th
            y = last_ys.flatten(0)[pidxs]
            x = last_xs.flatten(0)[pidxs]

        if y.size(0) == 0:
            return None, None, None, None, None, None

        lt = torch.tensor([y.min(), x.min()])
        rb = torch.tensor([y.max(), x.max()])
        # block_num, last_block_list = self.find_adjacent_pixels(coordinates)

        _, fc, fh, fw = feature_value.shape
        ys = []
        xs = []
        block_list = []
        # todo: According to the predicted position, the probability
        #  points are aggregated into regions, and the feature map is first segmented.
        # last_block_list = [torch.stack(block, 0) for block in last_block_list]
        high_wide = []
        block_y = []
        block_x = []

        for i, pixes_i in enumerate(range(lt[0] * 2, rb[0] * 2 + 1)):
            for j, pixes_j in enumerate(range(lt[1] * 2, rb[1] * 2 + 1)):
                block_y.append(pixes_i)
                block_x.append(pixes_j)
        ys.append(torch.tensor(block_y))
        xs.append(torch.tensor(block_x))
        block_list.append(torch.stack((torch.tensor(block_y), torch.tensor(block_x)), dim=1))
        high_wide.append(torch.tensor([i + 1, j + 1]))

        ys = torch.cat(ys, dim=0)
        xs = torch.cat(xs, dim=0)
        inds = (ys * fw + xs).long()

        return block_list, high_wide, ys, xs, inds, None

    def build_block_feature(self, block_list, feature_value, high_wide):
        block_feature_list = []
        for i, block in enumerate(block_list):
            y_index = block[:, 0].long()
            x_index = block[:, 1].long()
            h, w = high_wide[i]
            block_feature_list.append(feature_value[:, :, y_index, x_index].view(1, 192, h, w))
        return block_feature_list

    def run_dpinfer(self, features_value, query_logits):

        last_ys, last_xs = None, None
        # query_logits = self._run_convs(features_key[-1], self.qcls_conv)\
        det_cls_query, det_bbox_query, query_anchors = [], [], []

        n_block_all = []

        for i in range(len(features_value) - 1, -1, -1):
            block_list, high_wide, last_ys, last_xs, inds, block_num = self._split_feature(
                query_logits[i + 1],
                last_ys,
                last_xs,
                None,
                features_value[i])
            n_block_all.append(block_num)
            if block_list is None:
                return None, None
            block_feature_list = self.build_block_feature(block_list, features_value[i], high_wide)  # 输出一个列表，包含所有的特征块结构

        return block_feature_list, inds

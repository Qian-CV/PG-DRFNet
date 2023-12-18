# todo: 失败！——相比1.0版本将坏的block进行删除，而不是平移，速度提升了，但是精度效果不好
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


class QueryInfer(object):
    def __init__(self, anchor_num, num_classes, score_th=0.12, context=2):

        self.anchor_num = anchor_num
        self.num_classes = num_classes
        self.score_th = score_th
        self.context = context

    def _split_feature(self, query_logits, last_ys, last_xs, anchors, feature_value):
        if last_ys is None:
            N, _, qh, qw = query_logits.size()
            assert N == 1
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = torch.where(prob > self.score_th)[0]  # .float()
            y = torch.div(pidxs, qw).int()
            x = torch.remainder(pidxs, qw).int()
        else:  # todo: 下一层推理用级联的方法还是推理的方法？
            prob = torch.sigmoid_(query_logits).view(-1)
            pidxs = prob > self.score_th
            y = last_ys.flatten(0)[pidxs]
            x = last_xs.flatten(0)[pidxs]

        if y.size(0) == 0:
            return None, None, None, None, None

        block_num = len(y)
        _, fc, fh, fw = feature_value.shape
        ys = []
        xs = []
        # todo: 以点为中心，还是先分割特征图？
        for i in range(-1 * self.context, self.context + 1):
            for j in range(-1 * self.context, self.context + 1):
                ys.append(y * 2 + i)
                xs.append(x * 2 + j)

        ys = torch.stack(ys, dim=0).t()
        xs = torch.stack(xs, dim=0).t()

        block_pixes_num = (self.context * 2 + 1) ** 2
        good_idx = (ys >= 0) & (ys < fh) & (xs >= 0) & (xs < fw)
        if ys[good_idx].shape[0] != block_num * block_pixes_num:
            bad_indexes = torch.where(good_idx == 0)[0]  # 返回一个元组，第一位是横坐标代表第几个块，第二位是纵坐标代表第几个像素
            bad_indexes = torch.unique(bad_indexes, sorted=False, dim=0)
            mask = torch.ones_like(ys, dtype=torch.bool)
            mask[bad_indexes] = False
            ys = ys[mask].reshape(-1, block_pixes_num)
            xs = xs[mask].reshape(-1, block_pixes_num)
            block_num = len(ys)

        inds = (ys * fw + xs).long()
        yx = torch.stack((ys, xs), dim=2)
        block_list = [yx[i] for i in range(block_num)]

        return block_list, ys, xs, inds, block_num

    def build_block_feature(self, block_list, feature_value):
        block_feature_list = []
        for block in block_list:
            y_index = block[:, 0].long()
            x_index = block[:, 1].long()
            block_feature_list.append(feature_value[:, :, y_index, x_index].view(1, 192, self.context * 2 + 1, -1))
        return block_feature_list

    def run_qinfer(self, features_value, query_logits):

        last_ys, last_xs = None, None
        # query_logits = self._run_convs(features_key[-1], self.qcls_conv)
        det_cls_query, det_bbox_query, query_anchors = [], [], []

        n_block_all = []

        for i in range(len(features_value) - 1, -1, -1):
            block_list, last_ys, last_xs, inds, block_num = self._split_feature(query_logits[i + 1],
                                                                                last_ys,
                                                                                last_xs,
                                                                                None,
                                                                                features_value[i])
            n_block_all.append(block_num)
            if block_list is None:
                return None, None
            block_feature_list = self.build_block_feature(block_list, features_value[i])  # 输出一个列表，包含所有的特征块结构

        return block_feature_list, inds

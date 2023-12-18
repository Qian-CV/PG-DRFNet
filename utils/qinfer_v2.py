# todo:2.0版本，对索引关键位置进行聚类，建立多个兴趣特征块，问题是聚类算法复杂度较高
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
            return None, None, None, None, None, None
        # 在上一层中寻找同一区域的像素集合
        coordinates = torch.stack((y, x), dim=1)
        block_num, last_block_list = self.find_adjacent_pixels(coordinates)
        _, fc, fh, fw = feature_value.shape
        ys = []
        xs = []
        block_list = []
        # todo: 根据预测位置，聚合概率点为区域，先分割特征图
        last_block_list = [torch.stack(block, 0) for block in last_block_list]
        high_wide = []
        block_centers, max_size = self.get_block_center(last_block_list)

        for block_center in block_centers:
            block_y = []
            block_x = []
            for i, pixes_i in enumerate(
                    range(int(((block_center[0] - (max_size / 2).ceil()) * 2).item()),
                          int(((block_center[0] + (max_size / 2).ceil()) * 2).item()) + 1)):
                for j, pixes_j in enumerate(range(int(((block_center[1] - (max_size / 2).ceil()) * 2).item()),
                                                  int(((block_center[1] + (max_size / 2).ceil()) * 2).item()) + 1)):
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

    def find_adjacent_pixels(self, coordinates):
        pixel_sets = []
        for coord in coordinates:
            found = False
            for pixel_set in pixel_sets:
                for pixel in pixel_set:
                    if self.is_adjacent(coord, pixel):
                        pixel_set.append(coord)
                        found = True
                        break
                if found:
                    break

            if not found:
                pixel_sets.append([coord])
        return len(pixel_sets), pixel_sets

    def is_adjacent(self, coord1, coord2):
        y1, x1 = coord1
        y2, x2 = coord2
        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
            return True
        return False

    def get_block_center(self, last_block_list: List):
        max_size = 0
        yx_all = []
        for n, last_block in enumerate(last_block_list):
            y_max = last_block[:, 0].max()
            y_min = last_block[:, 0].min()
            x_max = last_block[:, 1].max()
            x_min = last_block[:, 1].min()
            y = (y_min + (y_max - y_min) / 2).round()
            x = (x_min + (x_max - x_min) / 2).round()
            yx_all.append(torch.stack((y, x), dim=0))
            if (y_max - y_min + 1) > max_size or (x_max - x_min + 1) > max_size:
                num_block = n
                max_size = max((y_max - y_min + 1), (x_max - x_min + 1))
        return yx_all, max_size

    def build_block_feature(self, block_list, feature_value, high_wide):
        block_feature_list = []
        for i, block in enumerate(block_list):
            y_index = block[:, 0].long()
            x_index = block[:, 1].long()
            h, w = high_wide[i]
            block_feature_list.append(feature_value[:, :, y_index, x_index].view(1, 192, h, w))
        return block_feature_list

    def run_qinfer(self, features_value, query_logits):

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
            block_feature_list = self.build_block_feature(block_list, features_value[i],
                                                          high_wide)  # 输出一个列表，包含所有的特征块结构
            # cls_result_list = []
            # det_result_list = []
            # # query_logits_list = []
            #
            # for block_feature in block_feature_list:
            #     cls_result = self._run_convs(block_feature, self.cls_conv)
            #     bbox_result = self._run_convs(block_feature, self.bbox_conv)
            #     cls_result = permute_to_N_HWA_K(cls_result, self.num_classes)
            #     bbox_result = permute_to_N_HWA_K(bbox_result, 5)
            #     # query_logit = self._run_convs(block_feature, self.qcls_conv).view(-1)
            #     cls_result_list.append(cls_result)
            #     det_result_list.append(bbox_result)
            #     # query_logits_list.append(query_logit)
            #
            # cls_result_all = torch.cat(cls_result_list, 1)
            # bbox_result_all = torch.cat(det_result_list, 1)
            # # query_logits = torch.cat(query_logits_list, 0)
            #
            # query_anchors.append(selected_anchors)
            # det_cls_query.append(cls_result_all)
            # det_bbox_query.append(bbox_result_all)

        return block_feature_list, inds

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmcv.cnn import ConvModule, Scale, is_norm
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb
from mmrotate.models.dense_heads import RotatedRTMDetHead

from utils import utils, qinfer_v3, qinfer_v1, qinfer_v2, qinfer_v1_1
from utils.utils import get_box_scales, get_anchor_center_min_dis, permute_to_N_HWA_K
from utils.fvcore.nn.focal_loss import sigmoid_focal_loss


@MODELS.register_module()
class RotatedRTMDetGuidanceHead(RotatedRTMDetHead):
    """Detection Head of Rotated RTMDet.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representations. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Default to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Default to True.
        angle_coder (:obj:`ConfigDict` or dict): Config of angle coder.
        loss_angle (:obj:`ConfigDict` or dict, Optional): Config of angle loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = True,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 loss_angle: OptConfigType = None,
                 **kwargs) -> None:
        # self.angle_version = angle_version
        # self.use_hbbox_loss = use_hbbox_loss
        # self.is_scale_angle = scale_angle
        # self.angle_coder = TASK_UTILS.build(angle_coder)
        super().__init__(
            num_classes,
            in_channels,
            angle_version,
            use_hbbox_loss,
            scale_angle,
            angle_coder,
            loss_angle,
            **kwargs)
        if loss_angle is not None:
            self.loss_angle = MODELS.build(loss_angle)
        else:
            self.loss_angle = None

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """

        cls_scores = []
        bbox_preds = []
        angle_preds = []
        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj(reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))

            reg_dist = scale(self.rtm_reg(reg_feat).exp()).float() * stride[0]
            if self.is_scale_angle:
                angle_pred = self.scale_angle(self.rtm_ang(reg_feat)).float()
            else:
                angle_pred = self.rtm_ang(reg_feat).float()

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)
        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            assign_metrics: Tensor, stride: List[int]):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 5, H, W) for rbox loss
                or (N, num_anchors * 4, H, W) for hbox loss.
            angle_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()

        if self.use_hbbox_loss:
            bbox_pred = bbox_pred.reshape(-1, 4)
        else:
            bbox_pred = bbox_pred.reshape(-1, 5)
        bbox_targets = bbox_targets.reshape(-1, 5)

        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets
            if self.use_hbbox_loss:
                pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(
                    pos_bbox_targets[:, :4])

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_angle = angle_pred.sum() * 0
            if self.loss_angle is not None:
                angle_pred = angle_pred.reshape(-1,
                                                self.angle_coder.encode_size)
                pos_angle_pred = angle_pred[pos_inds]
                pos_angle_target = pos_bbox_targets[:, 4:5]
                pos_angle_target = self.angle_coder.encode(pos_angle_target)
                if pos_angle_target.dim() == 2:
                    pos_angle_weight = pos_bbox_weight.unsqueeze(-1)
                else:
                    pos_angle_weight = pos_bbox_weight
                loss_angle = self.loss_angle(
                    pos_angle_pred,
                    pos_angle_target,
                    weight=pos_angle_weight,
                    avg_factor=1.0)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)
            loss_angle = angle_pred.sum() * 0

        return (loss_cls, loss_bbox, loss_angle, assign_metrics.sum(),
                pos_bbox_weight.sum(), pos_bbox_weight.sum())

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predict for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [t, b, l, r] format.
            bbox_preds (list[Tensor]): Angle pred for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        decoded_bboxes = []
        decoded_hbboxes = []
        angle_preds_list = []
        for anchor, bbox_pred, angle_pred in zip(anchor_list[0], bbox_preds,
                                                 angle_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.angle_coder.encode_size)

            if self.use_hbbox_loss:
                hbbox_pred = distance2bbox(anchor, bbox_pred)
                decoded_hbboxes.append(hbbox_pred)

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            bbox_pred = distance2obb(
                anchor, bbox_pred, angle_version=self.angle_version)
            decoded_bboxes.append(bbox_pred)
            angle_preds_list.append(angle_pred)

        # flatten_bboxes is rbox, for target assign
        flatten_bboxes = torch.cat(decoded_bboxes, 1)
        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         assign_metrics_list, sampling_results_list) = cls_reg_targets

        if self.use_hbbox_loss:
            decoded_bboxes = decoded_hbboxes

        (losses_cls, losses_bbox, losses_angle, cls_avg_factors,
         bbox_avg_factors, angle_avg_factors) = multi_apply(
            self.loss_by_feat_single, cls_scores, decoded_bboxes,
            angle_preds_list, labels_list, label_weights_list,
            bbox_targets_list, assign_metrics_list,
            self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        if self.loss_angle is not None:
            angle_avg_factors = reduce_mean(
                sum(angle_avg_factors)).clamp_(min=1).item()
            losses_angle = list(
                map(lambda x: x / angle_avg_factors, losses_angle))
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_angle=losses_angle)
        else:
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                angle_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            angle_pred_list (list[Tensor]): Box angle for a single scale
                level with shape (N, num_points * angle_dim, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (
                cls_score, bbox_pred, angle_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list, angle_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(
                    bbox_pred=bbox_pred, angle_pred=angle_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


@MODELS.register_module()
class RotatedRTMDetGuidanceSepBNHead(RotatedRTMDetGuidanceHead):
    """Rotated RTMDetHead with separated BN layers and shared conv layers.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        scale_angle (bool): Does not support in RotatedRTMDetSepBNHead,
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether to apply exponential on bbox_pred.
            Defaults to False.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 query_head_size: list,
                 query_layer_train,
                 small_object_scale,
                 small_center_dis_coeff,
                 query_loss_gammas,
                 query_loss_weights,
                 cls_layer_weight,
                 reg_layer_weight,
                 query_infer: str,
                 infer_version: str,
                 layers_whole_test: list,
                 layers_key_test: list,
                 layers_value_test: list,
                 query_threshold: float,
                 context: float,
                 loss_query_weight: float,

                 share_conv: bool = True,
                 scale_angle: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg: bool = False,
                 **kwargs) -> None:

        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        assert scale_angle is False, \
            'scale_angle does not support in RotatedRTMDetSepBNHead'
        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            scale_angle=False,
            **kwargs)

        # 新加入的初始化
        self.query_layer_train = query_layer_train
        self.query_featmap_sizes = None
        self.query_anchor_generator = utils.QueryAnchorGenerator([4, 8, 16, 32], [0.5, 1.0, 2.0], scales=[8])
        self.small_obj_scale = small_object_scale
        self.small_center_dis_coeff = small_center_dis_coeff
        self.query_loss_gammas = query_loss_gammas
        self.query_loss_weights = query_loss_weights
        self.cls_layer_weight = cls_layer_weight
        self.reg_layer_weight = reg_layer_weight
        self.query_infer = query_infer
        self.layers_whole_test = layers_whole_test
        self.layers_key_test = layers_key_test
        self.layers_value_test = layers_value_test
        self.infer_version = infer_version
        self.context = context
        self.loss_query_weight = loss_query_weight

        self.query_head = Head_3x3(query_head_size[0], query_head_size[1], query_head_size[2], query_head_size[3])
        if infer_version == 'v1':
            self.qInfer = qinfer_v1.QueryInfer(1, num_classes, query_threshold, context=context)
        elif infer_version == 'v1.1':
            self.qInfer = qinfer_v1_1.QueryInfer(1, num_classes, query_threshold, context=context)
        elif infer_version == 'v2':
            self.qInfer = qinfer_v2.QueryInfer(1, num_classes, query_threshold, context=context)
        elif infer_version == 'v3':
            self.qInfer = qinfer_v3.QueryInfer(1, num_classes, query_threshold, context=context)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()
        if self.with_objectness:
            self.rtm_obj = nn.ModuleList()
        for n in range(len(self.prior_generator.strides)):
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

            self.rtm_cls.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            self.rtm_ang.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.angle_coder.encode_size,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
            if self.with_objectness:
                self.rtm_obj.append(
                    nn.Conv2d(
                        self.feat_channels,
                        1,
                        self.pred_kernel_size,
                        padding=self.pred_kernel_size // 2))

        if self.share_conv:
            for n in range(len(self.prior_generator.strides)):
                for i in range(self.stacked_convs):
                    self.cls_convs[n][i].conv = self.cls_convs[0][i].conv
                    self.reg_convs[n][i].conv = self.reg_convs[0][i].conv

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls, self.rtm_reg,
                                             self.rtm_ang):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)
        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

    @torch.no_grad()
    def get_query_gt(self, small_anchor_centers, targets):
        small_gt_cls = []
        for lind, anchor_center in enumerate(small_anchor_centers):
            per_layer_small_gt = []
            for target_per_image in targets:
                target_box_scales = get_box_scales(
                    target_per_image.bboxes)  # fixme: 这里需要对加入旋转角的框计算面积，括号里面写instance的bbox tensor,新框架更新了面积方法
                small_inds = (target_box_scales < self.small_obj_scale[lind][1]) & (
                        target_box_scales >= self.small_obj_scale[lind][0])
                small_boxes = target_per_image.bboxes[small_inds]
                small_boxes_centers = small_boxes.centers
                center_dis, minarg = get_anchor_center_min_dis(small_boxes_centers, anchor_center)
                small_obj_target = torch.zeros_like(center_dis)

                if len(small_boxes) != 0:
                    min_small_target_scale = (target_box_scales[small_inds])[minarg]
                    small_obj_target[center_dis < min_small_target_scale * self.small_center_dis_coeff[lind]] = 1

                per_layer_small_gt.append(small_obj_target)
            small_gt_cls.append(torch.stack(per_layer_small_gt))

        return small_gt_cls

    def query_loss(self, gt_small_obj, pred_small_obj, gammas, weights):
        pred_logits = [permute_to_N_HWA_K(x, 1).flatten() for x in pred_small_obj]
        gts = [x.flatten() for x in gt_small_obj]
        sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)
        loss = self.loss_query_weight * sum(
            [sigmoid_focal_loss_jit(x, y, alpha=0.25, gamma=g, reduction="mean") * w for (x, y, g, w) in
             zip(pred_logits, gts, gammas, weights)])
        return loss

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """
        if self.training or (not self.training and not self.query_infer):
            cls_scores = []
            bbox_preds = []
            angle_preds = []
            for idx, (x, stride) in enumerate(
                    zip(feats, self.prior_generator.strides)):
                cls_feat = x
                reg_feat = x

                for cls_layer in self.cls_convs[idx]:
                    cls_feat = cls_layer(cls_feat)
                cls_score = self.rtm_cls[idx](cls_feat)

                for reg_layer in self.reg_convs[idx]:
                    reg_feat = reg_layer(reg_feat)

                if self.with_objectness:
                    objectness = self.rtm_obj[idx](reg_feat)
                    cls_score = inverse_sigmoid(
                        sigmoid_geometric_mean(cls_score, objectness))
                if self.exp_on_reg:
                    reg_dist = self.rtm_reg[idx](reg_feat).exp() * stride[0]
                else:
                    reg_dist = self.rtm_reg[idx](reg_feat) * stride[0]

                angle_pred = self.rtm_ang[idx](reg_feat)

                cls_scores.append(cls_score)
                bbox_preds.append(reg_dist)
                angle_preds.append(angle_pred)
            with torch.no_grad():  # fixme: 增加了防止爆显存
                query_feature = [feats[x] for x in self.query_layer_train]
                self.query_featmap_sizes = [featmap.size()[-2:] for featmap in query_feature]
            query_logits = self.query_head(query_feature)
            return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), tuple(query_logits)
        else:
            return self.forward_infer(feats)

    def forward_infer(self, feats: Tuple[Tensor, ...]) -> tuple:
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        # 搭建whole,query's feature and strides
        features_whole = [feats[x] for x in self.layers_whole_test]
        features_key = [feats[x] for x in self.layers_key_test]
        features_query = [feats[x] for x in self.layers_value_test]
        strides_whole = [self.prior_generator.strides[x] for x in self.layers_whole_test]
        strides_query = [self.prior_generator.strides[x] for x in self.layers_value_test]
        # 下面循环中用到的卷积参数对应层编号
        layer_num = len(feats) - len(self.layers_whole_test)

        for idx, (x, stride) in enumerate(
                zip(features_whole, strides_whole)):
            cls_feat = x
            reg_feat = x

            for cls_layer in self.cls_convs[idx + layer_num]:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.rtm_cls[idx + layer_num](cls_feat)

            for reg_layer in self.reg_convs[idx + layer_num]:
                reg_feat = reg_layer(reg_feat)

            if self.with_objectness:
                objectness = self.rtm_obj[idx + layer_num](reg_feat)
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, objectness))
            if self.exp_on_reg:
                reg_dist = self.rtm_reg[idx + layer_num](reg_feat).exp() * stride[0]
            else:
                reg_dist = self.rtm_reg[idx + layer_num](reg_feat) * stride[0]

            angle_pred = self.rtm_ang[idx + layer_num](reg_feat)

            cls_scores.append(cls_score)
            bbox_preds.append(reg_dist)
            angle_preds.append(angle_pred)

        # 开始计算query层的回归和分类
        query_logits = self.query_head(features_key)
        # 获得特征分块结果
        block_feature_list, anchor_inds = self.qInfer.run_qinfer(features_query, query_logits)
        self.anchor_inds = anchor_inds
        # 获得feature_mapsize
        num_levels = len(feats)
        self.featmap_sizes = [feats[i].shape[-2:] for i in range(num_levels)]

        if block_feature_list:
            cls_result_list = []
            det_result_list = []
            angle_result_list = []
            for block_feature in block_feature_list:
                cls_feat = block_feature
                reg_feat = block_feature
                for cls_layer in self.cls_convs[1]:
                    cls_feat = cls_layer(cls_feat)
                cls_score = self.rtm_cls[1](cls_feat)

                for reg_layer in self.reg_convs[1]:
                    reg_feat = reg_layer(reg_feat)

                if self.with_objectness:
                    objectness = self.rtm_obj[1](reg_feat)
                    cls_score = inverse_sigmoid(
                        sigmoid_geometric_mean(cls_score, objectness))
                if self.exp_on_reg:
                    reg_dist = self.rtm_reg[1](reg_feat).exp() * strides_query[0][0]
                else:
                    reg_dist = self.rtm_reg[1](reg_feat) * strides_query[0][0]

                angle_pred = self.rtm_ang[1](reg_feat)

                # 将每个特征块的检测结果合并到一个list中
                cls_result_list.append(cls_score)
                det_result_list.append(reg_dist)
                angle_result_list.append(angle_pred)

            if self.infer_version == 'v2':
                cls_result_all = cls_result_list
                bbox_result_all = det_result_list
                angle_result_all = angle_result_list
            else:
                cls_result_all = torch.cat(cls_result_list, 0)
                bbox_result_all = torch.cat(det_result_list, 0)
                angle_result_all = torch.cat(angle_result_list, 0)
                # 不同的infer_version有着不同的处理方式
                if self.infer_version == 'v1' or self.infer_version == 'v1.1':
                    cls_result_all.view(-1, self.context * 2 + 1, self.context * 2 + 1)
                    bbox_result_all.view(-1, self.context * 2 + 1, self.context * 2 + 1)
                    angle_result_all.view(-1, self.context * 2 + 1, self.context * 2 + 1)
                # 将结果插入到多层结果的第一的位置
            cls_scores.insert(0, cls_result_all)
            bbox_preds.insert(0, bbox_result_all)
            angle_preds.insert(0, angle_result_all)

        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds), None

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     query_logits: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box predict for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [t, b, l, r] format.
            bbox_preds (list[Tensor]): Angle pred for each scale
                level with shape (N, num_anchors * angle_dim, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(cls_scores, bbox_preds, angle_preds,
                                         batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # 开始计算query损失
        with torch.no_grad():
            query_centers = self.query_anchor_generator.get_center_and_anchor(
                self.query_featmap_sizes)  # fixme:是否使用旋转的fake anchor
            gt_query = self.get_query_gt(query_centers, batch_gt_instances)
        _query_loss = self.query_loss(gt_query, query_logits, self.query_loss_gammas, self.query_loss_weights)
        loss_dict['loss_guidance'] = _query_loss
        # 动态感受野，平衡不同层的权重
        if type(self.cls_layer_weight) and type(self.reg_layer_weight) is list:
            loss_dict['loss_cls'] = [x * y for x, y in zip(loss_dict['loss_cls'], self.cls_layer_weight)]
            loss_dict['loss_bbox'] = [x * y for x, y in zip(loss_dict['loss_bbox'], self.reg_layer_weight)]
        # elif type(self.cls_layer_weight) and type(self.reg_layer_weight) is str:
        #     cls_layer_weight = self.dynamic_receptive_field(batch_gt_instances)
        #     reg_layer_weight = cls_layer_weight
        #     loss_dict['loss_cls'] = [x * y for x, y in zip(loss_dict['loss_cls'], cls_layer_weight)]
        #     loss_dict['loss_bbox'] = [x * y for x, y in zip(loss_dict['loss_bbox'], reg_layer_weight)]

        return loss_dict

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        query_logits: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angle for each scale level
                with shape (N, num_points * angle_dim, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 5),
                  the last dimension 5 arrange as (x, y, w, h, t).
        """
        if not self.query_infer:
            return super().predict_by_feat(cls_scores, bbox_preds, angle_preds,
                                           score_factors, batch_img_metas, cfg, rescale, with_nms)
        else:
            assert len(cls_scores) == len(bbox_preds)
            num_levels = len(self.featmap_sizes)

            if score_factors is None:
                # e.g. Retina, FreeAnchor, Foveabox, etc.
                with_score_factors = False
            else:
                # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
                with_score_factors = True
                assert len(cls_scores) == len(score_factors)

            mlvl_priors = self.prior_generator.grid_priors(
                self.featmap_sizes,
                dtype=cls_scores[-1].dtype,
                device=cls_scores[-1].device)

            result_list = []

            for img_id in range(len(batch_img_metas)):
                img_meta = batch_img_metas[img_id]
                cls_score_list = select_single_mlvl(
                    cls_scores, img_id, detach=True)
                bbox_pred_list = select_single_mlvl(
                    bbox_preds, img_id, detach=True)
                angle_pred_list = select_single_mlvl(
                    angle_preds, img_id, detach=True)
                if with_score_factors:
                    score_factor_list = select_single_mlvl(
                        score_factors, img_id, detach=True)
                else:
                    score_factor_list = [None for _ in range(num_levels - 1)]

                # 构建whole和value的预选框
                anchors_whole = [mlvl_priors[x] for x in self.layers_whole_test]
                anchors_value = [mlvl_priors[x] for x in self.layers_value_test]
                anchors_all = mlvl_priors[1:]

                if self.anchor_inds is not None:
                    select_anchors_value = anchors_value[0][self.anchor_inds.flatten(0)].view(-1, 2)
                    anchors_all[0] = select_anchors_value
                mlvl_priors_choose = anchors_whole if len(cls_score_list) == len(
                    self.layers_whole_test) else anchors_all

                results = self._predict_by_feat_single(
                    cls_score_list=cls_score_list,
                    bbox_pred_list=bbox_pred_list,
                    angle_pred_list=angle_pred_list,
                    score_factor_list=score_factor_list,
                    mlvl_priors=mlvl_priors_choose,
                    img_meta=img_meta,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms)
                result_list.append(results)
            return result_list

    # def dynamic_receptive_field(self, batch_gt_instances) -> List:
    #     score = torch.zeros(4)
    #     base_recognition_size = 8
    #     distinguish_thr = [base_recognition_size * x for x in [4, 8, 16, 32]]
    #     field_1, field_2, field_3, field_4 = distinguish_thr
    #     for instance in batch_gt_instances:
    #         length_size = torch.sqrt(instance.bboxes.areas)
    #         for item in length_size:
    #             if item.item() <= field_1:
    #                 score[0] += 1
    #             elif field_1 < item.item() <= field_2:
    #                 score[1] += 1
    #             elif field_2 < item.item() <= field_3:
    #                 score[2] += 1
    #             else:
    #                 score[3] += 1
    #
    #     receptive_field_weight = score
    #     return receptive_field_weight


class Head_3x3(nn.Module):
    def __init__(self, in_channels, conv_channels, num_convs, pred_channels, pred_prior=None):
        super(Head_3x3, self).__init__()
        self.num_convs = num_convs

        self.subnet = []
        channels = in_channels
        for i in range(self.num_convs):
            layer = nn.Conv2d(channels, conv_channels, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.add_module('layer_{}'.format(i), layer)
            self.subnet.append(layer)
            channels = conv_channels

        self.pred_net = nn.Conv2d(channels, pred_channels, kernel_size=3, stride=1, padding=1)

        nn.init.xavier_normal_(self.pred_net.weight)
        if pred_prior is not None:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            nn.init.constant_(self.pred_net.bias, bias_value)
        else:
            nn.init.constant_(self.pred_net.bias, 0)

    def forward(self, features):
        preds = []
        for feature in features:
            x = feature
            for i in range(self.num_convs):
                x = nn.functional.relu(self.subnet[i](x))
            preds.append(self.pred_net(x))
        return preds

    def get_params(self):
        weights = [x.weight for x in self.subnet] + [self.pred_net.weight]
        biases = [x.bias for x in self.subnet] + [self.pred_net.bias]
        return weights, biases

# coding: utf-8
import cv2
import mmcv
import numpy as np
import os
import torch

from mmdet.apis import inference_detector, init_detector

from utils_my import mkdir
from argparse import ArgumentParser
import torch.nn as nn
from typing import Optional, Sequence, Union
from mmcv.transforms import Compose
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
from mmcv.ops import RoIPool

import matplotlib.pyplot as plt

ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def inference_detector(
        model: nn.Module,
        imgs: ImagesType,
        test_pipeline: Optional[Compose] = None
) -> Union[DetDataSample, SampleList]:
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`DetDataSample` or list[:obj:`DetDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg

    if test_pipeline is None:
        cfg = cfg.copy()
        test_pipeline = get_test_pipeline_cfg(cfg)
        if isinstance(imgs[0], np.ndarray):
            # Calling this method across libraries will result
            # in module unregistered error if not prefixed with mmdet.
            test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'

        test_pipeline = Compose(test_pipeline)

    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    result_list = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # TODO: remove img_id.
            data_ = dict(img=img, img_id=0)
        else:
            # TODO: remove img_id.
            data_ = dict(img_path=img, img_id=0)
        # build the data pipeline
        data_ = test_pipeline(data_)

        data_['inputs'] = [data_['inputs']]
        data_['data_samples'] = [data_['data_samples']]

        # forward the model
        with torch.no_grad():
            results = model.test_step(data_)

        result_list.append(results)

    if not is_batch:
        return result_list[0]
    else:
        return result_list


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # threshold_min = 0.53 # dota是0.3
    # threshold_max = 0.59 # dota是0.4
    threshold_min = 0.12  # 用于SSDD
    threshold_max = threshold_min  # 用于SSDD
    condition_min = heatmap < np.max(heatmap) * threshold_min
    condition_max = heatmap >= np.max(heatmap) * threshold_max
    heatmap = np.where(condition_min, 0, heatmap)
    heatmap = np.where(condition_max, 1, heatmap)

    # plt.imshow(heatmap, 'jet')
    return heatmap


def draw_feature_map(model, img_path, save_dir):
    '''
    :param model: 加载了参数的模型
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    mkdir(save_dir)
    img = mmcv.imread(img_path)
    img_y, img_x, img_c = img.shape
    if img_y > img_x:
        img = img[img_y - img_x:img_y, :, :]
    elif img_x > img_y:
        img = img[:, img_x - img_y:img_x, :]
    modeltype = str(type(model)).split('.')[-1].split('\'')[0]
    model.eval()
    model.draw_heatmap = True
    featuremaps = inference_detector(model,
                                     img)  # 1.这里需要改model，让其在forward的最后return特征图。我这里return的是一个Tensor的tuple，每个Tensor对应一个level上输出的特征图。
    i = 0
    for featuremap in featuremaps:
        heatmap = featuremap_2_heatmap(featuremap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.6 + img  # 这里的0.4是热力图强度因子
        # superimposed_img = heatmap
        cv2.imwrite(os.path.join(save_dir, 'featuremap_' + str(i) + '.png'), superimposed_img)  # 将图像保存到硬盘
        i = i + 1


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('save_dir', help='Dir to save heatmap image')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    draw_feature_map(model, args.img, args.save_dir)


if __name__ == '__main__':
    main()

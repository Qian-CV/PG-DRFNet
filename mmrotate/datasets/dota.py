# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS
import os

@DATASETS.register_module()
class DOTADataset(BaseDataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255)]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'png',
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        ######################################################################################
        # todo:
        """
        下面是我新加入的代码，对于DOTA数据集的验证部分，自定义部分trainval进行读取，避免了爆显存的问题。
        可以自定义的划分训练集和验证集，给定分好的txt文件，自动判断在是目录（全部数据标注）还是txt（部分数据标注），
        用analyze_tools中的utils_my.py中进行划分。
        """
        ann_folder = self.ann_file
        if self.test_mode is True and not os.path.isdir(ann_folder) and ann_folder != '':
            ann_subdir = os.path.dirname(ann_folder)
            ff = open(ann_subdir + '/val_from_trainval.txt', 'r')
            ann_files = ff.readlines()
            ff.close()
            for i, ann_file in enumerate(ann_files):
                ann_files[i] = ann_subdir + '/labelTxt/' + ann_file.split()[
                    0] + '.txt'
        else:
            ann_files = glob.glob(ann_folder + '/*.txt')
        ########################################################################################
        data_list = []
        if self.ann_file == '':
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)

            return data_list
        else:
            # 没错，是我注释掉的
            # txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            # if len(txt_files) == 0:
            #     raise ValueError('There is no txt file in '
            #                      f'{self.ann_file}')
            for txt_file in ann_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                                 img_name)

                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.split()
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = bbox_info[8]
                        instance['bbox_label'] = cls_map[cls_name]
                        difficulty = int(bbox_info[9])
                        if difficulty > self.diff_thr:
                            instance['ignore_flag'] = 1
                        else:
                            instance['ignore_flag'] = 0
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)

            return data_list

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """

        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]


@DATASETS.register_module()
class DOTAv15Dataset(DOTADataset):
    """DOTA-v1.5 dataset for detection.

    Note: ``ann_file`` in DOTAv15Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv15Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60)]
    }


@DATASETS.register_module()
class DOTAv2Dataset(DOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
         'helipad'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60), (119, 11, 32),
                    (0, 0, 142)]
    }

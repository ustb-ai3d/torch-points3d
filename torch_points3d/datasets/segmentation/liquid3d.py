import os
import os.path as osp
import shutil
import json
from tqdm.auto import tqdm as tq
from itertools import repeat, product
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
import torch_geometric.transforms as T
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.shapenet_part_tracker import ShapenetPartTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties
from torch_points3d.utils.download import download_url


def gen_liquid3d_raw():
    # 大概的数据结构，每执行一次本函数，取出一个【输入+输出】
    # in_pos: [points_num, 3] 每个点的坐标
    # in_feats: [points_num, feats_channel] 每个点的特征。如果模型不支持输入特征，则忽略该向量
    # out: [points_num, 3] 每个点的输出。原始点云分割任务中是[points_num, 16]，对应每个点的16个通道代表16个类的概率预测。在这里我们只输出3个通道，可能是速度也可能是加速度的xyz值。

    in_pos, in_feats, out = torch.ones((None, 3)), torch.ones((None, 3)), torch.ones((None, 3))  # 随便写的
    return in_pos, in_feats, out


class Liquid3D(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, split="trainval"):
        super().__init__(root, transform, pre_transform, pre_filter)

        if split == "train":
            path = self.processed_paths[0]
            raw_path = self.processed_raw_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
            raw_path = self.processed_raw_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
            raw_path = self.processed_raw_paths[2]
        elif split == "trainval":
            path = self.processed_paths[3]
            raw_path = self.processed_raw_paths[3]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))

        self.data, self.slices = self.load(path)

    @property
    def raw_file_names(self):
        return ["some_file_1", "some_file_2", ...]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

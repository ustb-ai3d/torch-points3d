import os
import os.path as osp
import json
from tqdm.auto import tqdm as tq
import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties


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

    @property
    def processed_raw_paths(self):
        processed_raw_paths = [
            os.path.join(self.processed_dir, "raw_{}".format(s)) for s in ["train", "val", "test", "trainval"]
        ]
        return processed_raw_paths

    @property
    def processed_file_names(self):
        return [os.path.join("{}.pt".format(split)) for split in ["train", "val", "test", "trainval"]]

    def _process_filenames(self, filenames):
        data_raw_list = []
        data_list = []

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for name in tq(filenames):
            id_scan += 1
            data = read_txt_array(osp.join(self.raw_dir, name))  # 按行读文件，每个数转为浮点型存入data列表
            pos = data[:, :3]
            x = data[:, 3:6]
            id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
            data = Data(pos=pos, x=x, id_scan=id_scan_tensor)
            data = SaveOriginalPosId()(data)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            data_raw_list.append(data.clone() if has_pre_transform else data)
            if has_pre_transform:
                data = self.pre_transform(data)
                data_list.append(data)
        if not has_pre_transform:
            return [], data_raw_list
        return data_raw_list, data_list

    def _save_data_list(self, datas, path_to_datas, save_bool=True):
        if save_bool:
            torch.save(self.collate(datas), path_to_datas)

    def _re_index_trainval(self, trainval):
        if len(trainval) == 0:
            return trainval
        train, val = trainval
        for v in val:
            v.id_scan += len(train)
        assert (train[-1].id_scan + 1 == val[0].id_scan).item(), (train[-1].id_scan, val[0].id_scan)
        return train + val

    def process(self):
        raw_trainval = []
        trainval = []
        for i, split in enumerate(["train", "val"]):
            path = osp.join(self.raw_dir, "train_test_split", f"shuffled_{split}_file_list.json")
            with open(path, "r") as f:
                filenames = [
                    osp.sep.join(name.split("/")[1:]) + ".txt" for name in json.load(f)
                ]  # Removing first directory.
            data_raw_list, data_list = self._process_filenames(sorted(filenames))
            if split == "train" or split == "val":
                if len(data_raw_list) > 0:
                    raw_trainval.append(data_raw_list)
                trainval.append(data_list)

            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(data_raw_list, self.processed_raw_paths[i], save_bool=len(data_raw_list) > 0)

        self._save_data_list(self._re_index_trainval(trainval), self.processed_paths[3])
        self._save_data_list(
            self._re_index_trainval(raw_trainval), self.processed_raw_paths[3], save_bool=len(raw_trainval) > 0
        )

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


class ShapeNetDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        self.train_dataset = Liquid3D(
            self._data_path,
            split="train",
            pre_transform=self.pre_transform,
            transform=self.train_transform,
        )

        self.val_dataset = Liquid3D(
            self._data_path,
            split="val",
            pre_transform=self.pre_transform,
            transform=self.val_transform,
        )

        self.test_dataset = Liquid3D(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)

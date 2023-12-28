import os
import os.path as osp
import json
from tqdm.auto import tqdm as tq
import numpy as np
import torch
import zstandard as zstd
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import read_txt_array
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
from torch_points3d.datasets.base_dataset import BaseDataset, save_used_properties

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

    def _handle_data(self, data):
        data_idxs = np.arange(len(data) - 1)
        assert (len(data_idxs) > 0)
        sample_list = []
        for data_i in data_idxs:
            sample = {}

            for k in ['pos', 'vel', 'grav', 'm', 'viscosity']:
                if k in data[data_i]:
                    sample[k] = np.stack([
                        data[data_i + i * self.stride].get(
                            k, None).astype("float32")
                        for i in range(2)
                    ], 0)
                else:
                    sample[k] = [None]

            for k in ['box', 'box_normals']:
                if k in data[0]:
                    sample[k] = np.stack([
                        data[0].get(k, None).astype("float32")
                        for i in range(2)
                    ], 0)
                else:
                    sample[k] = [np.empty((0, 3))]
                sample[k] = np.reshape(sample[k], (len(sample[k]), -1, 3))

            for k in ['frame_id', 'scene_id']:
                sample[k] = np.stack([
                    data[data_i + i * self.stride].get(k, None)
                    for i in range(2)
                ], 0)

            if sample['grav'][0] is not None:
                sample['grav'] = np.full_like(sample['vel'],
                                              np.expand_dims(
                                                  sample['grav'],
                                                  1))  # / 100

            # sample = self.transform(sample)
            sample_list.append(sample)

        return sample_list

    def _process_filenames(self, filenames, path):
        data_raw_list = []
        data_list = []

        has_pre_transform = self.pre_transform is not None

        id_scan = -1
        for name in tq(filenames):
            id_scan += 1
            decompressor = zstd.ZstdDecompressor()
            with open(os.path.join(path, name), 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()), raw=False)
            # data = read_txt_array(osp.join(path, name))  # 按行读文件，每个数转为浮点型存入data列表
            sample_list = self._handle_data(data)
            for sample in sample_list:
                pos = torch.concat((sample['pos'][0], sample['box'][0]), dim=0)
                x1 = torch.concat((sample['vel'][0], torch.zeros_like(sample['box'][0])), dim=0)
                x2 = torch.concat((torch.zeros_like(sample['pos'][0]), sample['box_normals'][0]), dim=0)
                x = torch.concat((x1, x2), dim=-1)
                y = torch.concat((sample['pos'][1], sample['box'][1]), dim=0)
                id_scan_tensor = torch.from_numpy(np.asarray([id_scan])).clone()
                data = Data(pos=pos, x=x, y=y, id_scan=id_scan_tensor)
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
        trainval = []
        for i, split in enumerate(["train", "valid"]):
            path = osp.join(self.raw_dir, split)
            filenames = [name for name in os.listdir(path) if name.split('.')[-1] == 'zst']
            data_raw_list, data_list = self._process_filenames(sorted(filenames), path)
            if split == "train" or split == "valid":
                trainval.append(data_list)

            self._save_data_list(data_list, self.processed_paths[i])
            self._save_data_list(data_raw_list, self.processed_raw_paths[i], save_bool=len(data_raw_list) > 0)

        self._save_data_list(self._re_index_trainval(trainval), self.processed_paths[3])

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


class Liquid3dDataset(BaseDataset):
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

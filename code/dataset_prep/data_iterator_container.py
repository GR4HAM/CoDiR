import os
import torch
from torchvision import datasets, transforms
from cfg.config_general import cfg


import pickle
from torch.utils.data.sampler import SubsetRandomSampler
# from batch_class_sampler import BatchClassSampler

from dataset_prep.coco_dali_pipeline import COCOPipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# from dataset_prep.data_prep_manager import Data_prep_manager

class Data_iterator_container(object):
    def __init__(self, data_dir, dataset, batch_size, dataset_split='train', per_class=False,
                 shuffle=None, num_workers=0, datasource=None, im_size=128,
                 dali=True):
        assert os.path.exists(data_dir)
        assert dataset in {'COCO' }
        self.dataset = dataset
        self.data_dir  = data_dir
        self.batch_size = batch_size
        self.per_class = per_class
        self.im_size = im_size
        self.dataset_split = dataset_split
        self.datasource = datasource


        if self.dataset in {'COCO'}:
            if shuffle is None:
                self.shuffle = True
            else:
                self.shuffle = shuffle

            if cfg.DEBUG:
                self.epoch_length = 20
            else:
                self.epoch_length = datasource.epoch_length // self.batch_size + 1 * (
                    not datasource.epoch_length % self.batch_size == 0)

            if dali:
                if self.dataset_split == 'train':
                    self.stop_at_epoch = True
                else:
                    self.stop_at_epoch = True
                device_id = 0

                self.data_pipe = COCOPipeline(datasource, 1, device_id)
                self.loader_labels = ["input_img", "target_class", "target_env", "indices", "img_ids"]

                self.loader = DALIGenericIterator(
                    self.data_pipe,
                    self.loader_labels,
                    datasource.epoch_length / 1,
                    stop_at_epoch=self.stop_at_epoch, auto_reset=True)

            else:
                self.loader = torch.utils.data.DataLoader( datasource,
                    batch_size=batch_size, shuffle=self.shuffle, num_workers=num_workers)

        else:
            raise ValueError("dataset not implemented")


    def get_iterator(self):
        return iter(self.loader)

    def get_img_ids(self):
        return self.dpm.img_ids

    def get_img_names(self):
        return self.dpm.images

    def get_img_loc(self):
        return self.dpm.img_data_loc

    def delete_loader(self):
        del self.loader




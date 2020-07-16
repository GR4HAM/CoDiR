import torch
import torchvision.transforms as transforms
import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
import yaml
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './../..')))
sys.path.append(dir_path)
code_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './..')))
sys.path.append(code_path)
from cfg.config_general import cfg, cfg_from_file
from dataset_prep.data_iterator_container import Data_iterator_container
from trainer import Trainer
from trainer_master import full_training




def parse_args():
    parser = argparse.ArgumentParser(description='Train a NLP with backpropagation or evolutionary strategies')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='../cfg/COCO_R18_CLASS.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='../../output_dir/')
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=None)
    args = parser.parse_args()
    return args

def run_main(args):

    print('Using config:')
    pprint.pprint(cfg)

    #seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    cfg.SEED = args.manualSeed

    num_workers=0
    if sys.executable[:13] == '/export/home1':
        print("remote server interpreter detected")
        if not args.data_dir == '':
            cfg.DATASET.ROOT_DATA_DIR = args.data_dir
        else:
            cfg.DATASET.ROOT_DATA_DIR = '../../../Data/'

    elif sys.executable[:7] == '/home/g':
        print("local ubuntu interpreter detected")
        print("turning off dali")
        cfg.TRAIN.NVIDIA_DALI = False
        if not args.data_dir == '':
            cfg.DATASET.ROOT_DATA_DIR = args.data_dir
        else:
            cfg.DATASET.ROOT_DATA_DIR = '~/Data/'
    else:
        raise ValueError("Unable to recognize this interpreter at "+str(sys.executable))

    cfg.DATASET.ROOT_DATA_DIR = os.path.expanduser(cfg.DATASET.ROOT_DATA_DIR)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    assert os.path.exists(args.output_dir)
    output_dir_run = 'out_%s_%s_%s' % \
                 (cfg.DATASET.NAME, cfg.CONFIG_NAME, timestamp)

    trainer = Trainer(args.output_dir, output_dir_run)

    batch_size = cfg.TRAIN.BATCH_SIZE


    if cfg.SAVE.LOG_PARAMS:
        # copy config file to output dir
        dst = os.path.join(trainer.log_dir, "config_file.yml")
        with open(dst, 'w') as yaml_file:
            yaml.dump(cfg, yaml_file, default_flow_style=False)




    print("creating dataloader")


    from dataset_prep.coco import CocoCaptions
    #prepare datasources
    data_source_train = CocoCaptions(os.path.join(cfg.DATASET.ROOT_DATA_DIR, "CocoCaptions","processed"),
                                     split="train",
                                     transforms=transforms.Compose([
                                         # transforms.RandomCrop(128, pad_if_needed=True),
                                         transforms.RandomResizedCrop(cfg.MODEL.IMAGE_DIM, (
                                             cfg.TRAIN.IMAGE_AREA_FACTOR_LOWER_BOUND, 1.0)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]),
                                     class_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.CLASS,
                                                          "label_origin": cfg.MODEL.CLASS.LABEL_ORIGIN,
                                                          "by_tag_type": cfg.MODEL.CLASS.BY_TAG_TYPE,
                                                          "type_of_class": cfg.MODEL.CLASS.TYPE_OF_CLASS},
                                     env_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.ENV,
                                                        "label_origin": cfg.MODEL.ENVIRONMENT.LABEL_ORIGIN,
                                                        "by_tag_type": cfg.MODEL.ENVIRONMENT.BY_TAG_TYPE,
                                                        "type_of_class": cfg.MODEL.ENVIRONMENT.TYPE_OF_CLASS},
                                     batch_size=cfg.TRAIN.BATCH_SIZE)
    data_source_val = CocoCaptions(os.path.join(cfg.DATASET.ROOT_DATA_DIR, "CocoCaptions", "processed"),
                                     split="validation",
                                     transforms=transforms.Compose([
                                         transforms.RandomResizedCrop(cfg.MODEL.IMAGE_DIM, (
                                             cfg.TRAIN.IMAGE_AREA_FACTOR_LOWER_BOUND, 1.0)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]),
                                   class_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.CLASS,
                                                        "label_origin": cfg.MODEL.CLASS.LABEL_ORIGIN,
                                                        "by_tag_type": cfg.MODEL.CLASS.BY_TAG_TYPE,
                                                        "type_of_class": cfg.MODEL.CLASS.TYPE_OF_CLASS},
                                   env_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.ENV,
                                                      "label_origin": cfg.MODEL.ENVIRONMENT.LABEL_ORIGIN,
                                                      "by_tag_type": cfg.MODEL.ENVIRONMENT.BY_TAG_TYPE,
                                                      "type_of_class": cfg.MODEL.ENVIRONMENT.TYPE_OF_CLASS},
                                     batch_size=cfg.TRAIN.BATCH_SIZE)
    data_source_test = CocoCaptions(os.path.join(cfg.DATASET.ROOT_DATA_DIR, "CocoCaptions", "processed"),
                                   split="test",
                                   transforms=transforms.Compose([
                                       transforms.RandomResizedCrop(cfg.MODEL.IMAGE_DIM, (
                                           cfg.TRAIN.IMAGE_AREA_FACTOR_LOWER_BOUND, 1.0)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]),
                                    class_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.CLASS,
                                                         "label_origin": cfg.MODEL.CLASS.LABEL_ORIGIN,
                                                         "by_tag_type": cfg.MODEL.CLASS.BY_TAG_TYPE,
                                                         "type_of_class": cfg.MODEL.CLASS.TYPE_OF_CLASS},
                                    env_vocab_options={"size": cfg.MODEL.VOCAB_SIZE.ENV,
                                                       "label_origin": cfg.MODEL.ENVIRONMENT.LABEL_ORIGIN,
                                                       "by_tag_type": cfg.MODEL.ENVIRONMENT.BY_TAG_TYPE,
                                                       "type_of_class": cfg.MODEL.ENVIRONMENT.TYPE_OF_CLASS},
                                    batch_size=cfg.TRAIN.BATCH_SIZE)

    #prepare the data containers, which determines how data will be fed
    data_container_train = Data_iterator_container(cfg.DATASET.ROOT_DATA_DIR, cfg.DATASET.NAME, batch_size,
                                                      dataset_split='train',
                                                      num_workers=num_workers, datasource=data_source_train,
                                                      im_size=cfg.MODEL.IMAGE_DIM,
                                                      dali=cfg.TRAIN.NVIDIA_DALI)


    data_container_val = Data_iterator_container(cfg.DATASET.ROOT_DATA_DIR, cfg.DATASET.NAME, batch_size,
                                                      dataset_split='validation',
                                                      num_workers=num_workers, datasource=data_source_val,
                                                      im_size=cfg.MODEL.IMAGE_DIM,
                                                      dali=cfg.TRAIN.NVIDIA_DALI)

    data_container_test = Data_iterator_container(cfg.DATASET.ROOT_DATA_DIR, cfg.DATASET.NAME, batch_size,
                                                      dataset_split='test',
                                                      num_workers=num_workers, datasource=data_source_test,
                                                      im_size=cfg.MODEL.IMAGE_DIM,
                                                      dali=cfg.TRAIN.NVIDIA_DALI)

    if cfg.DEBUG:
        cfg.TRAIN.SNAPSHOT_INTERVAL = 1
        cfg.SAVE.LOG_BATCH_INTERVAL = 5


    full_training(trainer, data_container_train, data_container_val, data_container_test)




if __name__ == "__main__":
    print("CUDA FOUND?")
    print(torch.cuda.is_available())

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATASET.ROOT_DATA_DIR = args.data_dir

    run_main(args)
    print("end of main")

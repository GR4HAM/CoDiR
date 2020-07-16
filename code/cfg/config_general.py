import os

import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DEBUG = False
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.PHASE = 'train' #train or eval
__C.SEED = None


# SAVE options
__C.SAVE = edict()

__C.SAVE.SAVE_MODEL = True #True if you want to save best model
__C.SAVE.LOG_PARAMS = True #True if you want to log parameters with tensorboard
__C.SAVE.LOG_BATCH_INTERVAL = 50 #log every x batches
__C.SAVE.SAVED_MODEL_LOC = '' #provide model location if you want to start from pre-trained model


# DATASET options
__C.DATASET = edict()
__C.DATASET.NAME = 'COCO'
__C.DATASET.ROOT_DATA_DIR = '~/Data/'

# Training options
__C.TRAIN = edict()

__C.TRAIN.METHOD = 'fisher' #use fisher method for IPM formulation
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.STEP_EVERY_X_BATCHES = 1 # if 1, optimizer step every batch; if 2, optimizer step every 2 batches etc
__C.TRAIN.MAX_EPOCH = 50 #stop training after this amount of epochs
__C.TRAIN.SNAPSHOT_INTERVAL_START = 1 #start saving parameters after this interval_start
__C.TRAIN.SNAPSHOT_INTERVAL = 5 #save/evaluate parameters after each x epochs
__C.TRAIN.DISCRIMINATOR_LR = 5.e-3 #critic learning rate
__C.TRAIN.RHO = 1.e-6 #weight for penalty term for FISHER method
__C.TRAIN.NVIDIA_DALI = True #set this to false if you want to use the traditional pytorch loader
__C.TRAIN.IMAGE_AREA_FACTOR_LOWER_BOUND = 0.8 #default is to use are between 0.8 and 1.0 of original area for random resized crop


# Training options
__C.MODEL = edict()
__C.MODEL.MODEL_TYPE = 'resnet18' #'resnet18', 'resnet101', 'inception_v3'
__C.MODEL.OUT_METHOD = 'cos_sim' #cosine similarity to check check class membership
__C.MODEL.FEATURE_TYPE = 'outputs' #default: 'outputs'
__C.MODEL.IMAGE_DIM = 224
__C.MODEL.VOCAB_SIZE = edict()
__C.MODEL.VOCAB_SIZE.CLASS = 91 #size of vocab for classes (n_c)
__C.MODEL.VOCAB_SIZE.ENV = 300 #size of vocab for environments (n_l)
__C.MODEL.FEATURE_DIM = 512 #only necessary if using own linear model

__C.MODEL.AMT_OF_THRESHES = 100 #amount of thresholds to evaluate during training
__C.MODEL.THRESH_START = 0.0 #minimum threshold to evaluate
__C.MODEL.THRESH_END = 1.0 #max treshold to evaluate

#Class options
__C.MODEL.CLASS = edict()
__C.MODEL.CLASS.DIM = 91 #how many rows in the CoDiR rep -> = MODEL.VOCAB_SIZE.CLASS
__C.MODEL.CLASS.LABEL_ORIGIN = 'image' #'image'=class or 'sentence'=capt
__C.MODEL.CLASS.TYPE_OF_CLASS = 'instances' #instances or all_in_vocab; method to select labels in coco.py
__C.MODEL.CLASS.BY_TAG_TYPE = False #filter by type of tag (ie only nouns, adjectives, verbs

#Environment options
__C.MODEL.ENVIRONMENT = edict()
__C.MODEL.ENVIRONMENT.DIM = 300 #n_e, amount of environments
__C.MODEL.ENVIRONMENT.LABEL_ORIGIN = 'sentence' #'image' or 'sentence'
__C.MODEL.ENVIRONMENT.TYPE_OF_CLASS = 'all_in_vocab' #instances or all_in_vocab
__C.MODEL.ENVIRONMENT.BY_TAG_TYPE = True #filter by type of tag
__C.MODEL.ENVIRONMENT.R = 40 #R, max amt of positive features
__C.MODEL.ENVIRONMENT.INDIV_WEIGHT = False #how are masks applied for backprop, ie 'weighted' vs 'non-weighted'



def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #for k, v in a.iteritems():
    for k,v in a.items():
        # a must specify keys that are in b
        #if not b.has_key(k):
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(os.path.join(os.getcwd(),filename), 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def merge_first_cfg_into_second(a,b):
    import yaml

    with open(os.path.join(os.getcwd(), a), 'r') as f:
        a_cfg = edict(yaml.load(f))

    with open(os.path.join(os.getcwd(), b), 'r') as f:
        b_cfg = edict(yaml.load(f))


    _merge_a_into_b(b_cfg, __C)
    _merge_a_into_b(a_cfg, __C)

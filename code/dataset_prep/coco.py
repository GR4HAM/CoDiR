import torch.utils.data as data
from cfg.config_general import cfg
from PIL import Image
import os
import os.path
import errno
import pickle
import string
import torch
import nltk
from nltk.corpus import stopwords
import numpy as np
from pycocotools.coco import COCO
from random import shuffle

def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def preprocess_sentence(sentence):
    #lowercase and remove punctuation
    return sentence.lower().translate(str.maketrans('', '', string.punctuation)).split()



def create_vocab(file_location, coco_one, coco_two, vocab_options):
    if vocab_options["label_origin"] == "image":
        create_vocab_from_image(file_location, coco_one, vocab_options["size"])
    elif vocab_options["label_origin"] == "sentence":
        create_vocab_from_sentence(file_location, coco_one, coco_two, vocab_options["size"])


def create_vocab_from_sentence(file_location, coco_one, coco_two, vocab_size):
    # create a vocab for sentence descriptions

    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

    stop_words = stopwords.words('english')
    vocab_count_dict = {}

    tag_types = {'ADJ', 'NOUN','VERB'}

    all_sentences = []
    print("tokenizing sentences")
    for coco in [coco_one, coco_two]:
        keylist = list(coco.imgs.keys())
        for progress_counter, img_id in enumerate(keylist):
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                all_sentences += preprocess_sentence(ann["caption"])

    print("tagging sentences... (this might take a while)")
    tagged_sents = nltk.pos_tag(all_sentences, tagset='universal')
    print("removing pronouns, stopwords, ...")
    for w, tag in tagged_sents:
        if tag in tag_types and not w in stop_words:
            if not w in vocab_count_dict:
                    vocab_count_dict[w] = 1
            else:
                vocab_count_dict[w] += 1
    vocab = {"word2idx":{}}
    vocab["word2idx"]["_MASKED_"] = 0
    vocab["word2idx"]["_EOS_"] = 1
    vocab["word2idx"]["_OOV_"] = 2
    vocab_index = 3

    sorted_vocab = sorted(vocab_count_dict.items(), key=lambda kv: kv[1], reverse=True)
    for tu in sorted_vocab:
        if vocab_index<vocab_size:
            vocab["word2idx"][tu[0]] = vocab_index
            vocab_index+=1

    vocab["size"] = vocab_index
    with open(file_location, 'wb') as save_file:
        pickle.dump(vocab, save_file)

def create_vocab_from_image(file_location, coco_inst, vocab_size):
    #create a vocab for image class labels

    all_inst_cats = coco_inst.loadCats(coco_inst.getCatIds())
    vocab = {"word2idx": {}}
    vocab_index = 0
    #first add supercategories, then categories
    for cat in all_inst_cats:
        if vocab_index == vocab_size:
            break
        if not cat['supercategory'] in vocab["word2idx"]:
            vocab["word2idx"][cat['supercategory']] = vocab_index
            vocab_index += 1
    for cat in all_inst_cats:
        if vocab_index == vocab_size:
            break
        if not cat['name'] in vocab["word2idx"]:
            vocab["word2idx"][cat['name']] = vocab_index
            vocab_index += 1

    vocab["size"] = vocab_index
    with open(file_location, 'wb') as save_file:
        pickle.dump(vocab, save_file)


def get_rep_for_target(target_in, vocab, type_of_class='instances', instances=None):
    #for both class labels or sentence descriptions, return a target representation indicating the presence of a label
    #for a particular image


    if type_of_class in {'instances','super_categories'}: #use all labels including supercategories
        #take image class labels
        target = torch.zeros(vocab["size"], dtype=torch.float, requires_grad=False)
        for ins in instances:
            target[vocab['word2idx'][ins]] += 1.0


    elif type_of_class=='all_in_vocab':
        # any word in the vocab is included, taken from sentence descriptions
        target = torch.zeros(vocab["size"], dtype=torch.float, requires_grad=False)
        for sentence in target_in:
            for w in preprocess_sentence(sentence):
                if w in vocab["word2idx"]:
                    target[vocab["word2idx"][w]] += 1.0
        if torch.sum(target) == 0.:
            target[vocab["word2idx"]["_OOV_"]] = 1. #add dummy value if no words are found
        else:
            target = target / torch.sum(target)

    else:
        raise ValueError("Unknown type of class in get_rep_for_target: "+str(type_of_class))

    return target

def create_target_by_type( coco, vocab, vocab_options, indices_list, image_id_list):
    if vocab_options["label_origin"] == "image":
        #retrieve targets from image classes
        target_dict = create_reps_for_image_instances(coco,vocab, indices_list, image_id_list)
    elif vocab_options["label_origin"] == "sentence":
        #retrieve targets from most frequence words in sentence
        target_dict = create_sentence_reps( coco, vocab, vocab_options, indices_list, image_id_list)
    else:
        raise ValueError("Unknown label_origin in create_target_by_type: "+str(vocab_options["label_origin"]))
    return target_dict

def create_targets(targets_file_loc, coco_cap, coco_env, class_vocab, env_vocab, class_vocab_options,  env_vocab_options,
                   indices_list, image_id_list):

    class_targets = create_target_by_type( coco_cap, class_vocab, class_vocab_options,
                                          indices_list, image_id_list)
    env_targets = create_target_by_type( coco_env, env_vocab, env_vocab_options,
                                        indices_list, image_id_list)

    all_targets = {}

    assert class_targets["id"] == env_targets["id"]
    all_targets["id"] = class_targets["id"]
    all_targets["class"] = {}
    all_targets["env"] = {}

    #join targets by ids
    for id in class_targets["id"]:
        all_targets["class"][id] = class_targets["target"][id]
        all_targets["env"][id] = env_targets["target"][id]

    with open(targets_file_loc, 'wb') as save_file:
        pickle.dump(all_targets, save_file)




def create_sentence_reps( coco_inst, vocab, vocab_options, indices_list, image_id_list):
    #create representation for words from sentences, return a dictionary containing all samples

    print("Creating target list of type " + str(vocab_options['type_of_class']))

    nbow_file = []
    target_dict={}
    target_dict["target"] = {}
    ids = []

    for idx in indices_list:
        #get annotations for each index
        img_id = image_id_list[idx]#temp_ids[idx]
        ann_ids = coco_inst.getAnnIds(imgIds=img_id)
        anns = coco_inst.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        target = get_rep_for_target(target, vocab, type_of_class=vocab_options['type_of_class'])
        nbow_file.append(target)
        ids.append(idx)
        target_dict["target"][idx] = target

    target_dict["id"] = ids

    return target_dict


def create_file_list(image_dir, img_list_fn, idx_numpy_list_fn, img_id_list_fn,  coco_cap):
    #Method to save a list of all indices, image ids and filenames
    assert os.path.exists(image_dir)

    # get list of ids from coco
    temp_ids = list(coco_cap.imgs.keys())
    len_ids = len(temp_ids)

    idx_list = []
    img_id_list = []

    with open(img_list_fn, "w") as text_file:
        for idx in range(len_ids):

            img_id = temp_ids[idx]

            img_name = coco_cap.imgs[img_id]['file_name']
            text_file.write(img_name+"\n")
            idx_list.append(idx)
            img_id_list.append(img_id)

    np.save(idx_numpy_list_fn, np.array(idx_list))
    np.save(img_id_list_fn, np.array(img_id_list))


def create_reps_for_image_instances(coco_inst, vocab, indices_list,  image_id_list,
                        type_of_class='instances'):

    #type_of_class can be 'instances' or 'segmentation'
    assert type_of_class in {'instances', 'segmentation'}
    print("Creating target list of type "+str(type_of_class))

    target_dict={}
    target_dict["target"] = {}
    ids = []

    for idx in indices_list:

        # get annotations for each index
        img_id = image_id_list[idx]
        anns_inst = coco_inst.loadAnns(coco_inst.getAnnIds(imgIds=img_id))

        instances = []
        for ann_c in anns_inst:
            instances.append(coco_inst.cats[ann_c['category_id']]['name'])
            instances.append(coco_inst.cats[ann_c['category_id']]['supercategory'])
        target = get_rep_for_target(None, vocab, type_of_class=type_of_class, instances=instances)

        target_dict["target"][idx] = target
        ids.append(idx)

    target_dict["id"] = ids
    return target_dict





class CocoCaptions(data.Dataset):
    """
    Returns images and labels for coco captions dataset:`
    MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Requires pycocotools

    """

    training_file = 'train2014'
    validation_file = 'val2014'
    test_file = 'test2014'
    annotations_file_train_and_val = 'annotations'

    def __init__(self, root, split="train", transforms={},
                     distortions=None,
                 class_vocab_options={"size": 91, "label_origin": "image", "by_tag_type": False, "type_of_class":None},
                 env_vocab_options={"size": 300, "label_origin": "sentence", "by_tag_type": True, "type_of_class":"all_in_vocab"},
                 batch_size=128):
        '''
        root: root where data can be found; vocab, list and target files will be saved here
        transforms: dict containing image transforms

        {type}_vocab_options:
            "size": amount of entries in vocab, default = 91/300
            "label_origins": str for label origin
                "image": only use image class labels for vocab
                "sentence": use most frequent words in image descriptions
            "by_tag_type": if True, prefilter vocab by tag type (ie only nouns, adjectives and verbs)
            "type_of_class": relevant for "sentence" label_origin -
                            "all_in_vocab" > use any word in the current vocab
                            "instances"/"super_categories" -> use class labels from images including supercateogories
        '''


        self.root = os.path.expanduser(root)


        self.transforms = transforms
        self.split = split #boolean training set or test set
        self.distortions = distortions
        self.annFile = {}

        self.batch_size = batch_size #necessary for dali loader
        self.im_size = cfg.MODEL.IMAGE_DIM

        self.class_vocab_file = "coco_vocab_origin_"+str(class_vocab_options["label_origin"])+"_size_"+str(
            class_vocab_options["size"])+".pickle"
        self.env_vocab_file = "coco_vocab_origin_" + str(env_vocab_options["label_origin"]) + "_size_" + str(
            env_vocab_options["size"]) + ".pickle"
        self.targets_file = "coco_targets_"+str(split)+"_origin_"+str(class_vocab_options["label_origin"])+"_and_"+str(
            env_vocab_options["label_origin"]) +"_classsize_"+str(class_vocab_options["size"])+"_envsize_"+str(
            env_vocab_options["size"])+".pickle"

        if self.split=="train":
            self.img_root = os.path.join(self.root,"train2014")
            self.shuffled=True
            self.list_split = "train"
        else: #both for "val" and "test" use this folder
            self.img_root = os.path.join(self.root, "val2014")
            self.shuffled=False
            self.list_split = "validation"
        if not cfg.TRAIN.NVIDIA_DALI:
            self.class_coco = self.get_relevant_coco("class", class_vocab_options, split)
            self.env_coco = self.get_relevant_coco("env", env_vocab_options, split)
        else:
            self.class_coco = None
            self.env_coco = None

        index_list_fn = "coco_index_list_" + self.list_split + ".npy"

        self.index_list_loc = os.path.join(self.root, index_list_fn)
        img_list_fn = "coco_image_list_" + self.list_split + ".txt"

        self.image_list_loc = os.path.join(self.root, img_list_fn)
        image_ID_list_fn = "coco_img_id_list_" + self.list_split +  ".npy"
        self.image_ID_list_loc = os.path.join(self.root, image_ID_list_fn)


        if not self._check_exists():
            raise RuntimeError('Dataset not found...' )

        if not self._check_if_list_files_exist():
            self.create_list_files()
        self.images, self.img_ids = self.load_list_files()
        self.indices = self.load_idxs()

        if not self.__check_vocab_exists("class") or not self.__check_vocab_exists("env"):
            if split == "train":
                other_split = "validation"
            else:
                other_split = "train"


            if not self.__check_vocab_exists("class"):
                if self.class_coco is None:
                    self.class_coco = self.get_relevant_coco("class", class_vocab_options, split)
                    self.env_coco = self.get_relevant_coco("env", env_vocab_options, split)
                other_coco_class = self.get_relevant_coco("class", class_vocab_options, other_split)
                create_vocab(os.path.join(self.processed_folder, self.class_vocab_file), \
                         self.class_coco,  other_coco_class, class_vocab_options)
            if not self.__check_vocab_exists("env"):
                if self.class_coco is None:
                    self.class_coco = self.get_relevant_coco("class", class_vocab_options, split)
                    self.env_coco = self.get_relevant_coco("env", env_vocab_options, split)
                other_coco_env = self.get_relevant_coco("env", env_vocab_options, other_split)
                create_vocab(os.path.join(self.processed_folder, self.env_vocab_file), \
                         self.env_coco,  other_coco_env, env_vocab_options)

        self.class_vocab = self.load_vocabulary("class")
        self.env_vocab = self.load_vocabulary("env")

        if not self.__check_targets_exists():
            if self.class_coco is None:
                self.class_coco = self.get_relevant_coco("class", class_vocab_options, split)
                self.env_coco = self.get_relevant_coco("env", env_vocab_options, split)
            create_targets(os.path.join(self.processed_folder, self.targets_file),
                           self.class_coco, self.env_coco,
                           self.class_vocab, self.env_vocab,
                           class_vocab_options, env_vocab_options,
                           self.indices, self.img_ids)
        self.targets_dict = self.load_targets()

        self.epoch_length = len(self.targets_dict["id"])

    def get_relevant_coco(self, type, vocab_options, split):
        ## type: "class" or "env"
        if vocab_options["label_origin"] in {'sentence'}:
            if split == "train":
                self.annFile[type] = "annotations/captions_train2014.json"
            else:
                self.annFile[type] = "annotations/captions_val2014.json"
        elif vocab_options["label_origin"] in {'image'}:
            if split == "train":
                self.annFile[type] = "annotations/instances_train2014.json"
            else:
                self.annFile[type] = "annotations/instances_val2014.json"

        return COCO(os.path.join(self.processed_folder, self.annFile[type]))

    def create_list_files(self):
        print("creating file list; dataset=" + str(cfg.DATASET.NAME) + "; dataset_split=" + str(self.split))
        assert os.path.exists(self.root)


        if self.split == 'train':
            coco_cap = COCO(os.path.join(self.root,
                                         "annotations/captions_train2014.json"))
        else:
            coco_cap = COCO(os.path.join(self.root,
                                         "annotations/captions_val2014.json"))

        create_file_list(self.root, self.image_list_loc,
                         self.index_list_loc, self.image_ID_list_loc, coco_cap)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns:input and output dictionaries.
        """
        input = {}
        target = {}

        coco = self.class_coco
        index_own = self.indices[index]
        img_id = self.img_ids[index_own]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        input["tensor"] = img
        input["img_id"] = img_id
        target["id"] = index_own
        target["class"] = {}
        target["environment"] = {}
        target["class"]["tensor"] = (self.targets_dict["class"][index_own]>0).float()
        target["environment"]["tensor"] = (self.targets_dict["env"][index_own]>0).float()

        return input, target

    def get_next_batch(self):
        """
        Method used by dali pipeline loader
        Args: /
        Returns:input and output dictionaries.
        """

        input_img = []
        input_target_class = []
        input_target_env = []
        cur_idxs = []
        cur_img_ids = []

        for _ in range(self.batch_size):
            # current index
            i_c = self.get_index()

            # keep track of indices for this batch
            cur_idxs.append(np.array([i_c], dtype=np.int32))
            # pass image names
            cur_img_ids.append(np.array(self.img_ids[i_c], dtype=np.int32))
            #get img
            img_name = self.images[i_c]
            img_path = os.path.join(self.img_root, img_name)
            with open(img_path, 'rb') as f:
                cur_img_input = np.frombuffer(f.read(), dtype=np.uint8)

            input_img.append(cur_img_input)

            #get targets
            class_target = (self.targets_dict["class"][i_c]>0).float().numpy()
            input_target_class.append(class_target)
            input_target_env.append((self.targets_dict["env"][i_c]>0).float().numpy())


        return input_img, input_target_class, input_target_env, cur_idxs, cur_img_ids

    def get_index(self):
        #used for dali pipeline loader
        if len(self.indices) == 0:
            self.indices = self.load_idxs()
            if self.shuffled:
                shuffle(self.indices)
        return self.indices.pop()

    def load_idxs(self):
        indices = np.load(self.index_list_loc).tolist()

        if self.split == 'validation':
            indices = indices[:20252]
        elif self.split == 'test':
            indices = indices[20252:]

        if self.shuffled:
            shuffle(indices)
        return indices

    def load_list_files(self):
        imgs = []
        with open(self.image_list_loc, "r") as f_i:
            for line in f_i:
                if line is not '':
                    imgs.append(line.rstrip())

        img_ids = np.load(self.image_ID_list_loc).tolist()

        return imgs, img_ids


    def __len__(self):
        return len(self.targets_dict["id"])

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self, individual_file='all'):
        if individual_file == 'all':
            return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.validation_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.test_file)) and \
               os.path.exists(os.path.join(self.processed_folder, self.annotations_file_train_and_val))
        elif individual_file == 'train':
            return os.path.exists(os.path.join(self.processed_folder, self.training_file))
        elif individual_file == 'validation':
            return os.path.exists(os.path.join(self.processed_folder, self.validation_file))
        elif individual_file == 'test':
            return os.path.exists(os.path.join(self.processed_folder, self.test_file))
        elif individual_file == 'annotation':
            return os.path.exists(os.path.join(self.processed_folder, self.annotations_file_train_and_val))
        else:
            raise ValueError("not understood which individual file is requested")

    def __check_vocab_exists(self, type="class"):
        if type == "class":
            return os.path.exists(os.path.join(self.processed_folder, self.class_vocab_file))
        elif type =="env":
            return os.path.exists(os.path.join(self.processed_folder, self.env_vocab_file))
        else:
            raise ValueError("Tried to check unknown vocabulary type: " + str(type))

    def __check_targets_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.targets_file))

    def __check_nbow_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.nbow_file))

    def _check_if_list_files_exist(self):

        return os.path.exists(self.index_list_loc) and \
               os.path.exists(self.image_ID_list_loc) and \
               os.path.exists(self.image_list_loc)

    def load_vocabulary(self, type="class"):
        if type == "class":
            with open(os.path.join(self.processed_folder, self.class_vocab_file), "rb") as in_f:
                vocab = pickle.load(in_f)
        elif type == "env":
            with open(os.path.join(self.processed_folder, self.env_vocab_file), "rb") as in_f:
                vocab = pickle.load(in_f)
        else:
            raise ValueError("Tried to load unknown vocabulary type: "+str(type))
        return vocab

    def load_targets(self):
        with open(os.path.join(self.processed_folder, self.targets_file), "rb") as in_f:
            targets = pickle.load(in_f)
        return targets








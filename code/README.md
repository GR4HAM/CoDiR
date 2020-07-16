##CoDiR

Pytorch code to train CoDiR representations on the COCO dataset.


##Installation 

Run the python code locally with the following packages installed. NVIDIA Dali is optional but speeds up training significantly.

Requirements:
- NVIDIA Dali 0.12.0
- pycocotools 2.0+
- TensorboardX 1.6+
- easydict
- scikit
- torchvision 0.2.2+
- tqdm
- torch 1.0.1+

##To get started

In the cfg folder, configuration files are given to train different types of models. The preferred cfg file can be entered at the top of 'main.py' or enterred as an argument on the command line, e.g.: 
 
> python main.py --cfg '../cfg/COCO_R18_CLASS.yml'

After training completes, the model with best performance on the dataset is evaluated on the test set.

Here is an overview of the available config files:
- COCO_R18_CLASS.yml: Train a ResNet-18 CoDiR model with environments made from class labels. n_l = 91, n_e=300, R=40
- COCO_R101_CLASS.yml: Train a ResNet-101 CoDiR model with environments made from class labels. n_l = 91, n_e=300, R=40
- COCO_IV3_CLASS.yml: Train a Inception-v3 CoDiR model with environments made from class labels. n_l = 91, n_e=300, R=40
- COCO_R18_CAPT_NL_300.yml: Train a ResNet-18 CoDiR model with environments made from caption words. n_l = 300, n_e=300, R=40
- COCO_R101_CAPT_NL_300.yml: Train a ResNet-101 CoDiR model with environments made from caption words. n_l = 300, n_e=300, R=40
- COCO_IV3_CAPT_NL_300.yml: Train a Inception-v3 CoDiR model with environments made from caption words. n_l = 300, n_e=300, R=40
- COCO_R18_CAPT_NL_1000.yml: Train a ResNet-18 CoDiR model with environments made from caption words. n_l = 1000, n_e=1000, R=100
- COCO_R101_CAPT_NL_1000.yml: Train a ResNet-101 CoDiR model with environments made from caption words. n_l = 1000, n_e=1000, R=100
- COCO_IV3_CAPT_NL_1000.yml: Train a Inception-v3 CoDiR model with environments made from caption words. n_l = 1000, n_e=1000, R=100

To check the meaning of different parameters in the config file, consult cfg/config_general.py

##Dataset

The dataset is COCO that requires the pycocotools 2.0+ to be installed. We also provide the indices and image names in the train/validation/test set, as well as well as the vocab files required for the experiments. Place these in the CocoCaptions data folder when running the code. 
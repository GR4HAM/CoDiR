import sys
import torch
import torch.optim as optim
from tqdm import tqdm
import os
from utils import mkdir_p, save_data_results, weights_init
from cfg.config_general import cfg
import numpy as np
from tensorboard_logger import Tensorboard_logger


class Trainer(object):
    def __init__(self, output_dir, output_dir_run):

        self.is_pycharm_hosted = "PYCHARM_HOSTED" in os.environ
        self.output_dir = output_dir

        if cfg.SAVE.LOG_PARAMS:
            #create log dir
            self.log_dir = os.path.join(output_dir,str(cfg.DATASET.NAME),str(cfg.CONFIG_NAME),"DEBUG_"+str(cfg.DEBUG),"PHASE_"+str(cfg.PHASE),
                                        output_dir_run)
            mkdir_p(self.log_dir)

            #initialize tensorboardX writer
            self.logger = Tensorboard_logger(self.log_dir)

    def load_networks(self, amt_of_train_batches, load_from=''):
        #loads a pretrained or saved network
        if cfg.CONFIG_NAME in {'COCO'}:

            input_dim = cfg.MODEL.IMAGE_DIM**2
            feature_dim = cfg.MODEL.FEATURE_DIM
            n_c = cfg.MODEL.CLASS.DIM
            n_l = cfg.MODEL.VOCAB_SIZE.ENV
            n_e = cfg.MODEL.ENVIRONMENT.DIM
            R = cfg.MODEL.ENVIRONMENT.R

            if cfg.MODEL.OUT_METHOD == 'cos_sim':
                from models.CoDiR_image import CoDiR_image
                disc = CoDiR_image(input_dim, feature_dim, n_c, n_l, n_e, R, amt_of_train_batches)

            disc.apply(weights_init)

            self.model = disc

            if load_from != '':
                self.load_best_model(load_from=load_from)
                print('******++++++++**********')
                print('Load from: ', load_from)
                print('******++++++++*********')

            if cfg.CUDA:
                self.model = self.model.cuda()
            else:
                self.model = self.model.cpu()


    def prepare_for_training(self):
        #essentially creates optimizer
        print(" ...  preparing for training ...")

        assert self.model is not None
        disc_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("trainable params disc: " + str(disc_trainable_params))

        self.discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        disc_params = []
        for p in self.model.parameters():
            if p.requires_grad:
                disc_params.append(p)

        self.optimizer = optim.Adam(disc_params,
                                     lr=self.discriminator_lr,
                                     betas=(0.5, 0.999))

    def close(self):
        self.logger.close()


    def process_batch_inputs(self, batch):
        #deal with the batch depending on pytorch loader or nvidia dali loader was used

        if not cfg.TRAIN.NVIDIA_DALI:
            #regular pytorch loader
            inputs, targets = batch
            if cfg.CUDA:
                inputs["tensor"] = inputs["tensor"].cuda()
                if "tensor" in targets["class"]:
                    targets["class"]["tensor"] = targets["class"]["tensor"].cuda()
                else:
                    targets["class"]["integer"] = targets["class"]["integer"].cuda()
                if "tensor" in targets["environment"]:
                    targets["environment"]["tensor"] = targets["environment"]["tensor"].cuda()
                else:
                    targets["environment"]["integer"] = targets["environment"]["integer"].cuda()
        else:
            # nvidia dali loader
            inputs = {}
            targets = {}
            inputs["tensor"] = batch[0]['input_img']
            inputs["img_id"] = batch[0]['img_ids']
            targets["id"] = batch[0]['indices']
            targets["class"] = {}
            targets["environment"] = {}
            targets["class"]["tensor"] = batch[0]['target_class']
            targets["environment"]["tensor"] = batch[0]['target_env']


        return inputs, targets

    def train_epoch(self, epoch, iterator):
        #training through a full epoch
        self.model.train()

        iter_end = iterator.epoch_length
        batch_idx = 0
        reset_optimizer = True
        train_metrics = {"train_loss":0.0}
        global_batch_counter = epoch*iterator.epoch_length
        mone_tensor = torch.tensor(-1.0)
        if cfg.CUDA:
            mone_tensor = mone_tensor.cuda()

        if self.is_pycharm_hosted:
            batch_generator = tqdm(iterator.loader,
                                             total=iterator.epoch_length, file=sys.stdout)
        else:
            batch_generator = iterator.loader

        for batch in batch_generator:
            if batch_idx >= iter_end:
                break

            if reset_optimizer:
                self.model.zero_grad()
                self.optimizer.zero_grad()
                reset_optimizer = False

            inputs, targets = self.process_batch_inputs(batch)

            #feed through model
            model_out = self.model(inputs["tensor"], targets, batch_index=batch_idx, mode="train")
            model_out["loss"].backward(mone_tensor)
            train_metrics["train_loss"]-= model_out["loss"].item()

            if batch_idx % cfg.TRAIN.STEP_EVERY_X_BATCHES == 0:
                self.optimizer.step()
                if cfg.MODEL.OUT_METHOD in {'cos_sim'}:
                    # for FISHER GAN
                    self.model.Lambda.data += self.model.rho * self.model.Lambda.grad.data
                    self.model.Lambda.grad.data.zero_()
                reset_optimizer = True

            if batch_idx % cfg.SAVE.LOG_BATCH_INTERVAL == 0:
                #save metrics
                model_metrics = self.model.get_metrics( reset=False)
                self.logger.save_metrics(train_metrics, type_of_metric="intra_epoch", phase="train", batch_counter=global_batch_counter)
                self.logger.save_metrics(model_metrics, type_of_metric="intra_epoch", phase="train", batch_counter=global_batch_counter)

                if self.is_pycharm_hosted:
                    batch_generator.set_description('epoch %d train_loss=%g f1_score=%g *********' % (
                                                epoch, train_metrics["train_loss"], model_metrics["f1_score"]))
                    batch_generator.update(1)

            batch_idx += 1
            global_batch_counter += 1

        if self.is_pycharm_hosted:
            batch_generator.close()

        # save metrics
        self.logger.save_metrics(train_metrics, type_of_metric="post_epoch", phase="train", batch_counter=epoch)
        self.logger.save_metrics(self.model.get_metrics(reset=True), type_of_metric="post_epoch", phase="train", batch_counter=epoch)

    def calc_templates(self, iterator):
        #run through the training set whilst calculating templates and setting thresholds
        self.model.eval()

        iter_end = iterator.epoch_length
        batch_idx = 0

        if self.is_pycharm_hosted:
            batch_generator = tqdm(iterator.loader,
                                   total=iterator.epoch_length, file=sys.stdout)
        else:
            batch_generator = iterator.loader

        for batch in batch_generator:
            if batch_idx >= iter_end:
                break

            inputs, targets = self.process_batch_inputs(batch)

            #feed through model in "templates" mode
            model_out = self.model(inputs["tensor"], targets, batch_index=batch_idx, mode="templates")

            batch_idx += 1

        if self.is_pycharm_hosted:
            batch_generator.close()


    def evaluate_epoch(self, epoch, iterator, phase="val"):
        #evaluate the model on the valuation or test set
        self.model.eval()

        iter_end = iterator.epoch_length
        batch_idx = 0
        eval_metrics = {}
        global_batch_counter = epoch*iterator.epoch_length

        if self.is_pycharm_hosted:
            batch_generator = tqdm(iterator.loader,
                                             total=iterator.epoch_length, file=sys.stdout)
        else:
            batch_generator = iterator.loader

        for batch in batch_generator:
            if batch_idx >= iter_end:
                break

            inputs, targets = self.process_batch_inputs(batch)

            #feed through model in "eval" mode
            model_out = self.model(inputs["tensor"], targets, batch_index=batch_idx, mode="eval")


            if batch_idx % cfg.SAVE.LOG_BATCH_INTERVAL == 0:
                #save metrics
                model_metrics = self.model.get_metrics( reset=False)
                self.logger.save_metrics(eval_metrics, type_of_metric="intra_epoch", phase=phase, batch_counter=global_batch_counter)
                self.logger.save_metrics(model_metrics, type_of_metric="intra_epoch", phase=phase, batch_counter=global_batch_counter)

                if self.is_pycharm_hosted:
                    batch_generator.set_description('epoch %d eval on %s set: f1_score=%g *********' % (
                                                epoch, phase, model_metrics["f1_score"]))
                    batch_generator.update(1)


            batch_idx += 1
            global_batch_counter += 1

        if self.is_pycharm_hosted:
            batch_generator.close()
        else:
            print('epoch %d eval on %s set: f1_score=%g *********' % (
                                                epoch, phase, model_metrics["f1_score"]))

        # save metrics
        self.logger.save_metrics(eval_metrics, type_of_metric="post_epoch", phase=phase, batch_counter=epoch)
        self.logger.save_metrics(self.model.get_metrics(reset=True), type_of_metric="post_epoch", phase=phase, batch_counter=epoch)
        if hasattr(self.model, 'cur_thresh'):
            self.logger.save_histogram(self.model.cur_thresh, "thresholds", epoch)

        return model_metrics["f1_score"], model_metrics["precision"], model_metrics["recall"]

    def get_features_for_iterator(self, iterator, type="tensor_dict"):
        #do a run through the training set whilst obtaining and saving representations
        #type can be "tensor_dict" or "numpy_arr"
        #tensor_dict: return a dictionary containing different tensors
        #numpy_arr: features are returned as large numpy arrays

        self.model.eval()

        iter_end = iterator.epoch_length
        batch_idx = 0

        feature_dict = {}

        if type == "numpy_arr":
            features_arr = []
            class_targets_arr = []
            env_targets_arr = []

        if self.is_pycharm_hosted:
            batch_generator = tqdm(iterator.loader,
                                             total=iterator.epoch_length, file=sys.stdout)
        else:
            batch_generator = iterator.loader

        for batch in batch_generator:
            if batch_idx >= iter_end:
                break

            inputs, targets = self.process_batch_inputs(batch)

            model_out = self.model(inputs["tensor"], targets, batch_index=batch_idx, mode="features")

            bsize = targets["id"].shape[0]
            for i in range(bsize):
                cur_id = targets["id"][i].item()
                if not cur_id in feature_dict:
                    if type == "tensor_dict":
                        feature_dict[cur_id] = {}
                        feature_dict[cur_id]["coco_img_id"] = inputs["img_id"][i].item()
                        feature_dict[cur_id]["features"] = model_out["features"][i].cpu()
                        feature_dict[cur_id]["class"] = targets["class"]["tensor"][i].cpu()
                        feature_dict[cur_id]["environment"] = targets["environment"]["tensor"][i].cpu()
                        feature_dict[cur_id]["prediction"] = model_out["prediction"][i].cpu()
                    elif type == "numpy_arr":
                        features_arr.append(model_out["features"][i].cpu().numpy())
                        class_targets_arr.append(targets["class"]["tensor"][i].cpu().numpy())
                        env_targets_arr.append(targets["environment"]["tensor"][i].cpu().numpy())

                else:
                    raise ValueError("unexpected behavior: id already in dict of targets")

            batch_idx += 1

        if self.is_pycharm_hosted:
            batch_generator.close()

        if type == "numpy_arr":
            feature_dict["features"] = np.array(features_arr)
            feature_dict["class"] = np.array(class_targets_arr)
            feature_dict["environment"] = np.array(env_targets_arr)

        return feature_dict

    def save_model(self, results):
        #save the model and relevant tensors
        if not results is None:
            save_data_results(results, self.log_dir)
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best_model"))

        if cfg.MODEL.OUT_METHOD in {'cos_sim'}:
            torch.save(self.model.avg_E_P_f, os.path.join(self.log_dir, "best_avg_E_P_f"))
            torch.save(self.model.avg_E_Q_f, os.path.join(self.log_dir, "best_avg_E_Q_f"))
            torch.save(self.model.cur_thresh_pt, os.path.join(self.log_dir, "best_threshes"))
            torch.save(self.model.random_feature_vec, os.path.join(self.log_dir, "random_feature_vec"))



    def load_best_model(self, load_from=''):
        if load_from == '':
            # load best model from own model directory
            load_loc = os.path.join(self.log_dir, "best_model")
            load_loc_eqf = os.path.join(self.log_dir, "best_avg_E_Q_f")
            load_loc_epf = os.path.join(self.log_dir, "best_avg_E_P_f")
            load_loc_best_threshes = os.path.join(self.log_dir, "best_threshes")
            load_loc_random_feature_vec = os.path.join(self.log_dir, "random_feature_vec")

        else:
            load_loc = os.path.join(self.output_dir, load_from, "best_model")
            load_loc_eqf = os.path.join(self.output_dir, load_from,  "best_avg_E_Q_f")
            load_loc_epf = os.path.join(self.output_dir, load_from, "best_avg_E_P_f")
            load_loc_best_threshes = os.path.join(self.output_dir, load_from, "best_threshes")
            load_loc_random_feature_vec = os.path.join(self.output_dir, load_from, "random_feature_vec")


        state_dict = \
            torch.load(load_loc,
                       map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        print('******++++++++**********')
        print('Load from: ', load_loc)
        print('******++++++++*********')

        if cfg.MODEL.OUT_METHOD in {'cos_sim'}:
            print("also loading avg_E_Q_f from " + str(load_loc_eqf))
            self.model.avg_E_Q_f = torch.load(load_loc_eqf)
            self.model.avg_E_P_f = torch.load(load_loc_epf)
            self.model.cur_thresh_pt = torch.load(load_loc_best_threshes)
            self.model.random_feature_vec = torch.load(load_loc_random_feature_vec)

            if cfg.CUDA:
                self.model.avg_E_Q_f = self.model.avg_E_Q_f.cuda()
                self.model.avg_E_P_f =  self.model.avg_E_P_f.cuda()
                self.model.cur_thresh_pt = self.model.cur_thresh_pt.cuda()
                self.model.random_feature_vec[0] =  self.model.random_feature_vec[0].cuda()
            else:
                self.model.avg_E_Q_f = self.model.avg_E_Q_f.cpu()
                self.model.avg_E_P_f = self.model.avg_E_P_f.cpu()
                self.model.cur_thresh_pt = self.model.cur_thresh_pt.cpu()
                self.model.random_feature_vec[0] = self.model.random_feature_vec[0].cpu()

from abc import ABC, abstractmethod
from cfg.config_general import cfg
from utils import torch_integer_to_one_hot, tonp
import torch
import torch.nn.functional as F
import numpy as np

class CoDiR_model(ABC, torch.nn.Module):
    #Abstract class that should be implemented (for an example, see CoDiR_image.py)

    def __init__(self, input_dim, feature_dim, n_c, n_l, n_e, R, amt_of_train_batches, selection_method="thresh", rho=1.e-6):
        super(CoDiR_model, self).__init__()
        self.input_dim = input_dim #only relevant when creating own linear model
        self.feature_dim = feature_dim #only relevant when creating own linear model
        self.n_c = n_c #amount of class rows = amount of classes
        self.n_e = n_e #amount of environment columns
        self.n_l = n_l #size of environment vocab, ie mount of classes of contextual information in environments
        self.R = R #max amount of attributes per environment
        self.AMT_OF_TRAIN_BATCHES = amt_of_train_batches
        self.SELECTION_METHOD=selection_method #only thresh implemented atm

        #threshold parameters
        self.cur_thresh = 0.5 #initial thresh
        self.amt_of_threshes = cfg.MODEL.AMT_OF_THRESHES
        self.thresh_start = cfg.MODEL.THRESH_START
        self.thresh_end = cfg.MODEL.THRESH_END
        self.START_THRESHOLD_CALC = 2./3.
        self.scores_during_training = {}

        #FisherGAN param
        self.rho = rho
        self.Lambda = torch.FloatTensor([0] * self.n_c * self.n_e)
        if torch.cuda.is_available() and cfg.CUDA:
            self.Lambda = self.Lambda.cuda()
        self.Lambda = torch.autograd.Variable(self.Lambda, requires_grad=True)

        #metrics
        self.f1_sims = 0.
        self.amt_sims = 0.
        self.emd_estimate = 0.
        self.precision_sum =0.
        self.recall_sum =0.
        self.f1_sum =0.
        self.amt_of_samples=0.
        self.create_random_feature_vec()
        self.define_network()

    @abstractmethod
    def define_network(self):
        #default implementation
        self.startnet = torch.nn.Linear(self.input_dim,self.feature_dim )
        self.emd_layer = torch.nn.Linear(self.feature_dim,self.n_c*self.n_e )

    def forward_network_pass(self, input):
        features = self.startnet(input)
        out = self.emd_layer(features)
        return out

    def forward(self, input, target, batch_index=-1, mode="train"):
        #depending on the mode, perform different calculations
        if mode == "train":
            #feed through  model
            out = self.forward_network_pass(input)
            #get groundtruth in one_hot format
            if "tensor" in target["class"]:
                ground_truth_one_hot_class = target["class"]["tensor"]
            else:
                ground_truth_one_hot_class = torch_integer_to_one_hot(target["class"]["integer"], self.n_c)
            if "tensor" in target["environment"]:
                ground_truth_one_hot_env = target["environment"]["tensor"]
            else:
                ground_truth_one_hot_env = torch_integer_to_one_hot(target["environment"]["integer"], self.n_l)
            #obtain masks for backpropagation
            mask_class, mask_env = self.get_masks_for_backprop(ground_truth_one_hot_class, ground_truth_one_hot_env)
            #calculate the EMD and corresponding loss
            loss, E_P_f, E_Q_f, out_P, out_Q = self.compute_emd(out, mask_class, mask_env)

            if hasattr(self, 'avg_E_Q_f') and batch_index%cfg.SAVE.LOG_BATCH_INTERVAL == 0:
                #if available, calculate the f1 at regular intervals
                sims = self.get_similarities(logits = out)
                self.calc_f1(sims, ground_truth_one_hot_class)
            if batch_index % cfg.SAVE.LOG_BATCH_INTERVAL == 0:
                self.emd_estimate += torch.sum(E_P_f-E_Q_f).item()

            output_dict = {"loss" : loss}

        elif mode == "templates":
            # same as mode "train", but determine templates and set thresholds
            with torch.no_grad():
                out = self.forward_network_pass(input)
                if "tensor" in target["class"]:
                    ground_truth_one_hot_class = target["class"]["tensor"]
                else:
                    ground_truth_one_hot_class = torch_integer_to_one_hot(target["class"]["integer"], self.n_c)
                if "tensor" in target["environment"]:
                    ground_truth_one_hot_env = target["environment"]["tensor"]
                else:
                    ground_truth_one_hot_env = torch_integer_to_one_hot(target["environment"]["integer"], self.n_l)

                mask_class, mask_env = self.get_masks_for_backprop(ground_truth_one_hot_class, ground_truth_one_hot_env)
                loss, E_P_f, E_Q_f, out_P, out_Q = self.compute_emd(out, mask_class, mask_env)

                if batch_index <= self.START_THRESHOLD_CALC * self.AMT_OF_TRAIN_BATCHES:
                    self.update_templates(E_P_f, E_Q_f, out_P, out_Q, mask_class, mask_env, batch_index)
                else:
                    if self.SELECTION_METHOD == "thresh":
                        sims = self.get_similarities(logits=out)
                        self.add_new_threshold_estimate(sims, ground_truth_one_hot_class)
                        if batch_index >= self.AMT_OF_TRAIN_BATCHES - 2:
                            self.update_cur_threshes()
                output_dict = {}
        elif mode == "eval":
            #same as mode "train" but only evaluation, no backpropagation
            with torch.no_grad():
                out = self.forward_network_pass(input)
                if "tensor" in target["class"]:
                    ground_truth_one_hot_class = target["class"]["tensor"]
                else:
                    ground_truth_one_hot_class = torch_integer_to_one_hot(target["class"]["integer"], self.n_c)

                sims = self.get_similarities(logits=out)
                self.calc_f1(sims, ground_truth_one_hot_class)
                output_dict = {}
        elif mode == "features":
            with torch.no_grad():
                out = self.forward_network_pass(input)
                sims = self.get_similarities(logits=out)
                output_dict = {"features": out, "prediction":(sims > self.cur_thresh_pt)}

        return output_dict

    def create_random_feature_vec(self):
        #create a random feature vec that gives the composition of the environments
        import random

        #initialize
        self.random_feature_vec = []
        cur_rfv = torch.zeros((self.n_l, self.n_e),
                              dtype=torch.float32)

        #for each environment, create a random combination of labels
        for i in range(self.n_e):
            if i%50 == 0:
                print("Creating feature " + str(i + 1) + " of " + str(self.n_e))
            max_doc_size = self.R
            min_doc_size_pos = 1
            cur_rand_amt_features = random.randint(min_doc_size_pos, max_doc_size)

            if cur_rand_amt_features > 0:
                for j_p in range(cur_rand_amt_features):
                    selected_dim = random.randint(0, self.n_l - 1)
                    cur_rfv[selected_dim, i] = 1

        if cfg.CUDA:
            cur_rfv = cur_rfv.cuda()
        self.random_feature_vec.append(cur_rfv)
        print("done creating random feature vec")



    def get_similarities(self, logits):

        #create template and instance representations
        d_templates = self.avg_E_P_f - self.avg_E_Q_f
        d_instances = self.avg_E_P_f.unsqueeze(0) - logits

        #calculate similarities over rows
        sims = F.cosine_similarity(d_templates.view(1, self.n_c, self.n_e),
                                   d_instances.view(-1, self.n_c, self.n_e)
                                   , dim=2)
        return sims



    def add_new_threshold_estimate(self, similarity_estimates, ground_truth):
        #add outcomes for this batch relating to different thresholds
        #so they can later help decide the optimal threshold
        amt_of_threshes = self.amt_of_threshes
        ground_truth = ground_truth.reshape(-1, self.n_c)
        cur_TP = np.zeros((amt_of_threshes - 1, self.n_c))
        cur_FP = np.zeros((amt_of_threshes - 1, self.n_c))
        cur_FN = np.zeros((amt_of_threshes - 1, self.n_c))

        for thresh_count in range(0, amt_of_threshes - 1):

            th_s = self.thresh_start
            th_e = self.thresh_end
            th_a = amt_of_threshes
            thresh = th_s + (th_e - th_s) * (thresh_count + 1.) / float(th_a)
            predictions = (similarity_estimates > thresh).float()

            cur_TP[thresh_count, :] = torch.sum(ground_truth*predictions, dim=0).cpu().numpy()
            cur_FP[thresh_count, :] = torch.sum((1 - ground_truth) * (predictions), dim=0).cpu().numpy()
            cur_FN[thresh_count, :] = torch.sum(ground_truth * (1 - predictions), dim=0).cpu().numpy()

        self.scores_during_training["TP"] += cur_TP
        self.scores_during_training["FP"] += cur_FP
        self.scores_during_training["FN"] += cur_FN



    def update_cur_threshes(self):
        #from the observed outcomes during training, select the optimal threshold
        TP = self.scores_during_training["TP"]
        FP = self.scores_during_training["FP"]
        FN = self.scores_during_training["FN"]

        th_s = self.thresh_start
        th_e = self.thresh_end
        th_a = self.amt_of_threshes

        P_denom = np.maximum((TP + FP), 1.)  # numerical stability
        R_denom = np.maximum((TP + FN), 1.)  # numerical stability
        P = TP / P_denom +1.e-10
        R = TP / R_denom +1.e-10
        f1 = 2 / (1 / P + 1 / R)

        self.cur_thresh = th_s + (th_e - th_s) * (np.argmax(f1, \
                                                            axis=0) + 1.) / float(th_a)
        self.cur_thresh_pt = (torch.from_numpy(self.cur_thresh)).float()
        if cfg.CUDA:
            self.cur_thresh_pt = self.cur_thresh_pt.cuda()



    def get_masks_for_backprop(self, ground_truth_class, ground_truth_env):
        # Expand ground_truth_class to create a mask mod_two_c with shape [batch_size , n_c , n_e ],
        # such that mod_two_c[k,i,:] = 1 if the k-th sample, s_k , belongs to class i,
        # 0 otherwise.

        mod_two_c = ground_truth_class.clone()
        mod_two_c = mod_two_c.unsqueeze(2).expand(-1, -1, self.n_e).contiguous().view(
            ground_truth_class.shape[0] , self.n_c * self.n_e)

        rfv = self.random_feature_vec[0]

        # Multiply ground_truth_env and rfv, then expand the result, to create a mask mod_two_env
        # with shape [batch_size , n c , n e ], such that mod_two_env[k,:,j] = a where a is
        # the sum of all the labels of the k-th sample that are present in environment j.

        # ground truth shape: bs , n_c_attr
        # rfv shape: 1, n_c_attr , n_e
        m2e = torch.matmul(ground_truth_env, rfv)
        # m2e shape: #bs , n_e
        mod_two_env = m2e.unsqueeze(1).expand(-1, self.n_c,-1).contiguous().view(-1,self.n_c*self.n_e)

        #Note: last dims of the masks are flattened to a shape of n_c*n_e here to simplify subsequent code
        return mod_two_c, mod_two_env


    def compute_emd(self, logits, mask_class, mask_env):
        #calculate EMD and loss  for the critics
        one_tensor = torch.tensor(1)

        if cfg.CUDA:
            one_tensor = one_tensor.cuda()

        logits = logits.view(-1, self.n_c * self.n_e)
        out_P = logits * mask_env
        out_Q = logits * mask_class

        if cfg.MODEL.ENVIRONMENT.INDIV_WEIGHT:
            E_P_f = torch.sum(out_P, dim=0) / torch.max(torch.sum(mask_env, dim=0), one_tensor.float())
            E_P_f2 = torch.sum(out_P * out_P, dim=0) / torch.max(torch.sum(mask_env, dim=0),
                                                                 one_tensor.float())

            E_Q_f = torch.sum(out_Q, dim=0) / torch.max(torch.sum(mask_class, dim=0), one_tensor.float())
            E_Q_f2 = torch.sum(out_Q * out_Q, dim=0) / torch.max(torch.sum(mask_class, dim=0),
                                                                 one_tensor.float())
        else:
            if cfg.TRAIN.METHOD == 'fisher':
                E_P_f = torch.mean(out_P, dim=0)
                E_P_f2 = torch.mean(out_P * out_P, dim=0)

                E_Q_f = torch.mean(out_Q, dim=0)
                E_Q_f2 = torch.mean(out_Q * out_Q, dim=0)


        if cfg.TRAIN.METHOD == 'fisher':
            constraint = (1 - (0.5 * E_P_f2 + 0.5 * E_Q_f2))

            loss = torch.sum(
                E_P_f - E_Q_f + self.Lambda * constraint - self.rho / 2 * constraint ** 2)

        return loss, E_P_f, E_Q_f, out_P, out_Q

    def calc_f1(self, sims, ground_truth):
        #calculate the multilabel f1 score

        gt = (ground_truth>0).float()
        TP_arr = torch.sum((sims > self.cur_thresh_pt).float() * gt, dim=1)
        FP_arr = torch.sum((sims > self.cur_thresh_pt).float() * (1.-gt), dim=1)
        FN_arr = torch.sum((1.0-(sims > self.cur_thresh_pt).float()) * gt, dim=1)

        self.precision_sum += torch.sum((TP_arr/(TP_arr + FP_arr+1.e-10))).item()
        self.recall_sum += torch.sum((TP_arr / (TP_arr + FN_arr + 1.e-10))).item()
        self.f1_sum += torch.sum(2*TP_arr/ (2*TP_arr+FP_arr+FN_arr+ 1.e-10)).item()
        self.amt_of_samples += sims.shape[0]
        self.amt_sims += 1.


    def update_templates(self, E_P_f, E_Q_f, out_P, out_Q, mask_class, mask_env, batch_index):
        # update the templates with values of current batch

        one_tensor = torch.tensor(1)
        if cfg.CUDA:
            one_tensor = one_tensor.cuda()

        if batch_index == 0:
            if cfg.MODEL.ENVIRONMENT.INDIV_WEIGHT:
                self.sum_E_Q_f = torch.sum(out_Q, dim=0)
                self.mod2_c_count = torch.sum(mask_class, dim=0)
                self.sum_E_P_f = torch.sum(out_P, dim=0)
                self.mod2_attr_count = torch.sum(mask_env, dim=0)
            else:
                self.sum_E_Q_f = E_Q_f.clone()
                self.sum_E_P_f = E_P_f.clone()

            self.scores_during_training["TP"] = np.zeros((self.amt_of_threshes - 1,
                                                          self.n_c))
            self.scores_during_training["FP"] = np.zeros((self.amt_of_threshes - 1,
                                                          self.n_c))
            self.scores_during_training["FN"] = np.zeros((self.amt_of_threshes - 1,
                                                          self.n_c))

        else:
            if cfg.MODEL.ENVIRONMENT.INDIV_WEIGHT:
                self.sum_E_Q_f += torch.sum(out_Q, dim=0)
                self.mod2_c_count += torch.sum(mask_class, dim=0)
                self.sum_E_P_f += torch.sum(out_P, dim=0)
                self.mod2_attr_count += torch.sum(mask_env, dim=0)
            else:
                self.sum_E_Q_f += E_Q_f.clone()
                self.sum_E_P_f += E_P_f.clone()

        if cfg.MODEL.ENVIRONMENT.INDIV_WEIGHT:
            self.avg_E_Q_f = self.sum_E_Q_f / torch.max(self.mod2_c_count, one_tensor.float())
            self.avg_E_P_f = self.sum_E_P_f / torch.max(self.mod2_attr_count, one_tensor.float())
            self.avg_E_Q_f[self.mod2_c_count == 0] = 0
        else:
            self.avg_E_Q_f = self.sum_E_Q_f / (batch_index+1.)
            self.avg_E_P_f = self.sum_E_P_f / (batch_index+1.)

    def get_metrics(self, reset: bool = False):
        #return current metrics
        metric_dict = {}
        metric_dict['emd_estimate'] = self.emd_estimate
        metric_dict['precision'] = self.precision_sum/(self.amt_of_samples+1.e-10)
        metric_dict['recall'] = self.recall_sum/(self.amt_of_samples+1.e-10)
        metric_dict['f1_score'] = self.f1_sum/(self.amt_of_samples+1.e-10)

        if reset:
            self.emd_estimate = 0.
            self.f1_sims = 0.
            self.amt_sims = 0.
            self.precision_sum =0.
            self.recall_sum =0.
            self.f1_sum =0.
            self.amt_of_samples=0.


        return {x: y for x, y in metric_dict.items()}


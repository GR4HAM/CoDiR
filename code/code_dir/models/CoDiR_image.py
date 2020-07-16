from cfg.config_general import cfg
import torchvision
from models.CoDiR_model import CoDiR_model

class CoDiR_image(CoDiR_model):
    #implement abstract CoDiR_model class

    def __init__(self, input_dim, feature_dim, n_c, n_l, n_e, R, amt_of_train_batches,selection_method="thresh"):
        super().__init__(input_dim, feature_dim, n_c, n_l, n_e, R, amt_of_train_batches, selection_method)


    def define_network(self):
        #Start from well known models
        if cfg.MODEL.MODEL_TYPE == 'resnet18':
            self.startnet = torchvision.models.resnet18(num_classes=self.n_c * self.n_e)
        elif cfg.MODEL.MODEL_TYPE == 'resnet101':
            self.startnet = torchvision.models.resnet101(num_classes=self.n_c * self.n_e)
        elif cfg.MODEL.MODEL_TYPE == 'inception_v3':
            self.startnet = torchvision.models.inception_v3(num_classes=self.n_c * self.n_e, aux_logits=False)
        else:
            #default, use resnet18
            print("WARNING: unknown model_type, using resnet18")
            self.startnet = torchvision.models.resnet18(num_classes=self.n_c * self.n_e)

    def forward_network_pass(self, input):
        out = self.startnet(input)
        return out


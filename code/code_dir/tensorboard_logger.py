from tensorboardX import SummaryWriter
from cfg.config_general import cfg

class Tensorboard_logger(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def save_metrics(self, metrics_dict, type_of_metric, phase, batch_counter):
        assert type(type_of_metric) is str
        assert type(phase) is str

        if cfg.DEBUG:
            phase = "debug/"+phase

        for k,v in metrics_dict.items():

            assert type(k) is str
            self.writer.add_scalar(phase+"/"+type_of_metric+"/"+k,v,batch_counter)

    def save_histogram(self, np_array, name, i):


        self.writer.add_histogram(name, np_array, i, bins="auto")


    def close(self):
        # make sure pending events are flushed to disk and files are closed properly
        self.writer.close()
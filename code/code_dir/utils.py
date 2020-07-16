import torch
from cfg.config_general import cfg
import os
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_idx2word(word2idx):
    #create idx2word
    idx2word = {}
    for w,i_w in word2idx.items():
        if i_w in idx2word:
            print("WARNING FOUND INDEX IN IDX2WORD BUT SHOULDNT HAVE")
        else:
            idx2word[i_w] = w
    return idx2word

def save_data_results(res, out_dir, filename='quantitative_eval.csv'):
    r_all = ""
    for a in res:

        r_all +=  str(a)+"\t"

    with open('%s/%s' % (out_dir, filename), 'a') as fp:
        fp.write(r_all+'\n')

def torch_integer_to_one_hot(integer_tensor, num_classes):
    #expected input shape of integer_tensor = [batch_size, 1] or [batch_size, seq_len, 1]
    #returns: one_hot vector of integer_tensor

    one_tensor = torch.tensor(1)
    if len(integer_tensor.shape)>1 and integer_tensor.shape[1]>1:
        rel_dim = 2

        ground_truth_one_hot = torch.FloatTensor(integer_tensor.shape[0], integer_tensor.shape[1], num_classes)  # bs x (seq_len x) n_c
        integer_tensor = integer_tensor.unsqueeze(rel_dim)
    else:
        rel_dim = 1
        ground_truth_one_hot = torch.FloatTensor(integer_tensor.shape[0], num_classes)  # bs x (seq_len x) n_c
    if cfg.CUDA:
        one_tensor = one_tensor.cuda()
        ground_truth_one_hot = ground_truth_one_hot.cuda()
    ground_truth_one_hot.zero_()
    ground_truth_one_hot.scatter_(rel_dim, integer_tensor, one_tensor)
    return ground_truth_one_hot


def tonp(pt_var):
    #convert from pt to numpy for debug reasons
    return pt_var.detach().cpu().numpy()


def weights_init(m):
    classname = m.__class__.__name__
    #initialize conv but not basicconv2d from inception that is already initialized
    if classname.find('Conv') != -1 and (classname.find('BasicConv2d')==-1) and classname.find('MeanPool') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('Layernorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)




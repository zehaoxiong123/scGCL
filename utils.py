#from.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch.nn.functional as F
import os.path as osp
import os
from torch.autograd import Variable
import argparse
from torch_geometric.utils import dropout_adj
from singlecell import Singlecell
import scipy.sparse as sp
import numpy as np
# np.random.seed(0)
import scipy
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from datetime import datetime
import pandas as pd

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="AFGRL")
    parser.add_argument("--dataset", type=str, default="adam", help="Name of the dataset. Supported names are: wikics, cs, computers, photo, and physics")

    parser.add_argument('--checkpoint_dir', type=str, default = './model_checkpoints', help='directory to save checkpoint')
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--task", type=str, default="clustering", help="Downstream task. Supported tasks are: node, clustering, similarity")
    
    parser.add_argument("--layers", nargs='?', default='[2048]', help="The number of units of each layer of the GNN. Default is [256]")
    parser.add_argument("--pred_hid", type=int, default=2048, help="The number of hidden units of layer of the predictor. Default is 512")
    
    parser.add_argument("--topk", type=int, default=4, help="The number of neighbors to search")
    parser.add_argument("--clus_num_iters", type=int, default=20)
    parser.add_argument("--num_centroids", type=int, default=9, help="The number of centroids for K-means Clustering")
    parser.add_argument("--num_kmeans", type=int, default=5, help="The number of K-means Clustering for being robust to randomness")
    parser.add_argument("--eval_freq", type=float, default=5, help="The frequency of model evaluation")
    parser.add_argument("--mad", type=float, default=0.9, help="Moving Average Decay for Teacher Network")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")    
    parser.add_argument("--es", type=int, default=300, help="Early Stopping Criterion")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--aug_params", "-p", nargs="+", default=[0.3, 0.4, 0.3, 0.2],help="Hyperparameters for augmentation (p_f1, p_f2, p_e1, p_e2). Default is [0.2, 0.1, 0.2, 0.3]")
    return parser.parse_known_args()


class Augmentation:

    def __init__(self, p_f1=0.2, p_f2=0.1, p_e1=0.2, p_e2=0.3):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"

    def _feature_masking(self, data, device):
        feat_mask1 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f1
        feat_mask2 = torch.FloatTensor(data.x.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(device), feat_mask2.to(device)
        x1, x2 = data.x.clone(), data.x.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        edge_index1, edge_attr1 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e1)
        edge_index2, edge_attr2 = dropout_adj(data.edge_index, data.edge_attr, p=self.p_e2)

        new_data1, new_data2 = data.clone(), data.clone()
        new_data1.x, new_data2.x = x1, x2
        new_data1.edge_index, new_data2.edge_index = edge_index1, edge_index2
        new_data1.edge_attr, new_data2.edge_attr = edge_attr1, edge_attr2

        return new_data1, new_data2

    def __call__(self, data):
        return self._feature_masking(data)

def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "muraro":
        dataset = "Muraro"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_10x_bladder":
        dataset = "Quake_10x_Bladder"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "plasschaert":
        dataset = "Plasschaert"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "macosko":
        dataset = "Macosko"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "wang_lung":
        dataset = "Wang_lung"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_10x_limb_muscle":
        dataset = "Quake_10x_Limb_Muscle"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_10x_spleen":
        dataset = "Quake_10x_Spleen"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "romanov":
        dataset = "Romanov"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/"+dataset+"/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "adam":
        dataset = "Adam"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_smart-seq2_diaphragm":
        dataset = "Quake_Smart-seq2_Diaphragm"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_smart-seq2_limb_muscle":
        dataset = "Quake_Smart-seq2_Limb_Muscle"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_smart-seq2_lung":
        dataset = "Quake_Smart-seq2_Lung"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "quake_smart-seq2_trachea":
        dataset = "Quake_Smart-seq2_Trachea"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "romanov":
        dataset = "Romanov"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "young":
        dataset = "Young"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "chen":
        dataset = "Chen"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "tosches_turtle":
        dataset = "Tosches_turtle"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/" + dataset + "/data.h5"
        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "covid-19":
        dataset = "Covid-19"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/COVID-19_interpretable/GSE164948_covid_control_RNA_counts.csv"

        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "ad":
        dataset = "AD"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/AD_interpretable/GSE138852_counts.csv"

        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "zeisel":
        dataset = "Zeisel"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/Zeisel/data.csv"

        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    elif dataset == "klein":
        dataset = "Klein"
        root = osp.join(root, "pyg")
        filepath = "./test_csv/Klein/data.csv"

        params = {"kwargs": {"root": root, "name": dataset, "filepath": filepath},
                  "name": dataset, "class": Singlecell, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)

def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)

def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if data.train_mask is None:
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
            else:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    else:  # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


def l2_normalize(x):
    return x / torch.sqrt(torch.sum(x**2, dim=1).unsqueeze(1))


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name in ['dataset']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def fill_ones(x):
    n_data = x.shape[0]
    x = torch.sparse_coo_tensor(x._indices(), torch.ones(x._nnz()).to(x.device), [n_data, n_data])

    return x

def read_label_2_str():
    data = pd.read_csv("./test_csv/Klein/label.csv", header=0, sep=",")
    label_num = np.array(data)[:,1].astype(str)
    for i in range(len(label_num)):
        if label_num[i] == "0":
            label_num[i] = "Day 0"
        elif label_num[i] == "1":
            label_num[i] = "Day 2"
        elif label_num[i] == "2":
            label_num[i] = "Day 4"
        elif label_num[i] == "3":
            label_num[i] = "Day 7"
    a = pd.DataFrame(label_num)
    a.to_csv("./test_csv/Klein/label_str.csv")

if __name__=="__main__":
    read_label_2_str()

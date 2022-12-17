from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import sys
from torch import optim
from tensorboardX import SummaryWriter

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average, loss_fn, repeat_1d_tensor, currentTime
import copy
import pandas as pd
from data import Dataset
from embedder import embedder
from utils import config2string
from embedder import Encoder
import faiss
from ZINB_loss import ZINB,NB
import utils

class AFGRL_ModelTrainer(embedder):
    
    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

    def _init(self):
        args = self._args
        self._task = args.task
        print("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        # self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        self._device = "cpu"
        # torch.cuda.set_device(self._device)
        self._dataset = Dataset(root=args.root, dataset=args.dataset)
        self._loader = DataLoader(dataset=self._dataset)
        #设置输入维度为[500,1024]
        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers
        self._model = AFGRL(layers, args).to(self._device)
        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay= 1e-5)

    def train(self):

        self.best_test_acc, self.best_dev_acc, self.best_test_std, self.best_dev_std, self.best_epoch = 0, 0, 0, 0, 0 
        self.best_dev_accs = []
        self.best_embeddings = None
        sillog = []
        # get Random Initial accuracy
        self.infer_embeddings(0)
        print("initial accuracy ")
        self.evaluate(self._task, 0, sillog)

        f_final = open("results/{}.txt".format(self._args.embedder), "a")

        # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                # augmentation = utils.Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]),
                #                                   float(self._args.aug_params[2]), float(self._args.aug_params[3]))

                batch_data.to(self._device)
                # view1, view2 = augmentation._feature_masking(batch_data, self._device)

                emb, loss = self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                     neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                     edge_weight=batch_data.edge_attr, epoch=epoch)
                # emb, loss = self._model(x=view1.x, x2=view2.x, y=batch_data.y, edge_index=view1.edge_index,
                #                         edge_index_2=view2.edge_index,
                #                         neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                #                         edge_weight=view1.edge_attr, edge_weight_2=view2.edge_attr, epoch=epoch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()

                st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), epoch, self._args.epochs, loss.item())
                print(st)

            if (epoch) % 5 == 0:
                self.infer_embeddings(epoch)
                # self.evaluate(self._task, epoch)
                self.evaluate(self._task, epoch, sillog)
                


        print("\nTraining Done!")
        print("[Final] {}".format(self.st_best))
        print('Saving checkpoint...')
        torch.save(self.best_embeddings, os.path.join(self._args.checkpoint_dir,
                                            'embeddings_{}_{}.pt'.format(self._args.dataset,
                                                                         self._args.task)))
        # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
        a = pd.DataFrame(self.best_embeddings).T
        a.to_csv("./results/student.csv")
        self.st_best = '** [last epoch: {}] last NMI: {:.4f} **\n'.format(self._args.epochs, self.best_test_acc)
        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))


class AFGRL(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super().__init__()
        dec_dim = [512, 256]
        #student_encoder将输入的数据进行GCN操作
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        #teacher_encoder对student_encoder进行深拷贝
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        #不对teacher_encoder的权重进行更新
        set_requires_grad(self.teacher_encoder, False)
        #mad取值0.9,创建EMA对象
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        #根据student_encoder在teacher_encoder中找到最近邻的区域化嵌入
        self.neighbor = Neighbor(args)
        #rep_dim设置为1024
        rep_dim = layer_config[-1]
        rep_dim_o = layer_config[0]
        #设置student_predictor[1024,2048,1024]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid), nn.ReLU(), nn.Linear(args.pred_hid, rep_dim), nn.ReLU())
        self.ZINB_Encoder = nn.Sequential(nn.Linear(rep_dim, dec_dim[0]), nn.ReLU(),
                                          nn.Linear(dec_dim[0], dec_dim[1]), nn.ReLU())
        self.pi_Encoder =  nn.Sequential(nn.Linear(dec_dim[1], rep_dim_o),nn.Sigmoid())
        self.disp_Encoder = nn.Sequential(nn.Linear(dec_dim[1], rep_dim_o), nn.Softplus())
        self.mean_Encoder = nn.Linear(dec_dim[1], rep_dim_o)
        self.student_predictor.apply(init_weights)
        self.relu = nn.ReLU()
        self.topk = args.topk
        # self._device = args.device
        self._device = "cpu"
    def clip_by_tensor(self,t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = torch.tensor(t,dtype = torch.float32)
        t_min = torch.tensor(t_min,dtype = torch.float32)
        t_max = torch.tensor(t_max,dtype = torch.float32)

        result = torch.tensor((t >= t_min),dtype = torch.float32) * t + torch.tensor((t < t_min),dtype = torch.float32) * t_min
        result = torch.tensor((result <= t_max),dtype = torch.float32) * result + torch.tensor((result > t_max),dtype = torch.float32) * t_max
        return result

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):
        #student得到卷积之后的hi
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # student_ = self.student_encoder(x=x2, edge_index=edge_index_2, edge_weight=edge_weight_2)

        #pred得到映射器映射出的融合信息z
        pred = self.student_predictor(student)
        # pred_ = self.student_predictor(student_)
        z = self.ZINB_Encoder(student)
        pi = self.pi_Encoder(z)
        disp = self.disp_Encoder(z)
        disp = self.clip_by_tensor(disp,1e-4,1e4)
        mean = self.mean_Encoder(z)
        mean = self.clip_by_tensor(torch.exp(mean),1e-5,1e6)
        modify = 0
        with torch.no_grad():
            #teacher和student使用一个权重
            teacher = self.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
            # teacher_ = self.teacher_encoder(x=x2, edge_index=edge_index_2, edge_weight=edge_weight_2)
        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]), [x.shape[0], x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [x.shape[0], x.shape[0]])
        #
        ind, k = self.neighbor(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2), self.topk, epoch)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0, debug=False)
        zinb_loss = zinb.loss(x, mean, mean=True)
        # adj_recon = torch.matmul(z,z.T)
        loss1 = loss_fn(pred[ind[0]], teacher[ind[1]].detach())
        loss2 = loss_fn(pred[ind[1]], teacher[ind[0]].detach())
        # loss1 = loss_fn(pred, teacher_.detach())
        # loss2 = loss_fn(pred_, teacher.detach())
        recon_loss = torch.nn.MSELoss(reduction='mean')
        recon_loss_ = recon_loss(x,student)
        # adj_recon_ = recon_loss(adj.to_dense(),adj_recon)
        loss_reforce = (loss1 + loss2)
        if modify == 0:
            loss = zinb_loss + loss_reforce + recon_loss_
        elif modify == 1:
            loss = loss_reforce + recon_loss_
        elif modify == 2:
            loss = zinb_loss
        #ind,k返回值暂时去除
        return student, loss.mean()



class Neighbor(nn.Module):
    def __init__(self, args):
        super(Neighbor, self).__init__()
        # self.device = args.device
        self.device = "cpu"
        self.num_centroids = args.num_centroids
        self.num_kmeans = args.num_kmeans
        self.clus_num_iters = args.clus_num_iters

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k, epoch):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids
        niter = self.clus_num_iters

        pred_labels = []
        # d_means = []
        for seed in range(self.num_kmeans):
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, gpu=False, seed=seed + 1234)
            kmeans.train(teacher.cpu().numpy())
            _, I_kmeans = kmeans.index.search(teacher.cpu().numpy(), 1)

            clust_labels = I_kmeans[:,0]
            # d_means.append(D_kmeans)
            pred_labels.append(clust_labels)
        # d_means_s = np.stack(d_means, axis=0)
        # d_means_s = np.mean(d_means_s,axis=0)
        # d_means_s = torch.from_numpy(d_means_s).float()
        # print(d_means_s.shape)
        pred_labels = np.stack(pred_labels, axis=0)
        cluster_labels = torch.from_numpy(pred_labels).float()

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_kmeans):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn, I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality

        return pos_.coalesce()._indices(), I_knn.shape[1]

    def create_sparse(self, I):
        
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])
        
        assert len(similar) == len(index)
        indices = torch.tensor([index, similar],dtype=torch.int32).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result

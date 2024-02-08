import torch
from torch_geometric.utils.undirected import to_undirected
import os.path as osp
import os
import torch_geometric.utils
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import fill_diag

from utils import process
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree,remove_self_loops
from scipy.sparse import coo_matrix
class embedder:
    def __init__(self, args):
        self.args = args
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        if args.dataset == "dblp":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_dblp4057_mat(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "acm":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_acm_mat()
            features = process.preprocess_features(features)
        if args.dataset == "imdb":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb5k_mat(args.sc)
            #adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_imdb(args.sc)
            features = process.preprocess_features(features)
        if args.dataset == "freebase":
            adj_list, features, labels, idx_train, idx_val, idx_test, adj_fusion = process.load_freebase(args.sc)
            args.ft_size = features[0].shape[0]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]

        if args.dataset in ["acm", "imdb", "freebase", "dblp"]:
            adj_list = [process.sparse_mx_to_torch_sparse_tensor(adj) for adj in adj_list]
            adj_list = [adj.to_dense() for adj in adj_list]
            ##############################################
            adj_list = [process.normalize_graph(adj).to_sparse() for adj in adj_list]
            args.nb_nodes = adj_list[0].shape[0]
            args.nb_classes = labels.shape[1]
            #args.ft_size = features.shape[1]
            features_list = []
            features = shuffle_tensor(features, axis=1)
            X_NUM_A = int(features.shape[1] * (args.ol_x / 2 + 0.5))
            args.ft_size=X_NUM_A
            features_list.append(features[:, :X_NUM_A])
            data_features=features[:,-X_NUM_A:]
            if args.num_view==2:
                features_list.append(data_features)
            if args.num_view==3:
                data_features = shuffle_tensor(data_features, axis=1)
                half = X_NUM_A // 2 
                features_list.append(data_features[:,:half])
                features_list.append(data_features[:,half:])
        
            #for i in range(args.num_view):
                #features_list.append(features)
            self.adj_list = adj_list
            self.features = torch.FloatTensor(features)
            self.features = [torch.FloatTensor(features) for features in features_list]
            self.labels = torch.FloatTensor(labels).to(args.device)
            self.idx_train = torch.LongTensor(idx_train).to(args.device)
            self.idx_val = torch.LongTensor(idx_val).to(args.device)
            self.idx_test = torch.LongTensor(idx_test).to(args.device)
            #self.idx_p_list = idx_p_list
            #self.sample_edge_list = sample_edge_list
        if args.dataset in ['cora','citeseer','pubmed']:
            self.features, self.adj_list, self.labels,self.idx_train,self.idx_val,self.idx_test = data_load(name=args.dataset, ol_edge=args.ol_x, ol_x=args.ol_x,nv=args.num_view)
            self.labels = F.one_hot(self.labels).to(args.device)
            args.ft_size = self.features[0].shape[1]
            args.nb_nodes = self.labels.shape[0]
            args.nb_classes = self.labels.shape[1]

def overlap_split_edge(name, indices, ol=0.1,nv=2):
    path = 'fixed_edge_split'
    split_path = osp.join(path, name + f'ol_{ol}_{nv}.npz')
    if not osp.exists(path):
        os.mkdir(path)
    if osp.exists(split_path):
        f = np.load(split_path)
        if nv==2:
            return [f['ei_A'], f['ei_B']]
        if nv==3:
            return [f['ei_A'], f['ei_B'], f['ei_C']]
    else:
        # (2,N)
        indices = remove_self_loops(indices)[0]
        indices = indices[:, indices[0] < indices[1]]
        indices = shuffle_tensor(indices, axis=1)
        NUM_A = int(indices.shape[1] * (ol / 2 + 0.5))
        # NUM_B = int(indices.shape[1] * 0.9 * (ol + 5))
        # 0: A ; 1: B
        ei_A = to_undirected(indices[:, :NUM_A]).numpy()
        if nv==2:
            ei_B = to_undirected(indices[:, -NUM_A:]).numpy()
            np.savez(split_path, ei_A=ei_A, ei_B=ei_B)
            return [ei_A, ei_B]
        if nv==3:
            ei_B = to_undirected(indices[:, -NUM_A//2:]).numpy()
            ei_C = to_undirected(indices[:, NUM_A:NUM_A//2]).numpy()
            np.savez(split_path, ei_A=ei_A, ei_B=ei_B,ei_C=ei_C)
            return [ei_A, ei_B,ei_C]

def data_load(name='cora', ol_edge=1, ol_x=1,nv=2):
    path = 'data'
    if name in ['cora', 'citeseer', 'pubmed']:
        data = Planetoid(root=path, name=name, split='public', pre_transform=T.NormalizeFeatures())[0]
        # data = Planetoid(root=path, name=name, split='public')[0]
        ei, X, labels = data.edge_index, data.x, data.y
        num_nodes = len(data.y)
        ei_sparse_list=[]
        ei_list= overlap_split_edge(name, ei, ol_edge,nv)
        for ei in ei_list:
            ei = torch.sparse_coo_tensor(torch.from_numpy(ei).long(), torch.ones(ei.shape[1]).float(),
                                           size=(num_nodes, num_nodes)).coalesce()
            ei_sparse_list.append(ei)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    else:
        pass
    # to Sparse
    # ===============%%%%%%%%%%#
    # %%%%%%%%=================#
    X = shuffle_tensor(X, axis=1)
    # (N,D)
    X_NUM_A = int(X.shape[1] * (ol_x / 2 + 0.5))
    # X_NUM_B = int(X.shape[1] * 0.9 * (ol_x + 5))
    features_list=[]
    X_A = X[:, :X_NUM_A].float()
    features_list.append(X_A)
    data_features = X[:, -X_NUM_A:].float()
    if nv==2:
        features_list.append(data_features)
    if nv==3:
        data_features = shuffle_tensor(data_features, axis=1)
        half = X_NUM_A // 2 
        features_list.append(data_features[:,:half])
        features_list.append(data_features[:,half:])

    return features_list,ei_sparse_list, labels, train_mask, val_mask, test_mask
            

def shuffle_tensor(input_tensor, axis):
    idx = torch.randperm(input_tensor.shape[axis])
    shuff = input_tensor[:, idx]
    return shuff



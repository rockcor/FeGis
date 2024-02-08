import argparse
import os.path as osp
import numpy as np
from sklearn.preprocessing import normalize
from ruamel.yaml import YAML
import os
from models import DMG
from utils import *
import warnings
import torch
import random
import optuna
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve

from torch_geometric.transforms import NormalizeFeatures

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_args(model_name, dataset, yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_try', type=int, default=5, help='trys')#evaluate try
    parser.add_argument('--num_view', type=int, default=2, help='views')#evaluate iterater
    parser.add_argument("--model_name", default=model_name)
    parser.add_argument("--dataset","-D", default=dataset)
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=False, help='sparse adjacency matrix')
    #parser.add_argument('--use_pretrain', type=bool, default=False, help='use_pretrain')
    parser.add_argument('--isBias', type=bool, default=False, help='isBias')#bias in GNN
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=42, help='the seed to use')
    parser.add_argument('--optimizer', type=str, default='adam', help='the seed to use')
    
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')#dropout in GNN
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--save_path', type=str, default="./saved_results_private_final_msemaxcorrmin_train", help='root for saving the results')
    parser.add_argument("--common_dim", default=32, help="Dimensionality of c", type=int)
    parser.add_argument("--specific_dim", default=32, help="Dimensionality of p", type=int)
    
    parser.add_argument("--lr_utility", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--lr_shared", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--lr_decoder", default=1e-3, help="Learning rate for minimization", type=float)
    parser.add_argument("--lr_specific", default=1e-3, help="Learning rate for minimization", type=float)
    
    parser.add_argument("--weight_decay", default=0, help="Weight decay for parameters eta", type=float)
    parser.add_argument("--alpha", default=1, help="Reconstruction adj error coefficient", type=float)
    parser.add_argument("--lammbda", default=0.2, help="Reconstruction feature error coefficient", type=float)

    parser.add_argument("--utility_epochs", default=10, help="Number of epochs in task and data utility loss", type=int)  # 10
    parser.add_argument("--shared_epochs", default=10, help="Number of epochs in shared channel",type=int)  # 10
    parser.add_argument("--specific_epochs", default=2, help="Number of epochs in specific channel", type=int)  # 10
    parser.add_argument("--num_iterEM", default=10, help="Number of iters in clustering specific embedding", type=int)  # 10
    parser.add_argument("--num_iters", default=50, help="Number of iters in clustering specific embedding", type=int)  # 10

    parser.add_argument("--target_size", default=256, help="Number of hidden neurons for phi", type=int)
    parser.add_argument("--hid_units", default=256, help="Number of hidden neurons in GCN", type=int)
    parser.add_argument("--decolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--encolayer", default=2, help="Number of decoder layers", type=int)
    parser.add_argument("--ol_x", default=0.5, help="features overlap ratio", type=float)
    parser.add_argument("--group_size", default=16, help="num nodes in a group while extracting common info", type=int)
    parser.add_argument("--cluster_number", default=50, help="cluster number while transfering specific info", type=int)
    parser.add_argument("--attack", default=1, help="cluster number while transfering specific info", type=int)
    parser.add_argument("--node_level", default=0, help="cluster number while transfering specific info", type=int)



    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask
def unsupervised_link_prediction(X, test_edges_pos, test_edges_neg):
    # X_normalized = normalize(torch2scipy(X), axis=1, norm='l1')
    # layer norm
    X_normalized = normalize(X, axis=1, norm='l1')
    scores_matrix = X_normalized @ X_normalized.T
    # scores_matrix = np.clip(scores_matrix.toarray(), a_min=0, a_max=1)
    scores_matrix = np.clip(scores_matrix, a_min=0, a_max=1)
    roc_score, ap_score, labels_all, nodes_pair_all, preds_all = get_score(
        test_edges_pos.T, test_edges_neg.T, scores_matrix, apply_sigmoid=True)

    #return roc_score, ap_score, scores_matrix, labels_all, nodes_pair_all, preds_all
    return roc_score, ap_score
def cal_IOU(ei1, ei2):
    union = {tuple(i) for i in ei1.indices().t().numpy()} | {tuple(i) for i in ei2.indices().t().numpy()}
    intersec = {tuple(i) for i in ei1.indices().t().numpy()} & {tuple(i) for i in ei2.indices().t().numpy()}
    return len(intersec) / len(union)
def split_sample_edge(size, name='dblp', ol=0, train_prop=0.85, val_prop=0.05):
    path = 'fixed_edge_split'
    split_path = osp.join(path, name + f'{train_prop}_mask.npy')
    if not osp.exists(path):
        os.mkdir(path)
    if osp.exists(split_path):
        return np.load(split_path)
    else:
        shuffled_idx = np.random.permutation(np.arange(size))
        n_train = int(round(train_prop * size))
        n_val = int(round(val_prop * size))
        train = index_to_mask(shuffled_idx[:n_train], size)
        val = index_to_mask(shuffled_idx[n_train:n_train + n_val], size)
        test = index_to_mask(shuffled_idx[-(n_train + n_val):], size)
        mask = np.stack((train, val, test))
        np.save(split_path, mask)
    return mask

def neg_sample(ei, n_nodes, n_neg_samples, name, ol, prop):
    path = 'fixed_edge_split'
    split_path = osp.join(path, name + f'{ol}_{prop}_nsf.npy')
    if not osp.exists(path):
        os.mkdir(path)
    if osp.exists(split_path):
        return np.load(split_path)
    else:
        ns_f = negative_sampling(ei, n_nodes, n_neg_samples)
        np.save(split_path, ns_f)
    return ns_f
def get_score(edges_pos, edges_neg, score_matrix, apply_sigmoid=True):
    # Store positive edge predictions, actual values
    if apply_sigmoid == True:
        preds_pos = sigmoid(score_matrix[edges_pos[0], edges_pos[1]])
        preds_neg = sigmoid(score_matrix[edges_neg[0], edges_neg[1]])
    else:
        preds_pos = score_matrix[edges_pos[0], edges_pos[1]]
        preds_neg = score_matrix[edges_neg[0], edges_neg[1]]

    # Calculate optimal threshold
    preds_all = np.concatenate((preds_pos, preds_neg), axis=0)
    labels_all = np.concatenate((np.ones(len(preds_pos)), np.zeros(len(preds_neg))), axis=0)
    nodes_pair_all = np.concatenate((edges_pos, edges_neg), axis=0)

    #     threshold=get_threshold(labels_all,preds_all)
    # ans = np.where(preds_all < optimal_threshold, 0, 1)
    roc_score = roc_auc_score(labels_all, preds_all)

    ap_score = average_precision_score(labels_all, preds_all)
    # acc_score = accuracy_score(labels_all, ans)
    print(
        f"num_samples: {len(labels_all)}, "
        f"ap: {ap_score:.4f}, auc: {roc_score:.4f}")
    # f1 = f1_score(labels_all, ans, average='macro')

    return roc_score, ap_score, labels_all, nodes_pair_all, preds_all
def main():
    args = get_args(
        model_name="FeGis",
        dataset="cora",
    )
    args.specific_dim=args.common_dim
    ma_list,mi_list=[],[]
    ap_list,auc_list=[],[]
    printConfig(args)
    for i in range(args.num_try):
        seed = args.seed+i
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        embedder = DMG(args)
        if args.attack:
            emb_B,adj_list,test_macro_f1, test_micro_f1 = embedder.training()
            adj_A=adj_list[0].cpu()
            adj_B=adj_list[1].cpu()
            #print(f'edge IOU: {cal_IOU(adj_A,adj_B)}')
            ma_list.append(test_macro_f1)
            mi_list.append(test_micro_f1)
            # mothod: similarity with emb_B
            ei_B=adj_B.indices().t()
            num_edges_B=ei_B.shape[0]
            num_nodes=emb_B.shape[0]
            mask_ei = split_sample_edge(size=num_edges_B, name=args.dataset, ol=args.ol_x, train_prop=0.85, val_prop=0.05)
            # 0: train, 1: valid, 2:test
            edge_pos = ei_B[mask_ei[2]]
            edge_neg = neg_sample(ei_B.t(), num_nodes, num_edges_B, args.dataset, ol=args.ol_x, prop=0.85).T
            edge_neg = edge_neg[mask_ei[2]]
            auc,ap=unsupervised_link_prediction(emb_B.detach().cpu(), edge_pos, edge_neg)
            auc_list.append(auc)
            ap_list.append(ap)
        else:
            test_macro_f1, test_micro_f1 = embedder.training()
            ma_list.append(test_macro_f1)
            mi_list.append(test_micro_f1)
    ma=np.array(ma_list)
    mi=np.array(mi_list)
    auc=np.array(auc_list)
    ap=np.array(ap_list)
    return ma,mi,auc,ap


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=UserWarning)
    ma,mi,auc,ap = main()
    print('====> macro_f1s = {:.2f} + {:.2f}  micro_f1 = {:.2f} + {:.2f}'.format(100*ma.mean(),100*ma.std(),100*mi.mean(),100*mi.std()))
    print('====> avg-auc = {:.2f} + {:.2f}  avg-ap = {:.2f} + {:.2f}'.format(100*auc.mean(),100*auc.std(),100*ap.mean(),100*ap.std()))
        


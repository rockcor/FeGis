import os
import torch.nn as nn
import torch.optim as optim
from evaluate import evaluate
from loader import embedder
from utils.process import update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import trange
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from torch.nn import CrossEntropyLoss
from models.gcn_encoder import GCN,GCN2,GAT
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics import f1_score
from utils import process
import logging


def config_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fhandler = logging.FileHandler(log_path, mode='w')
        shandler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fhandler.setFormatter(formatter)
        shandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.addHandler(shandler)
    return logger



class DMG(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.criteria = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)

    def training(self):
        args=self.args
        logger = config_logger(self.args.save_path + '.log')
        logger.handlers.clear()
        #logger.info(self.args)
        # # ===================================================#

        features = [feature.to(self.args.device) for feature in self.features]
        adj_list = [adj.to(self.args.device) for adj in self.adj_list]


        logger.info("Started training...")

        ae_model = GNNDAE(args).to(args.device)
        args.cluster_number=args.nb_nodes//(8*args.group_size)

        optimizer_utility = create_optimizer(args.optimizer, ae_model, args.lr_utility, args.weight_decay)
        optimizer_shared = create_optimizer(args.optimizer, ae_model, args.lr_shared, args.weight_decay)
        optimizer_specific = create_optimizer(args.optimizer, ae_model, args.lr_specific, args.weight_decay)

        ae_model.train()

        best = 1e9
        val_accs = []
        val_micro_f1s = []
        val_macro_f1s = []
        test_accs = []
        test_micro_f1s = []
        test_macro_f1s = []
        pbar=trange(1,args.num_iters+1)
        for itr in pbar:

            
            utility_model,data_task_loss,clusters= utility(logger,ae_model,features,adj_list,args,optimizer_utility, self.labels, self.idx_train)
            common_loss, shared_model = shared_channel(logger,utility_model, features, adj_list, args,optimizer_shared,
                                                       self.args.shared_epochs,clusters)
            specific_loss, ae_model = specific_corr(logger,shared_model,features,adj_list,args, optimizer_specific,args.specific_epochs)
            
            loss = data_task_loss + common_loss + specific_loss

            ae_model.eval()
            embedding = []
            common, private,_ = ae_model.encode(features, adj_list)
            
            embedding.append(common[0])
            embedding.append(private[0])
            for i in range(1,args.num_view):
                if args.node_level:
                    node_centroids_data=private[i]
                else:
                    group_indices, aggregated_data_task, node_centroids_data=cluster_and_group_nodes(private[i], self.args.cluster_number, self.args.group_size, self.args.device)
                    
                embedding.append(node_centroids_data)
            embedding = torch.cat(embedding, dim=1)
            logits = ae_model.predict_eval(embedding)
            val_acc, val_f1_macro, val_f1_micro, test_acc, test_f1_macro, test_f1_micro = evaluate_eval(logits, self.labels,
                                                                                                   self.idx_val,
                                                                                                   self.idx_test)
            
            val_accs.append(val_acc)
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
            test_accs.append(test_acc)
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            
   
            pbar.set_postfix(loss=loss.item(),val_f1_macro=val_f1_macro, val_f1_micro=val_f1_micro, test_f1_macro=test_f1_macro, test_f1_micro=test_f1_micro)

    
        logger.info("Evaluating...")
        max_iter = val_accs.index(max(val_accs))
        test_acc=test_accs[max_iter]

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        test_f1_macro=test_macro_f1s[max_iter]

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        test_f1_micro=test_micro_f1s[max_iter]
        #logger.info('====>test acc=: {:.4f} macro_f1s=: {:.4f} micro_f1s= {:.4f}'.format(test_acc,test_f1_macro,test_f1_micro))
        print('====> macro_f1s=: {:.4f} micro_f1s= {:.4f}'.format(test_f1_macro,test_f1_micro))

        
        if args.attack:
            return node_centroids_data,adj_list,test_f1_macro,test_f1_micro

        else:
            return test_acc,test_f1_macro, test_f1_micro

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def evaluate_eval(logits,labels,idx_val,idx_test):
    preds = torch.argmax(logits, dim=1)
    val_lbls = torch.argmax(labels[idx_val], dim=1)
    test_lbls = torch.argmax(labels[idx_test], dim=1)

    val_acc = torch.sum(preds[idx_val] == val_lbls).float() / val_lbls.shape[0]
    val_f1_macro = f1_score(val_lbls.cpu(), preds[idx_val].cpu(), average='macro')
    val_f1_micro = f1_score(val_lbls.cpu(), preds[idx_val].cpu(), average='micro')


    test_acc = torch.sum(preds[idx_test] == test_lbls).float() / test_lbls.shape[0]
    test_f1_macro = f1_score(test_lbls.cpu(), preds[idx_test].cpu(), average='macro')
    test_f1_micro = f1_score(test_lbls.cpu(), preds[idx_test].cpu(), average='micro')
    return val_acc,val_f1_macro,val_f1_micro,test_acc,test_f1_macro ,test_f1_micro

def cluster_data(logger,specific_data,nb_classes,device,centre=None):
    specific_data_cpu = specific_data.cpu()
    specific_data_cpu_numpy=specific_data_cpu.detach().numpy()
    kmeans = MiniBatchKMeans(n_clusters=nb_classes,n_init=5)
    clusters=kmeans.fit_predict(specific_data_cpu_numpy)
    centroids_initial = torch.tensor(kmeans.cluster_centers_)
    if centre is not None:
        centroids_initial=centre
    else:
        centroids_initial = centroids_initial.to(device)
    centroids, clusters = _update_clusters_gpu(specific_data, centroids_initial)
    node_centroids = centroids[clusters]
    return clusters,node_centroids,centre
    

def _update_clusters_gpu(node_embs, centroids, niter=3):
    """ E-M: remapping nodes to new centroids (M), updating centroids (E) """
    for i in range(niter):
        clusters = torch.argmax(
            torch.stack([torch.nn.functional.cosine_similarity(node_embs, cen) for cen in centroids]), dim=0)
        centroids = torch.stack([torch.mean(node_embs[clusters == c], dim=0) for c in range(centroids.size(0))], dim=0)
    return centroids, clusters


def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    #sigma1=sigma2=1
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)
    #corr = torch.pow(torch.mean(x1 * x2),2) / (sigma1 * sigma2)

    return corr

    

def utility(logger,model, features, adj_list, args, optimizer, label, mask_train):
    
    clusters=None
    centre=None
    model.train()
    for utepoch in range(args.utility_epochs):
        optimizer.zero_grad(set_to_none=True)
        common, private,target_loss = model(features, adj_list)  # common and private is list,contain embedding in two views
        embedding = []
        embedding.append(common[0])
        embedding.append(private[0])
        for i in range(1,args.num_view):
            if args.node_level:
                node_centroids_data=private[i]
            else:
                clusters, aggregated_data_task, node_centroids_data=cluster_and_group_nodes(private[i], args.cluster_number, args.group_size, args.device)
            embedding.append(node_centroids_data)
        embedding= torch.cat(embedding, 1)
        label_train = label[mask_train]
        label_train = torch.argmax(label_train, dim=1)

        loss_task,recons_fea_loss_data=model.decode(embedding,mask_train,label_train, common, private, features, utepoch, adj_list)

        
        data_task_loss =loss_task+args.lammbda*recons_fea_loss_data
        data_task_loss.backward()
        optimizer.step()

    return model,data_task_loss,clusters


def shared_channel(logger,model, features, adj_list, args, optimizer, shared_epochs,clusters):
    model.train()
    for comepoch in range(shared_epochs):
        optimizer.zero_grad()
        common, private,_ = model(features, adj_list)  # common and private is list,contain embedding in two views
        common_group=[]
        if not args.node_level:
            group_indices, aggregated_data_task, nodes_representation_task=cluster_and_group_nodes(common[0], args.cluster_number, args.group_size, args.device)
            common_group.append(aggregated_data_task)
            for i in range(1,args.num_view):
                aggregated_shared_data, nodes_shared_representation=aggregate_shared_data(group_indices, common[i], args.device)
                common_group.append(aggregated_shared_data)
        else:
            common_group=common


        loss=0
        for i in range(1,args.num_view):
            loss+= 1-compute_corr(common_group[0], common_group[i])
        loss.backward()
        optimizer.step()
        #logger.info('====> shared__epoch: {} common_loss = {:.2f} '.format(comepoch, loss))
    return loss, model


def specific_corr(logger, model, features, adj_list, args, optimizer, specific_epochs):
    for encoder in model.encoder:
        for param in encoder.pipe_s.parameters():  # shared_encoder
            param.requires_grad = False
    model.train()
    for spepoch in range(specific_epochs):
        optimizer.zero_grad()
        common, private,_ = model(features, adj_list)  # common and private is list,contain embedding in two views
        loss=0
        for i in range(args.num_view):
            loss+=compute_corr(common[i], private[i])
        loss.backward()
        optimizer.step()
        logger.info('====> specific_epoch: {} specific_loss = {:.2f} '.format(spepoch, loss))

    # Re-enable gradient for shared_encoder
    for encoder in model.encoder:
        for param in encoder.pipe_s.parameters():  # shared_encoder
            param.requires_grad = True
    return loss, model


    

class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe_s = GCN(args.ft_size,  args.hid_units,args.common_dim, args.encolayer, args.dropout)
        self.pipe_p = GCN(args.ft_size, args.hid_units, args.common_dim, args.encolayer, args.dropout)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, args.ft_size))
        self.l = torch.nn.MSELoss(reduction='mean')
        self.l2 = sce_loss

    def random_remask(self,rep,remask_rate=0.5):
        num_nodes=rep.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.enc_mask_token
        return rep
 

    def forward(self, x, adj,mask_rate=0):
        if mask_rate==0:
            common = self.pipe_s(x, adj)
            private = self.pipe_p(x, adj)

        else:
            common = self.pipe_s(self.random_remask(x), adj)
            private = self.pipe_p(self.random_remask(x), adj)

        return [common,private],0


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        #self.dec_in_dim=args.common_dim+args.specific_dim
        self.dec_in_dim=args.common_dim
        self.linear = Linearlayer(self.dec_in_dim, args.hid_units, args.ft_size,args.decolayer)
        self.revGCN = GCN(self.dec_in_dim, args.hid_units, args.ft_size,args.decolayer,args.dropout)
        #self.GCN2 = GCN(args.ft_size, args.hid_units, self.dec_in_dim,args.encolayer)
        self.projector = nn.Sequential(
            nn.Linear(self.dec_in_dim, args.ft_size,bias=False),
            nn.ReLU(),
        )

        self.dec_mask_token = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.l = torch.nn.MSELoss(reduction='mean')
        self.l2 = sce_loss

    def random_remask(self,rep,remask_rate=0.5):
        num_nodes=rep.shape[0]
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token
        return rep


    def feature_recons_p(self, z_data,x_true,adj_data,num_remask=1):
        # Feature reconstruction loss
        recons_err=0
        for i in range(num_remask):
            z_data =self.random_remask(z_data)
            recons = self.linear(z_data, adj_data)
            recons_err+= self.l(recons, x_true)
        return recons_err

    def feature_recons_s(self, z_data,x_true,adj_data,num_remask=1):
        recons_err=0
        for i in range(num_remask):
            z_data =self.random_remask(z_data)
            recons = self.revGCN(z_data, adj_data)
            recons_err+= self.l(recons, x_true)
        return recons_err

    def forward(self, s, p,x_true,epoch,adj_data):
        recons_fea_err=self.feature_recons_s(s,x_true,adj_data,num_remask=1)
        recons_fea_err+=self.feature_recons_p(p,x_true,adj_data,num_remask=1)

        return recons_fea_err


class Predictor(nn.Module):
    def __init__(self, emd_dim, hid,nb_classes,args):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear((args.num_view+1)*emd_dim, hid),
                nn.ReLU(),
                nn.Dropout(p=args.dropout),
                nn.Linear(hid,nb_classes))

    def forward(self, z):
        ret = self.fc(z)
        return ret


class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.num_view
        self.encoder = nn.ModuleList()
        self.data_decoder = nn.ModuleList()
        self.encoder.append(GNNEncoder(args))
        self.data_decoder.append(Decoder(args))

        args.ft_size//=args.num_view-1
        for _ in range(args.num_view-1):
            self.encoder.append(GNNEncoder(args))
            self.data_decoder.append(Decoder(args))
        self.task_predictor=Predictor(args.common_dim, args.hid_units,args.nb_classes,args)

    def forward(self, x, adj_list):
        common = []
        private = []
        loss=0
        for i in range(self.args.num_view):
            tmp,view_loss = self.encoder[i](x[i], adj_list[i],mask_rate=0.5)
            common.append(tmp[0])
            private.append(tmp[1])
            loss+=view_loss


        return common, private,loss
    def encode(self, x, adj_list):
        common = []
        private = []
        loss=0
        for i in range(self.args.num_view):
            tmp,view_loss = self.encoder[i](x[i], adj_list[i],mask_rate=0)
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
            loss+=view_loss


        return common, private,loss

    def decode(self,emb_predict,mask_train,train_lbls,com_data,spe_data,x_true,epoch,adj_data):
        xent = nn.CrossEntropyLoss()
        logits=self.task_predictor(emb_predict)
        loss_task = xent(logits[mask_train], train_lbls)
        recons_fea_loss=0
        for i in range(1,self.args.num_view):
            recons_fea_loss+=self.data_decoder[i](com_data[i],spe_data[i],x_true[i],epoch,adj_data[i])
        return loss_task,recons_fea_loss


    def predict_eval(self,z_eval):
        logits = self.task_predictor(z_eval)
        return logits
def aggregate_shared_data(group_indices, shared_data, device): 
    total_nodes = shared_data.size(0) 
    feature_size = shared_data.size(1) 
    aggregated_shared_data = []
    nodes_shared_representation = torch.zeros(total_nodes, feature_size).to(device)
    
    for indices in group_indices:
        group_data = shared_data[indices]
        group_mean = group_data.mean(dim=0, keepdim=True)
        aggregated_shared_data.append(group_mean)
        nodes_shared_representation[indices] = group_mean.repeat(len(indices), 1)
        
    aggregated_shared_data = torch.cat(aggregated_shared_data, dim=0)
    return aggregated_shared_data, nodes_shared_representation

def cluster_and_group_nodes(specific_data, nb_classes, group_size, device, centre=None):
    clusters, _, _ = cluster_data(None, specific_data, nb_classes, device, centre)
    total_nodes = specific_data.size(0)
    feature_size = specific_data.size(1)
    nodes_representation = torch.zeros(total_nodes, feature_size, device=device)

    aggregated_data = []
    group_indices = []

    for cluster_id in range(nb_classes):
        indices = torch.nonzero(clusters == cluster_id, as_tuple=False).squeeze(1)
        cluster_data_emb = specific_data[indices]

        if len(indices) == 0:
            continue

        nnodes = len(indices)
        left = nnodes % group_size
        num_unleft = nnodes - left

        if num_unleft > 0:
            grouped = cluster_data_emb[:num_unleft].view(-1, group_size, feature_size)
            grouped_emb = grouped.mean(dim=1)  # 计�~W�~O�~D�~Z~D平�~]~G�~I��~A
            nodes_representation[indices[:num_unleft]] = grouped_emb.repeat(1, group_size).view(-1, feature_size)
            aggregated_data.append(grouped_emb)
            for i in range(0, num_unleft, group_size):
                group_indices.append(indices[i:i + group_size])

        if left > 0:
            leftover = cluster_data_emb[-left:]
            leftover_mean = leftover.mean(dim=0, keepdim=True)
            nodes_representation[indices[-left:]] = leftover_mean.repeat(left, 1)
            aggregated_data.append(leftover_mean)
            group_indices.append(indices[-left:])

    aggregated_data = torch.cat(aggregated_data, dim=0)
    return group_indices, aggregated_data, nodes_representation

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)


    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

import argparse

from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from hsic import dHSIC
# from tensorboardX import SummaryWriter

from copy import deepcopy


# def cycle_index(num, shift):
#     arr = torch.arange(num) + shift
#     arr[-shift:] = torch.arange(shift)
#     return arr

# class Discriminator(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
#         self.reset_parameters()

#     def reset_parameters(self):
#         size = self.weight.size(0)
#         uniform(size, self.weight)

#     def forward(self, x, summary):
#         h = torch.matmul(summary, self.weight)
#         return torch.sum(x*h, dim = 1)
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def triplet_margin_loss(anchor, postive, negative, margin, p=2, eps=1e-6):
    def distance(input1, input2, p, eps):
     # Compute the distance (p-norm)
        pnorm = torch.pow(torch.abs((input1 - input2 + eps)), p)
        pnorm = torch.pow(torch.sum(pnorm, dim=-1), 1.0 / p)
        return pnorm
 
    dist_pos = distance(anchor, postive, p, eps)
    dist_neg = distance(anchor, negative, p, eps)

    output = margin + dist_pos - dist_neg
    output[output<0]=0
 
    return output

class mgsc(nn.Module):

    def __init__(self, gnn, alpha, beta):
        super(mgsc, self).__init__()
        self.gnn = gnn
        self.embedding_dim = self.gnn.emb_dim
        self.alpha =alpha
        self.beta = beta
        self.k = 3
        self.d = self.embedding_dim / self.k
        
        self.pool = global_mean_pool
        # self.projection_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

        self.proj_head1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head2 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

        sizes = [self.embedding_dim] + [self.embedding_dim*8] * 3
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)


    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        # x = self.projection_head(x)
        return x
    
    def decorrelation_loss(self, x1, x2, lambd=0.013):
        # empirical cross-correlation matrix
        if x1.shape[0] == 1:
            c = self.projector(
            (x1)).T @ self.projector((x2)) 
        else:   
            c = self.bn(self.projector(
                (x1))).T @ self.bn(self.projector((x2)))
        # sum the cross-correlation matrix between all gpus
        c.div_(len(x1))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        decorrelation = on_diag + lambd * off_diag

        return decorrelation
    
    def loss_cal(self, x, x_aug, x_aug_large):
      
        triplet_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2)
        
        decorrelation = self.decorrelation_loss(x, x_aug, 0.013)
        
        batch_size, _ = x.size()
        
        x_mask=[]
        x_aug_mask=[]
        x_aug_large_mask=[]
        
        x = self.proj_head1(x)
        x_aug = self.proj_head1(x_aug)
        x_aug_large = self.proj_head1(x_aug_large)
        
        x = nn.functional.normalize(x, dim=-1)
        x_aug = nn.functional.normalize(x_aug, dim=-1)
        x_aug_large = nn.functional.normalize(x_aug_large, dim=-1)
        
        tloss = triplet_loss(x, x_aug, x_aug_large)
        
        c1 = dHSIC(*[x[:,0:int(self.d)], x[:,int(self.d):int(self.d * 2)],x[:,int(self.d *2):]])
        c2 = dHSIC(*[x_aug[:,0:int(self.d)], x_aug[:,int(self.d):int(self.d * 2)],x_aug[:,int(self.d *2):]])
        c3 = dHSIC(*[x_aug_large[:,0:int(self.d)], x_aug_large[:,int(self.d):int(self.d * 2)],x_aug_large[:,int(self.d *2):]])
        
        
        for i in range(self.k):
            x2 = x_aug.clone()
            x2[:,int(self.d * i) : int(self.d * (i + 1))] = 0
            x_aug_mask.append(x2)
            x3 = x_aug_large.clone()
            x3[:,int(self.d * i) : int(self.d * (i + 1))] = 0
            x_aug_large_mask.append(x3)

        weight = []
        for i in range(self.k):
            weight1 = x * x_aug_mask[i] - x * x_aug_large_mask[i] 
            weight1 = torch.sum(weight1,-1)
            weight1 = weight1.reshape(batch_size,1)
            weight.append(weight1)
        weight = torch.cat(weight, dim=1)
        weight = F.softmax(weight, dim=-1)
        weight = (1 - weight) * 0.5
        
        x = self.proj_head2(x)
        for i in range(self.k):
            # x_mask[i] = self.proj_head1(x_mask[i])
            x_aug_mask[i] = self.proj_head2(x_aug_mask[i]) 
            x_aug_large_mask[i] = self.proj_head2(x_aug_large_mask[i])
        
        x = nn.functional.normalize(x, dim=-1)
        for i in range(self.k):
            # x_mask[i] = self.proj_head1(x_mask[i])
            x_aug_mask[i] = nn.functional.normalize(x_aug_mask[i], dim=-1) 
            x_aug_large_mask[i] = nn.functional.normalize(x_aug_large_mask[i], dim=-1)
            
        
        
        mask_loss=[]    
        for i in range(self.k):
            loss1 = triplet_margin_loss(x, x_aug_mask[i], x_aug_large_mask[i], margin=0.2)
            loss1 = loss1.reshape(batch_size, 1)
            mask_loss.append(loss1)
        mask_loss = torch.cat(mask_loss, dim=1)
        mask_loss_batch = torch.sum(mask_loss * weight)
        
        # for i in range(self.k):
        #     loss1 = triplet_loss(x, x_aug_mask[i], x_aug_large_mask[i])
        #     tloss += 0.5 * loss1     
        # print ("tloss:", tloss, "  decorrelation:", decorrelation,"  c1:", c1, " c2:", c2, "c3:", c3)    
        return tloss + self.beta * mask_loss_batch + self.alpha * decorrelation  + 0.01 * (c1 + c2 + c3)



def train(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = deepcopy(dataset)
    dataset2 = deepcopy(dataset)
    dataset1.aug, dataset1.aug_ratio = args.aug, args.rate1
    dataset2.aug, dataset2.aug_ratio = args.aug, args.rate2

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers = 1, shuffle=False)
    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = 4, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = 4, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader, loader1, loader2), desc="Iteration")):
 
        batch1, batch2, batch3 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        batch3 = batch3.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        x3 = model.forward_cl(batch3.x, batch3.edge_index, batch3.edge_attr, batch3.batch)
        loss = model.loss_cal(x1, x2, x3)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    # parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 3, help='number of workers for dataset loading')
    parser.add_argument('--aug', type=str, default = None)
    parser.add_argument('--rate1', type=float, default = 0.1)
    parser.add_argument('--rate2', type=float, default = 0.2)
    parser.add_argument('--alpha', type=float, default=0.01, help='weight of decorrelation')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of fine-view')
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset, aug='none')
    # dataset2 = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset, aug=args.aug, aug_ratio=args.rate2)
    # print(dataset)

    #set up model
    gnn = GNN(args.num_layer, emb_dim = args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    alpha = args.alpha
    beta = args.beta
    
    model = mgsc(gnn, alpha, beta)
    
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, dataset, optimizer)

        print(train_acc)
        print(train_loss)

        if epoch % 20 == 0:
            torch.save(gnn.state_dict(), "./models_mgsc/mgsc_" + str(epoch) + "_{}_{}.pth".format(args.lr, args.aug))

if __name__ == "__main__":
    main()

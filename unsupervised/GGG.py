import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from aug import TUDataset_aug as TUDataset
from torch_geometric.loader import DataLoader
from torch import optim

from gin import Encoder
from evaluate_embedding import evaluate_embedding

from hsic import dHSIC
from arguments import arg_parse

import logging



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

class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, lambda1, lambda2, lambda3):
    super(simclr, self).__init__()
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.lambda3 = lambda3
    self.k = 3
    self.d = hidden_dim * num_gc_layers / self.k
    

    self.embedding_dim = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.proj_head1 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
    self.proj_head2 = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
    

    self.init_emb()

    # Barlow twins high-dim projector
    sizes = [self.embedding_dim] + [self.embedding_dim*8] * 3
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
    self.projector = nn.Sequential(*layers)
    self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

  def init_emb(self):
    # initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch, num_graphs):

    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)

    return y
     
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
    
    batch_size, _ = x.size()

    dloss = self.decorrelation_loss(x, x_aug)
    
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.2, p=2)

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
    
    x_aug_mask=[]
    x_aug_large_mask=[]
    
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
        x_aug_mask[i] = self.proj_head2(x_aug_mask[i]) 
        x_aug_large_mask[i] = self.proj_head2(x_aug_large_mask[i])
    
    x = nn.functional.normalize(x, dim=-1)
    for i in range(self.k):
        x_aug_mask[i] = nn.functional.normalize(x_aug_mask[i], dim=-1) 
        x_aug_large_mask[i] = nn.functional.normalize(x_aug_large_mask[i], dim=-1)

    mask_loss=[]    
    for i in range(self.k):
        loss1 = triplet_margin_loss(x, x_aug_mask[i], x_aug_large_mask[i], margin=0.2)
        loss1 = loss1.reshape(batch_size, 1)
        mask_loss.append(loss1)
    mask_loss = torch.cat(mask_loss, dim=1)
    mask_loss_batch = torch.sum(mask_loss * weight)
    
    return tloss + self.lambda1 * mask_loss_batch + self.lambda2 * dloss + self.lambda3 * (c1 + c2 + c3)

import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    print('start')
    args = arg_parse()
    seeds = args.seeds
    lr = args.lr
    DS = args.DS
    
    print(args)
    
    f=open('{}.txt'.format(args.DS, args.lr), 'a+', encoding='utf-8')
    
    accuracies = {'val':[], 'test':[]}
    epochs = 100
    batch_size = 128
    
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS) #获得文件所在绝对路径，并去掉文件名，连接.data

    dataset = TUDataset(path, name=DS, aug=args.aug, rate1=args.rate1, rate2=args.rate2).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, num_workers=16, pin_memory=True)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')
    
    acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        best_val = -1
        setup_seed(seed)
        model = simclr(args.hidden_dim, args.num_gc_layers, args.lambda1, args.lambda2, args.lambda3).to(device)        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(1, epochs+1)):
            loss_all = 0      
            model.train()
            for data in dataloader:
                # print(data)

                data, data_aug, data_aug_large = data
        
                optimizer.zero_grad()
                data = data.to(device)
    
                node_num, _ = data.x.size()
                x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                data_aug = data_aug.to(device)
                x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

                data_aug_large = data_aug_large.to(device)
                x_aug_large = model(data_aug_large.x, data_aug_large.edge_index, data_aug_large.batch, data_aug_large.num_graphs)
                
                loss = model.loss_cal(x, x_aug, x_aug_large)
            
                loss_all += loss.item() * data_aug_large.num_graphs
                loss.backward()
                optimizer.step()

            print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
            
            if epoch % 1 == 0:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader_eval)
                acc_val, acc = evaluate_embedding(emb, y)
                f.write(f"{(acc * 100):.2f}" +'\n')
    #             accuracies['val'].append(acc_val)
    #             accuracies['test'].append(acc)
    #             if acc_val > best_val:
    #                 best_val = acc_val
    #                 best_test = acc
                    
    #     acc_list.append(best_test)
    #     print('best test', best_test)
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {(final_acc * 100):.2f}±{(final_acc_std * 100):.2f}")
    f.write(f"{(final_acc * 100):.2f}±{(final_acc_std * 100):.2f}" +'\n')
    f.close()
    

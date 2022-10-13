import os
import os.path as osp
import shutil
from traceback import print_tb

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data
from torch_geometric.transforms import OneHotDegree
import torch.nn.functional as F
from torch_geometric.utils import degree
from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, cleaned=False, 
                 aug=None, rate1=0.1, rate2=0.25):
        self.name = name
        self.cleaned = cleaned
        self.rate1 = rate1
        self.rate2 = rate2
        self.aug =aug

        super(TUDataset_aug, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # print('self.data', self.data.x)
        # print('self.slices', self.slices)
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]

        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109' ):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)          

            print("Using degree as node features")
            if self.name == 'COLLAB':
                MAX_DEGREES = 300
            elif self.name == 'REDDIT-BINARY':
                MAX_DEGREES = 4
            elif self.name == 'REDDIT-MULTI-5K':
                MAX_DEGREES = 1
            elif self.name == 'IMDB-BINARY':
                MAX_DEGREES = 25
            
            one_hot_degree = OneHotDegree(MAX_DEGREES, in_degree=False, cat=False)
            self.data = one_hot_degree(self.data,self.slices)
            
            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)
        # print(11111)
        # print(self.data)
        # print('self.data', self.data.x.size())
    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url('{}/{}.zip'.format(url, self.name), folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)
        print(self.data)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[0],
                                                       slices[0 + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        _, num_feature = data.x.size()

        return num_feature

    def get_aug(self, data,  drop_percent):
        
        if self.aug == 'dnodes':
            data_aug = drop_nodes(data, drop_percent)
        elif self.aug == 'pedges':
            data_aug = permute_edges(data, drop_percent)
        elif self.aug == 'subgraph':
            data_aug = subgraph(data, drop_percent)
        elif self.aug == 'mask_nodes':
            data_aug = mask_nodes(data, drop_percent)
        elif self.aug == 'none':
            """
            if data.edge_index.max() > data.x.size()[0]:
                print(data.edge_index)
                print(data.x.size())
                assert False
            """
            data_aug = data
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))
        
        elif self.aug == 'random1':
            n = np.random.randint(2)
            if n == 0:
               data_aug = drop_nodes(data, drop_percent)
            elif n == 1:
               data_aug = permute_edges(data, drop_percent)
            else:
                print('sample error')
                assert False
        
        elif self.aug == 'random2':
            n = np.random.randint(2)
            if n == 0:
               data_aug = drop_nodes(data, drop_percent)
            elif n == 1:
               data_aug = subgraph(data, drop_percent)
            else:
                print('sample error')
                assert False
        elif self.aug == 'random3':
            n = np.random.randint(2)
            if n == 0:
               data_aug = permute_edges(data, drop_percent)
            elif n == 1:
               data_aug = subgraph(data, drop_percent)
            else:
                print('sample error')
                assert False 
        elif self.aug == 'random4':
            n = np.random.randint(2)
            if n == 0:
               data_aug = permute_edges(data, drop_percent)
            elif n == 1:
               data_aug = mask_nodes(data, drop_percent)
            else:
                print('sample error')
                assert False
        elif self.aug == 'random5':
            n = np.random.randint(3)
            if n == 0:
               data_aug = drop_nodes(data, drop_percent)
            elif n == 1:
               data_aug = mask_nodes(data, drop_percent)
            elif n == 2:
               data_aug = subgraph(data, drop_percent)
            else:
                print('sample error')
                assert False
        elif self.aug == 'random6':
            n = np.random.randint(3)
            if n == 0:
               data_aug = drop_nodes(data, drop_percent)
            elif n == 1:
               data_aug = permute_edges(data, drop_percent)
            elif n == 2:
               data_aug = subgraph(data, drop_percent)
            else:
                print('sample error')
                assert False
        else:
            print('augmentation error')
            assert False

        return data_aug

    def get(self, idx):
        data, data1, data2 = self.data.__class__(), self.data.__class__(), self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):            
            data.num_nodes, data1.num_nodes, data2.num_nodes = self.data.__num_nodes__[idx], self.data.__num_nodes__[idx], self.data.__num_nodes__[idx]

        for key in self.data.keys:
            if key == 'num_nodes':
                continue
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key,
                                        item)] = slice(slices[idx],
                                                       slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key], data1[key], data2[key] = item[s], item[s], item[s]

        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """
        
        data_aug = self.get_aug(data1, self.rate1)
        
        data_aug_large = self.get_aug(data2, self.rate2)
        
        return data,  data_aug, data_aug_large

def drop_nodes(data, drop_percent=0.1):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * drop_percent)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index


    return data


def permute_edges(data, drop_percent=0.1):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num * drop_percent)
    edge_index = data.edge_index.transpose(0, 1).numpy()    
    
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]

    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

  
def subgraph(data, drop_percent=0.1):

    node_num, _ = data.x.size()
    sub_num = int(node_num * (1 - drop_percent))

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
    
    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            idx_neigh = set([np.random.choice(idx_unsub)])
        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]])).difference(set(idx_sub))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    
    edge_index = data.edge_index.numpy()
    
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0 
    edge_index = adj.nonzero(as_tuple=False).t()
    data.edge_index = edge_index

    return data
    
def mask_nodes(data, drop_percent=0.1):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * drop_percent)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data

    

















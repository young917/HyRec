import math
import argparse
import numpy as np
import datetime
from collections import defaultdict
import pickle
import torch
import PIL
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, init_row, init_col, init_k, batch_size):
        
        self.node_bases = init_row ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        self.edge_bases = init_col ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        
        self.batch_size = batch_size
        self.val_loadflag = val_loadflag
        if val_loadflag:
            self.valfile = val_arg
        else:
            self.valfile = None
        self.row_bases = row_bases
        self.col_bases = col_bases
        self.init_row = init_row
        self.init_col = init_col
        self.row_perm = row_perm
        self.col_perm = col_perm
            
        self.row_idx_list = []
        self.col_idx_list = []
        self.val_idx_list = []
        for i in range(0, len(row_idx_list), batch_size):
            curr_batch_size = min(batch_size, len(row_idx_list) - i)
            self.row_idx_list.append(row_idx_list[i:i+curr_batch_size])
            self.col_idx_list.append(col_idx_list[i:i+curr_batch_size])
            if val_loadflag is False:
                self.val_idx_list.append(val_arg[i:i+curr_batch_size])
        
        
    def __len__(self):
        return len(self.row_idx_list)
    
    def update_perm(self, row_perm, col_perm):
        self.row_perm = row_perm
        self.col_perm = col_perm
    
    def __getitem__(self, index):
        rows = torch.LongTensor(self.row_idx_list[index])
        cols = torch.LongTensor(self.col_idx_list[index])
        if self.val_loadflag is False:
            vals = torch.FloatTensor(self.val_idx_list[index])
        else:
            vals = torch.from_numpy(np.load(self.valfile + "_" + str(index) + ".npy"))
        row_idx = torch.div(self.row_perm[rows].unsqueeze(1), self.row_bases, rounding_mode="trunc") % self.init_row
        col_idx = torch.div(self.col_perm[cols].unsqueeze(1), self.col_bases, rounding_mode="trunc") % self.init_col
        
        inputs = row_idx * self.init_col + col_idx
        
        return inputs, vals


'''
coms_train = utils.ImageDataset(channel_list=chlist)
coms_trainloader = torch.utils.data.DataLoader(
    dataset = coms_train,
    batch_size = args.batch_size,
    shuffle = True,
    num_workers=8
)
'''

# incdata = GraphDataset(graph.row_idx, graph.col_idx, graph.val, args.batch_size, val_loadflag=False, graph.node_bases, graph.edge_bases, graph.init_row, graph.init_col, graph.node_perm, graph.edge_perm)

# adjdata = GraphDataset(graph.adj_row_idx, graph.adj_col_idx, graph.savename + "adj", args.batch_size, val_loadflag=True, graph.node_bases, graph.node_bases, graph.init_row, graph.init_row, graph.node_perm, graph.node_perm)

# linedata = GraphDataset(graph.line_row_idx, graph.line_col_idx, graph.savename + "line", args.batch_size, val_loadflag=True, graph.edge_bases, graph.edge_bases, graph.init_col, graph.init_col, graph.edge_perm, graph.edge_perm)

# inc_loader = torch.utils.data.DataLoader(dataset=incdata, batch_size=1, shuffle=True, num_workers=8)
# adj_loader = torch.utils.data.DataLoader(dataset=adjdata, batch_size=1, shuffle=True, num_workers=8)
# line_loader = torch.utils.data.DataLoader(dataset=linedata, batch_size=1, shuffle=True, num_workers=8)

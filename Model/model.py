from torch import nn
from tqdm import tqdm
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from itertools import chain
import random
import math
import time
import sys

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from collections import defaultdict
        
class SingFit(nn.Module):
    def __init__(self, init_row, init_col, k, device, sq_sum, num_tie=1, save_path=None):
        super(SingFit, self).__init__()
        self.init_row, self.init_col = init_row, init_col        
        self.k = k
        self.num_row, self.num_col = init_row**k, init_col**k
        self.device = device
        
        self.theta = nn.Parameter(torch.rand(init_row * init_col))
        self.theta_softplus = torch.nn.Softplus()
        
        self.node_bases = init_row ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        self.edge_bases = init_col ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        
        self.multi = math.sqrt(sq_sum)
        self.save_path = save_path
        
        self.eps=1e-20
        self.num_tie = num_tie
        
        self.intermediate_klist = []
        self.intermediate_rowbase_list = []
        self.intermediate_colbase_list = []
        remain = self.k % self.num_tie
        for i in range(self.num_tie):
            if remain > 0:
                k = self.k // self.num_tie + 1
                remain -= 1
                self.intermediate_klist.append(k)
            else:
                k = self.k // self.num_tie
                self.intermediate_klist.append(k)
        for i in range(self.num_tie):
            cur_k = self.intermediate_klist[i]
            self.intermediate_rowbase_list.append(self.init_row ** ((cur_k-1) - torch.arange(cur_k)))
            self.intermediate_colbase_list.append(self.init_col ** ((cur_k-1) - torch.arange(cur_k)))
                
    def forward(self, temp=1.0):
        initmat = self.theta_softplus(self.theta)
        initmat_scaled = initmat / torch.sqrt(torch.sum(torch.square(initmat)))
        
        mat = initmat_scaled.view(self.init_row, self.init_col)
        for i in range(1, self.k // self.num_tie):
            mat = torch.kron(mat, initmat_scaled.view(self.init_row, self.init_col))
        
        remain = self.k % self.num_tie
        expected_sv = None
        expected_size = None
        expected_degree = None
        for i in range(self.num_tie):
            if remain > 0:
                imat = torch.kron(mat, initmat_scaled.view(self.init_row, self.init_col))
                remain -= 1
                k = self.k // self.num_tie + 1
                assert k == self.intermediate_klist[i]
                imat = imat * (self.multi ** (1. / self.k * k))
            else:
                k = self.k // self.num_tie
                assert k == self.intermediate_klist[i]
                imat = mat * (self.multi ** (1. / self.k * k))
                
            imat[imat>1] = 1.0
            imat[imat<0] = 0.0
            imat = imat.flatten()
            logits = torch.stack((1-imat, imat), dim=-1)
            _imat = F.gumbel_softmax(torch.log(logits + self.eps), tau=temp, hard=True)
            imat_sample = _imat[:,1]
                
            if i == 0:
                expected_sv = torch.linalg.svdvals(imat_sample.view(self.init_row ** k, -1))
                expected_size = torch.sum(imat_sample.view(self.init_row ** k, -1), dim=0)
                expected_degree = torch.sum(imat_sample.view(self.init_row ** k, -1), dim=1)
            else:
                next_expected_sv = torch.linalg.svdvals(imat_sample.view(self.init_row ** k, -1))
                next_expected_size = torch.sum(imat_sample.view(self.init_row ** k, -1), dim=0)
                next_expected_degree = torch.sum(imat_sample.view(self.init_row ** k, -1), dim=1)
                expected_sv = torch.kron(expected_sv, next_expected_sv)
                expected_size = torch.kron(expected_size, next_expected_size)
                expected_degree = torch.kron(expected_degree, next_expected_degree)
        
        expected_sv, _ = torch.sort(expected_sv, descending=True)
        expected_size, _ = torch.sort(expected_size, descending=True)
        expected_degree, _ = torch.sort(expected_degree, descending=True)
        return expected_sv, expected_size, expected_degree
    
    def forward_elem(self, _input):          
        batch_size = _input.size()[0]
        cur_k = _input.size()[1]

        mat = self.theta_softplus(self.theta)
        mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))

        output = mat_scaled[_input[:,1:cur_k].flatten()].view(batch_size, cur_k-1)
        output = mat_scaled[_input[:, 0]] * torch.prod(output, 1)
        output = output * (self.multi ** (1. / self.k * cur_k))

        return output
    
    def intermediate_sample(self, batch_size):
        self.save_sample = {}
        for i in range(self.num_tie):
            cur_k = self.intermediate_klist[i]
            cur_num_row = self.init_row ** cur_k
            cur_num_col = self.init_col ** cur_k
            cur_node_bases = self.init_row ** ((cur_k-1) - torch.arange(cur_k)).to(self.device)
            cur_edge_bases = self.init_col ** ((cur_k-1) - torch.arange(cur_k)).to(self.device)
            
            sample_rows, sample_cols, sample_vals = [], [], []
            num_printed_entry = 0
            num_entry = cur_num_row * cur_num_col
            while num_printed_entry < num_entry:            
                if batch_size > num_entry - num_printed_entry: 
                    batch_size = num_entry - num_printed_entry

                curr_rows, curr_cols = torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.device), torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.device)                    
                curr_rows, curr_cols = curr_rows // cur_num_col, curr_cols % cur_num_col                    

                row_idx = torch.div(curr_rows.unsqueeze(1), cur_node_bases, rounding_mode="trunc") % self.init_row
                col_idx = torch.div(curr_cols.unsqueeze(1), cur_edge_bases, rounding_mode="trunc") % self.init_col  
                
                sample_inputs = row_idx * self.init_col + col_idx 
                assert sample_inputs.shape[0] == batch_size
                assert sample_inputs.shape[1] == cur_k
                samples = self.forward_elem(sample_inputs)
                
                curr_rows = curr_rows.detach().cpu().numpy()
                curr_cols = curr_cols.detach().cpu().numpy()
                samples = samples.detach().cpu().numpy()
                for bi in range(batch_size):
                    r, c, v = curr_rows[bi], curr_cols[bi], samples[bi]
                    if random.random() < v:
                        sample_rows.append(r)
                        sample_cols.append(c)
                        sample_vals.append(1.)
                        
                num_printed_entry += batch_size
            
            self.save_sample[i] = coo_matrix((sample_vals, (sample_rows, sample_cols)), shape=(cur_num_row, cur_num_col))
    
    
    def sample(self):
        until_rows, until_cols, until_vals = self.save_sample[0].row, self.save_sample[0].col, self.save_sample[0].data
        for ai in tqdm(range(1, self.num_tie)):
            cur_k = self.intermediate_klist[ai]
            next_rows, next_cols, next_vals = [], [], []

            cur_rows, cur_cols, cur_vals = self.save_sample[ai].row, self.save_sample[ai].col, self.save_sample[ai].data
            for ur, uc, uv in zip(until_rows, until_cols, until_vals):
                for cr, cc, cv in zip(cur_rows, cur_cols, cur_vals):
                    nr = ur * (self.init_row ** cur_k) + cr
                    nc = uc * (self.init_col ** cur_k) + cc
                    nv = uv * cv
                    assert nv == 0 or nv == 1
                    next_rows.append(nr)
                    next_cols.append(nc)
                    next_vals.append(nv)

            until_rows = next_rows
            until_cols = next_cols
            until_vals = next_vals
        
        return until_rows, until_cols, until_vals
    
    
    def write_matrix(self, graph, batch_size, file_name, nosave_flag=False):
        savedir = "/".join(file_name.split("/")[:-1])
        num_printed_entry = 0
        num_entry = self.num_row * self.num_col
        start_time = time.time()
        self.intermediate_sample(batch_size)
        sample_rows, sample_cols, sample_vals = self.sample()
        end_time = time.time()
        spent = (end_time - start_time) / 60
        print("Time : " + str(spent) + " (min.)")
        with open(savedir + "/eval_time.txt", "w") as f:
            f.write(str(end_time - start_time) + " sec\n")
        if nosave_flag is False:
            with open(file_name, 'w') as f:
                for r,c,v in zip(sample_rows, sample_cols, sample_vals):
                    f.write(f'{c} {r} {v}\n')

        # evaluate singular values, degrees, and sizes
        # computing kronecker products of intermediate singular values
        criterion = torch.nn.MSELoss()
        sample_sv_num = graph.svds.shape[0]
        sampled_svds = None
        for ai in range(self.num_tie):
            cur_rows, cur_cols, cur_vals = self.save_sample[ai].row, self.save_sample[ai].col, self.save_sample[ai].data
            cur_k = self.intermediate_klist[ai]
            cur_num_row = self.init_row ** cur_k
            cur_num_col = self.init_col ** cur_k
            cur_inci_coo = coo_matrix((cur_vals, (cur_rows, cur_cols)), shape=(cur_num_row, cur_num_col))
            _, cur_svds,_ = scipy.sparse.linalg.svds(cur_inci_coo, k=min(sample_sv_num, min(cur_num_row - 1, cur_num_col - 1)))
            cur_svds = torch.from_numpy(cur_svds.copy())
            if ai == 0:
                sampled_svds = cur_svds
            else:
                sampled_svds = torch.kron(sampled_svds, cur_svds)
                sampled_svds, _ = torch.sort(sampled_svds, descending=True)
                sampled_svds = sampled_svds[:graph.svds.shape[0]]
        sampled_svds, _ = torch.sort(sampled_svds, descending=True)
        min_len = min(sampled_svds.shape[0], graph.svds.shape[0])
        loss = criterion(sampled_svds[:min_len], graph.svds[:min_len]).detach().item()
        sample_sizes = torch.zeros(self.init_col ** self.k)
        sample_degrees = torch.zeros(self.init_row ** self.k)
        for r,c,v in zip(sample_rows, sample_cols, sample_vals):
            if v > 0.0:
                sample_sizes[c] += 1
                sample_degrees[r] += 1
       
        answer_sizes = graph.size_dist
        answer_sizes = torch.concat([answer_sizes, torch.zeros((self.init_col ** self.k) - graph.num_col)])
        answer_degrees = graph.degree_dist
        answer_degrees = torch.concat([answer_degrees, torch.zeros((self.init_row ** self.k) - graph.num_row)])
        # size
        sample_sizes, _ = torch.sort(sample_sizes, descending=True)
        sz_loss = criterion(sample_sizes, answer_sizes).item()
        # degree
        sample_degrees, _ = torch.sort(sample_degrees, descending=True)
        deg_loss = criterion(sample_degrees, answer_degrees).item()

        del self.save_sample
        
        return loss, sz_loss, deg_loss, sampled_svds
        
        
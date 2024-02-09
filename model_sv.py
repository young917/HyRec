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
import torch_scatter
    
# class StochasticFitting:
#     def __init__(self, init_row, init_col, k, device, sample_weight):

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)
        
# Total model that includes permutation
class KroneckerSVFitting(nn.Module):
    def __init__(self, init_row, init_col, k, device, sq_sum, binarize_flag=1, gumbel_flag=1, approx=1, evalapprox=1, check_flag=False, save_path=None):
        super(KroneckerSVFitting, self).__init__()
        self.init_row, self.init_col = init_row, init_col        
        self.k = k
        self.num_row, self.num_col = init_row**k, init_col**k
        self.device = device
        
        self.sos = nn.Parameter(torch.rand(init_row * init_col))
        self.sos_softplus = torch.nn.Softplus()
        
        self.node_bases = init_row ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        self.edge_bases = init_col ** ((self.k-1) - torch.arange(self.k)).to(self.device)
        
        self.multi = math.sqrt(sq_sum)
        
        self.binarize_flag = (binarize_flag == 1)
        self.gumbel_flag = (gumbel_flag == 1)
        self.sigmoid_flag = (gumbel_flag == 2)
        
        self.check_flag = check_flag
        self.save_path = save_path
        if self.check_flag:
            assert self.save_path is not None
        
        self.eps=1e-20
        self.approx = approx
        self.evalapprox = evalapprox
        
        self.approx_klist = []
        self.approx_rowbase_list = []
        self.approx_colbase_list = []
        remain = self.k % self.approx
        for i in range(self.approx):
            if remain > 0:
                k = self.k // self.approx + 1
                remain -= 1
                self.approx_klist.append(k)
            else:
                k = self.k // self.approx
                self.approx_klist.append(k)
        for i in range(self.approx):
            cur_k = self.approx_klist[i]
            self.approx_rowbase_list.append(self.init_row ** ((cur_k-1) - torch.arange(cur_k)))
            self.approx_colbase_list.append(self.init_col ** ((cur_k-1) - torch.arange(cur_k)))
                
    def forward(self, temp=1.0):
        initmat = self.sos_softplus(self.sos)
        initmat_scaled = initmat / torch.sqrt(torch.sum(torch.square(initmat)))
        # initmat_scaled *= ((self.multi) ** (1.0 / self.k))
        
        mat = initmat_scaled.view(self.init_row, self.init_col)
        for i in range(1, self.k // self.approx):
            mat = torch.kron(mat, initmat_scaled.view(self.init_row, self.init_col))
        # mat *= self.multi
        
        remain = self.k % self.approx
        expected_sv = None
        for i in range(self.approx):
            if remain > 0:
                imat = torch.kron(mat, initmat_scaled.view(self.init_row, self.init_col))
                remain -= 1
                k = self.k // self.approx + 1
                assert k == self.approx_klist[i]
                imat = imat * (self.multi ** (1. / self.k * k))
            else:
                k = self.k // self.approx
                assert k == self.approx_klist[i]
                imat = mat * (self.multi ** (1. / self.k * k))
                
            if self.sigmoid_flag:
                imat_sample = torch.sigmoid(imat / temp)
            else:
                imat[imat>1] = 1.0
                imat[imat<0] = 0.0
                imat = imat.flatten()
                logits = torch.stack((1-imat, imat), dim=-1) # (|E|*|V|, 2)
                _imat = F.gumbel_softmax(torch.log(logits + self.eps), tau=temp, hard=self.gumbel_flag)
                imat_sample = _imat[:,1] # (torch.zeros_like(mat) * _mat[:,0] + torch.ones_like(mat) * _mat[:,1])
                if self.check_flag:
                    _check = imat_sample[imat_sample > 0.5]
                    print("Sample Mean and # Nonzero " + str(torch.mean(logits[:,1]).item()) + "\t" + str(_check.shape[0]))
                    print(_check.shape)
                    with open(self.save_path + "train_gen_log.txt", '+a') as f:
                        # f.write(",".join([str(c) for c in _check[:10]]) + "\n")
                        f.write(str(_check.shape[0]) + "\t" + str(torch.mean(logits[:,1]).item()) + "\n")
                
            if i == 0:
                expected_sv = torch.linalg.svdvals(imat_sample.view(self.init_row ** k, self.init_col ** k))
            else:
                next_expected_sv = torch.linalg.svdvals(imat_sample.view(self.init_row ** k, self.init_col ** k))
                expected_sv = torch.kron(expected_sv, next_expected_sv)
        
        expected_sv, _ = torch.sort(expected_sv, descending=True)
        return expected_sv
    
    def pre_sample(self, batch_size, sparseflag=False):
        self.save_sample = {}
        for i in range(self.evalapprox):
            cur_k = self.approx_klist[i]
            cur_num_row = self.init_row ** cur_k
            cur_num_col = self.init_col ** cur_k
            cur_node_bases = self.init_row ** ((cur_k-1) - torch.arange(cur_k)).to(self.device)
            cur_edge_bases = self.init_col ** ((cur_k-1) - torch.arange(cur_k)).to(self.device)
            
            if sparseflag is False:
                self.save_sample[i] = torch.zeros(cur_num_row * cur_num_col)
            
            sample_rows, sample_cols, sample_vals = [], [], []
            if self.check_flag:
                sample_entire_vals = []
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
                if self.check_flag:
                    sample_entire_vals += samples.tolist()
                for bi in range(batch_size):
                    r, c, v = curr_rows[bi], curr_cols[bi], samples[bi]
                    if random.random() < v:
                        if sparseflag is False:
                            self.save_sample[i][r * cur_num_col + c] = 1.
                        else:
                            sample_rows.append(r)
                            sample_cols.append(c)
                            sample_vals.append(1.)
                        
                num_printed_entry += batch_size
            
            if sparseflag is False:
                self.save_sample[i] = self.save_sample[i].to(self.device)
            else:
                self.save_sample[i] = coo_matrix((sample_vals, (sample_rows, sample_cols)), shape=(cur_num_row, cur_num_col))
                if self.check_flag:
                    with open(self.save_path + "train_gen_log.txt", "+a") as f:
                        print("Gen Mean and # Nonzero  " + str(np.mean(sample_entire_vals)) + "\t" + str(len(sample_vals)))
                        f.write("[gen]\n")
                        f.write(str(len(sample_vals)) + "\t" + str(np.mean(sample_entire_vals)) + "\n")
    
    def forward_elem(self, _input, approx_flag=False):
        if approx_flag:
            batch_size = _input.size()[0]
            
            _row_idx = _input // self.init_col
            _col_idx = _input % self.init_col
            
            output = torch.ones(batch_size)
            for ai in range(self.evalapprox):
                if ai == 0:
                    sidx = 0
                else:
                    sidx = sum(self.approx_klist[:ai])
                if ai == (self.evalapprox - 1):
                    eidx = self.k
                else:
                    eidx = sidx + self.approx_klist[ai]
                cur_row = _row_idx[:, sidx:eidx]
                cur_col = _col_idx[:, sidx:eidx]
                cur_k = self.approx_klist[ai]
                
                _tmp_row_bases = self.approx_rowbase_list[ai].to(self.device)
                _tmp_col_bases = self.approx_colbase_list[ai].to(self.device)
                # _tmp_row_bases = (self.init_row ** ((cur_k-1) - torch.arange(cur_k))).to(self.device)
                # _tmp_col_bases = (self.init_col ** ((cur_k-1) - torch.arange(cur_k))).to(self.device)
                
                cur_k_row = torch.sum(_tmp_row_bases * cur_row, dim=1, keepdim=True)
                cur_k_col = torch.sum(_tmp_col_bases * cur_col, dim=1, keepdim=True)
                cur_k_inputs = cur_k_row * (self.init_col ** cur_k) + cur_k_col
                cur_k_outputs = self.save_sample[ai][cur_k_inputs].flatten()
                output = output * (cur_k_outputs.detach().cpu())
                # tmplist.append(self.save_sample[ai][cur_k_inputs])
            
            # output = torch.concat(tmplist, dim=1)
            # check = torch.sum(output==1) + torch.sum(output==0)
            # assert check == output.flatten().shape[0]
            # output = torch.prod(output, 1)
            return output
        
        else:            
            batch_size = _input.size()[0]
            cur_k = _input.size()[1]

            mat = self.sos_softplus(self.sos)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))

            output = mat_scaled[_input[:,1:cur_k].flatten()].view(batch_size, cur_k-1)
            output = mat_scaled[_input[:, 0]] * torch.prod(output, 1)
            output = output * (self.multi ** (1. / self.k * cur_k))

            return output
    
    def forward_elem_atonce(self):
        # until_rows, until_cols, until_vals = [], [], []
        until_rows, until_cols, until_vals = self.save_sample[0].row, self.save_sample[0].col, self.save_sample[0].data
        for ai in tqdm(range(1, self.evalapprox)):
            cur_k = self.approx_klist[ai]
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
    
    def write_matrix(self, graph, batch_size, file_name, atonce_flag=False):
        # node_perm = torch.randperm(self.num_row, dtype=torch.long).to(self.device)
        # edge_perm = torch.randperm(self.num_col, dtype=torch.long).to(self.device)
        
        start_time = time.time()
        num_printed_entry = 0
        # num_entry = graph.num_row * graph.num_col
        num_entry = self.num_row * self.num_col
        
        if self.evalapprox > 1 and atonce_flag:
            self.pre_sample(batch_size, sparseflag=True)
            sample_rows, sample_cols, sample_vals = self.forward_elem_atonce()
        else:
            if self.evalapprox > 1:
                self.pre_sample(batch_size)
            sample_rows, sample_cols, sample_vals = [], [], []
            pbar = tqdm(total=num_entry)
            with torch.no_grad():
                while num_printed_entry < num_entry:            
                    if batch_size > num_entry - num_printed_entry: 
                        batch_size = num_entry - num_printed_entry

                    # Build LSTM inputs
                    curr_rows, curr_cols = torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.device), torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.device)                    
                    curr_rows, curr_cols = curr_rows // self.num_col, curr_cols%self.num_col                    
                    # curr_rows, curr_cols = node_perm[curr_rows], edge_perm[curr_cols]            

                    row_idx = torch.div(curr_rows.unsqueeze(1), self.node_bases, rounding_mode="trunc") % self.init_row
                    col_idx = torch.div(curr_cols.unsqueeze(1), self.edge_bases, rounding_mode="trunc") % self.init_col  

                    # Get lstm outputs
                    inputs = row_idx * self.init_col + col_idx
                    if self.evalapprox > 1:
                        samples = self.forward_elem(inputs, approx_flag=True)
                    else:
                        samples = self.forward_elem(inputs)
                        samples = samples.detach().cpu().numpy()

                    curr_rows = curr_rows.detach().cpu().numpy()
                    curr_cols = curr_cols.detach().cpu().numpy()
                    for i in range(batch_size):
                        r, c, v = curr_rows[i], curr_cols[i], samples[i]
                        # if self.evalapprox > 1:
                        if self.check_flag:
                            assert v == 1 or v == 0, str(v)
                        if random.random() < v:
                            sample_rows.append(r)
                            sample_cols.append(c)
                            sample_vals.append(1.)

                    num_printed_entry += batch_size
                    pbar.update(batch_size)
        
        end_time = time.time()
        spent = (end_time - start_time) / 60
        print("Time : " + str(spent) + " (min.)")
        with open(file_name, 'w') as f:
            for r,c,v in zip(sample_rows, sample_cols, sample_vals):
                f.write(f'{c} {r} {v}\n')

        criterion = torch.nn.MSELoss()
        if self.evalapprox > 1 and atonce_flag:
            sampled_svds = None
            for ai in range(self.evalapprox):
                cur_rows, cur_cols, cur_vals = self.save_sample[ai].row, self.save_sample[ai].col, self.save_sample[ai].data
                cur_k = self.approx_klist[ai]
                cur_num_row = self.init_row ** cur_k
                cur_num_col = self.init_col ** cur_k
            
                cur_inci_coo = coo_matrix((cur_vals, (cur_rows, cur_cols)), shape=(cur_num_row, cur_num_col))
                _, cur_svds,_ = scipy.sparse.linalg.svds(cur_inci_coo, k=min(cur_num_row - 1, cur_num_col - 1))
                cur_svds = torch.from_numpy(cur_svds.copy())
                if ai == 0:
                    sampled_svds = cur_svds
                else:
                    sampled_svds = torch.kron(sampled_svds, cur_svds)
            sampled_svds, _ = torch.sort(sampled_svds, descending=True)
            
#             if self.check_flag:
#                 check_sampled_inci = coo_matrix((sample_vals, (sample_rows, sample_cols)), shape=(self.num_row, self.num_col))
#                 _, check_sampled_svds,_ = scipy.sparse.linalg.svds(check_sampled_inci, k=graph.svds.shape[0])
#                 check_sampled_svds = sorted(check_sampled_svds)[::-1]
#                 check_sampled_svds = torch.FloatTensor(check_sampled_svds)

#                 for ci in range(5):
#                     assert sampled_svds[ci] == check_sampled_svds[ci]
            
        else:
            sampled_inci = coo_matrix((sample_vals, (sample_rows, sample_cols)), shape=(self.num_row, self.num_col))
            _, sampled_svds,_ = scipy.sparse.linalg.svds(sampled_inci, k=graph.svds.shape[0])
            sampled_svds = sorted(sampled_svds)[::-1]
            sampled_svds = torch.FloatTensor(sampled_svds)

        min_len = min(sampled_svds.shape[0], graph.svds.shape[0])
        loss = criterion(sampled_svds[:min_len], graph.svds[:min_len]).detach().item()

        print(sampled_svds[:5], "\t", graph.svds[:5])
        
        if self.evalapprox > 1:
            del self.save_sample
        
        return loss, sampled_svds
        
        
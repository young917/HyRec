from itertools import chain
import torch
from scipy.sparse import coo_matrix
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import scipy
import math

# Class that saves hypergraph
class hyperGraph:
    '''
        file_name: file that saves hypergraphs        
        allow_dupli: allow duplicated hyper-edges
    '''
    def getDegree(self):
        deg_list = self.inci_csr.sum(1)
        deg_list = np.squeeze(np.array(deg_list))       
        return list(deg_list)
    
    '''
        Extact edge indices for a given node index
    '''
    def extract_node(self, input_node):
        indptr = self.inci_csr.indptr
        target_idx = self.inci_csr.indices[indptr[input_node]:indptr[input_node + 1]]
        target_val = self.inci_csr.data[indptr[input_node]:indptr[input_node + 1]]
        return target_idx, target_val
        
    '''
        Extract node indices for a given edge index
    '''
    def extract_edge(self, input_edge):
        indptr = self.inci_csc.indptr
        target_idx = self.inci_csc.indices[indptr[input_edge]:indptr[input_edge + 1]]
        target_val = self.inci_csc.data[indptr[input_edge]:indptr[input_edge + 1]]    
        return target_idx, target_val
    
    '''
        file_name: Name of the file
    '''
    def __init__(self, file_name, batch_size=0, binarize=1, svflag=True, saveflag=True):
        self.file_name = file_name
        self.savedir = "./data/" + self.file_name.split("/")[-1][:-4] + "/"
        if os.path.isdir(self.savedir) is False:
            os.makedirs(self.savedir)
        self.savename = self.savedir + str(batch_size) + "/"
        if os.path.isdir(self.savename) is False:
            os.makedirs(self.savename)
            
        self.binarize_flag = (binarize > 0) # default=True
        self.svflag = svflag
        with open(file_name) as f:
            raw_data = f.read()
        
        lines = raw_data.split('\n')
        first_line = lines[0].split(' ')
        # self.num_row, self.num_col = int(first_line[0]), int(first_line[1])
        self.num_col, self.num_row = int(first_line[0]), int(first_line[1])
        lines.pop(0)
        # Assume entres are unique
        lines = [[float(word) for word in line.split(" ")] for line in lines if line]
        # self.row_idx, self.col_idx, self.val = map(list, zip(*lines))
        self.col_idx, self.row_idx, self.val = map(list, zip(*lines))
        self.row_idx = list(map(int, self.row_idx))
        self.col_idx = list(map(int, self.col_idx))
        
        self.entry_sum = sum(self.val)       
        self.sq_sum = sum([entry**2 for entry in self.val])
        if self.binarize_flag is False:
            self.val = [v / math.sqrt(self.sq_sum) for v in self.val]
        print(f'entry sum: {self.entry_sum}, square sum: {self.sq_sum}')
        self.real_num_nonzero = len(self.val)
        
        # Build matrix           
        self.inci_coo = coo_matrix((self.val, (self.row_idx, self.col_idx)), \
                       shape=(self.num_row, self.num_col))
        self.inci_csr, self.inci_csc = self.inci_coo.tocsr(), self.inci_coo.tocsc()
        
        if self.svflag is False:
            # adj
            self.adj_sq_sum = 0
            self.adj_row_idx, self.adj_col_idx = [],[]
            self.adj_vals = []
            mapping = defaultdict(int)
            for ci in tqdm(range(self.num_col), desc="adj"):
                nodes, vals = self.extract_edge(ci)
                for vi_idx in range(len(nodes)):
                    vi = nodes[vi_idx]
                    key = (vi,vi)
                    mapping[key] = mapping[key] + vals[vi_idx] * vals[vi_idx]
                    for vj_idx in range(vi_idx+1, len(nodes)):
                        vj = nodes[vj_idx]
                        key = (vi,vj)
                        mapping[key] = mapping[key] + vals[vi_idx] * vals[vj_idx]                    
                        key = (vj,vi)
                        mapping[key] = mapping[key] + vals[vi_idx] * vals[vj_idx]
            keylist = sorted(list(mapping.keys()))
            bi = 0
            for i in range(0, len(keylist), batch_size):
                curr_batch_size = min(batch_size, len(keylist) - i)
                keys = keylist[i:i+curr_batch_size]
                nodes1, nodes2 = [k[0] for k in keys], [k[1] for k in keys]
                self.adj_row_idx += nodes1
                self.adj_col_idx += nodes2
                curr_val = [mapping[k] for k in keys]
                self.adj_sq_sum += sum([v**2 for v in curr_val])
                if saveflag:
                    np.save(self.savename + "adj_" + str(bi), curr_val)
                    bi += 1
                else:
                    self.adj_vals += curr_val

            # line
            self.line_sq_sum = 0
            self.line_row_idx, self.line_col_idx = [], []
            self.line_vals = []
            if os.path.isfile(self.savename + "line_row_idx.npy") and os.path.isfile(self.savename + "line_col_idx.npy"):
                if os.path.isfile(self.savename + "line_sq_sum.txt") is False:
                    for fname in os.listdir(self.savename):
                        if fname == "line_row_idx.npy" or fname == "line_col_idx.npy":
                            continue
                        if fname.endswith(".npy") is False:
                            continue
                        if fname.startswith("line_"):
                            cur_val = np.load(self.savename + fname)
                            self.line_sq_sum += np.sum(np.square(cur_val))
                    with open(self.savename + "line_sq_sum.txt", "w") as f:
                        f.write(str(self.line_sq_sum) + "\n")
                else:
                    with open(self.savename + "line_sq_sum.txt", "r") as f:
                        self.line_sq_sum = float(f.readline().rstrip())
                self.line_row_idx = np.load(self.savename + "line_row_idx.npy")
                self.line_col_idx = np.load(self.savename + "line_col_idx.npy")
            else:
                mapping = defaultdict(int)
                for ri in tqdm(range(self.num_row), desc="line"):
                    edges, vals = self.extract_node(ri)
                    for ei_idx in range(len(edges)):
                        ei = edges[ei_idx]
                        key = (ei,ei)
                        mapping[key] = mapping[key] + vals[ei_idx] * vals[ei_idx]
                        for ej_idx in range(ei_idx+1, len(edges)):
                            ej = edges[ej_idx]
                            key = (ei,ej)
                            mapping[key] = mapping[key] + vals[ei_idx] * vals[ej_idx]
                            key = (ej,ei)
                            mapping[key] = mapping[key] + vals[ei_idx] * vals[ej_idx]
                keylist = sorted(list(mapping.keys()))
                bi = 0
                for i in range(0, len(keylist), batch_size):
                    curr_batch_size = min(batch_size, len(keylist) - i)
                    keys = keylist[i:i+curr_batch_size]
                    edges1, edges2 = [k[0] for k in keys], [k[1] for k in keys]
                    self.line_row_idx += edges1
                    self.line_col_idx += edges2
                    curr_val = [mapping[k] for k in keys]
                    self.line_sq_sum += sum([v**2 for v in curr_val])
                    if saveflag:
                        np.save(self.savename + "line_" + str(bi), curr_val)
                        bi += 1
                    else:
                        self.line_vals += curr_val
                if saveflag:
                    np.save(self.savename + "line_row_idx", np.array(self.line_row_idx))
                    np.save(self.savename + "line_col_idx", np.array(self.line_col_idx))

            print("Load I, A, and L Done")
        
        
    def calculate_sv(self):
        calculate_flag = True
        
        if os.path.isfile(self.savedir + "svs.txt"):
            svds = []
            calculate_flag = False
            with open(self.savedir + "svs.txt", "r") as f:
                for line in f.readlines():
                    sv = float(line.rstrip())
                    svds.append(sv)
            if len(svds) < min(min(self.num_row - 1, self.num_col - 1), 1000): # 5000
                calculate_flag = True
            else:
                self.svds = torch.FloatTensor(svds)
        
        if calculate_flag:
            _, self.svds,_ = scipy.sparse.linalg.svds(self.inci_coo, k=min(min(self.num_row - 1, self.num_col - 1), 5000))
            self.svds = sorted(self.svds)[::-1]
            # self.svds = np.sort(self.svds)
            with open(self.savedir + "svs.txt", "w") as f:
                for si in range(len(self.svds)):
                    sv = self.svds[si]
                    f.write(str(sv) + "\n")
            self.svds = torch.FloatTensor(self.svds)
            
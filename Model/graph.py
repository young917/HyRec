from itertools import chain
import torch
from scipy.sparse import coo_matrix
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import scipy
import math
# Code from https://github.com/kbrother/NeuKron/blob/ec78e160d3d1003d270587b16908544b2800ce52/matrix/graph.py

class hyperGraph:
    
    def getDegree(self):
        deg_list = self.inci_csr.sum(1)
        deg_list = np.squeeze(np.array(deg_list))       
        return list(deg_list)

    def extract_node(self, input_node):
        indptr = self.inci_csr.indptr
        target_idx = self.inci_csr.indices[indptr[input_node]:indptr[input_node + 1]]
        target_val = self.inci_csr.data[indptr[input_node]:indptr[input_node + 1]]
        return target_idx, target_val
        
    def extract_edge(self, input_edge):
        indptr = self.inci_csc.indptr
        target_idx = self.inci_csc.indices[indptr[input_edge]:indptr[input_edge + 1]]
        target_val = self.inci_csc.data[indptr[input_edge]:indptr[input_edge + 1]]    
        return target_idx, target_val
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.savedir = "./data/" + self.file_name.split("/")[-1][:-4] + "/"
        if os.path.isdir(self.savedir) is False:
            os.makedirs(self.savedir)
            
        with open(file_name) as f:
            raw_data = f.read()
        
        lines = raw_data.split('\n')
        first_line = lines[0].split(' ')
        self.num_col, self.num_row = int(first_line[0]), int(first_line[1])
        lines.pop(0)
        # Assume entres are unique
        lines = [[float(word) for word in line.split(" ")] for line in lines if line]
        self.col_idx, self.row_idx, self.val = map(list, zip(*lines))
        self.row_idx = list(map(int, self.row_idx))
        self.col_idx = list(map(int, self.col_idx))
        
        self.size_dist = torch.zeros(self.num_col, dtype=torch.float)
        for r,c,v in zip(self.row_idx, self.col_idx, self.val):
            if v == 1.0:
                self.size_dist[c] += 1.0
        self.size_dist, _ = torch.sort(self.size_dist, descending=True)
        
        self.entry_sum = sum(self.val)       
        self.sq_sum = sum([entry**2 for entry in self.val])
        print(f'entry sum: {self.entry_sum}, square sum: {self.sq_sum}')
        self.real_num_nonzero = len(self.val)
        
        # Build matrix           
        self.inci_coo = coo_matrix((self.val, (self.row_idx, self.col_idx)), \
                       shape=(self.num_row, self.num_col))
        self.inci_csr, self.inci_csc = self.inci_coo.tocsr(), self.inci_coo.tocsc()
        
    def calculate_sv(self):
        calculate_flag = True
        
        if os.path.isfile(self.savedir + "svs.txt"):
            svds = []
            calculate_flag = False
            with open(self.savedir + "svs.txt", "r") as f:
                for line in f.readlines():
                    sv = float(line.rstrip())
                    svds.append(sv)
            if "tags" in self.file_name:
                len_svds = min(len(svds), 1000)
                svds = svds[:len_svds]
            elif "threads" in self.file_name:
                len_svds = min(len(svds), 1000)
                svds = svds[:len_svds]
            elif "coauth" in self.file_name:
                len_svds = min(len(svds), 500)
                svds = svds[:len_svds]
            self.svds = torch.FloatTensor(svds)
        
        if calculate_flag:
            len_svds = 5000
            if "tags" in self.file_name:
                len_svds = min(len_svds, 1000) # 1000
            elif "threads" in self.file_name:
                len_svds = min(len_svds, 1000) # 1000
            elif "coauth" in self.file_name:
                len_svds = min(len_svds, 500)
            
            _, svds,_ = scipy.sparse.linalg.svds(self.inci_coo, k=min(min(self.num_row - 1, self.num_col - 1), len_svds))
            svds = sorted(svds)[::-1]
                
            with open(self.savedir + "svs.txt", "w") as f:
                for si in range(len(svds)):
                    sv = svds[si]
                    f.write(str(sv) + "\n")
            
            svds = svds[:len_svds]
            self.svds = torch.FloatTensor(svds)
            
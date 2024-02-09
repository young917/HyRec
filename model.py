from torch import nn
import torch
from tqdm import tqdm
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from itertools import chain
import random
import math
import torch_scatter
import time
import sys

# Deep learning model
class KroneckerModel(nn.Module):
    def __init__(self, init_row, init_col, k,  sq_sum, adj_sq_sum, line_sq_sum, use_multi=True, model_type="double", gumbel_flag=True):
        super(KroneckerModel, self).__init__()
        
        self.input_size = init_row * init_col
        self.init_row, self.init_col, self.k = init_row, init_col, k
        self.gumbel_flag = gumbel_flag
        
        self.sos = nn.Parameter(torch.rand(init_row * init_col))        
        self.sos_softplus = torch.nn.Softplus() # torch.nn.Sigmoid()                 
        
        if use_multi: 
            if model_type == "double": 
                self.multiI = nn.Parameter(torch.DoubleTensor([math.sqrt(sq_sum)])) 
                self.multiA = nn.Parameter(torch.DoubleTensor([math.sqrt(adj_sq_sum)])) 
                self.multiL = nn.Parameter(torch.DoubleTensor([math.sqrt(line_sq_sum)])) 
            elif model_type == "float": 
                self.multiI = nn.Parameter(torch.FloatTensor([math.sqrt(sq_sum)])) 
                self.multiA = nn.Parameter(torch.FloatTensor([math.sqrt(adj_sq_sum)])) 
                self.multiL = nn.Parameter(torch.FloatTensor([math.sqrt(line_sq_sum)])) 
        else:
            self.multiI = math.sqrt(sq_sum)
            self.multiA = math.sqrt(adj_sq_sum)
            self.multiL = math.sqrt(line_sq_sum)
     
    '''
        _input: batch size x seq_len
        return value: Probability of input
    '''
    def forward(self, _input, opt):
        # Run model
        batch_size = _input.size()[0]
        # _input = _input.transpose(0, 1)
        # _, batch_size = _input.size()
        
        # Fix average and scale of sos
        if opt == "I":
            mat = self.sos_softplus(self.sos)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            # output = torch.gather(mat_scaled.repeat(batch_size,self.k-1), dim=1, index=_input[:,1:self.k])
            output = mat_scaled[_input[:,1:self.k].flatten()].view(batch_size, self.k-1)
            output = mat_scaled[_input[:, 0]] * torch.prod(output, 1)
            output = output * self.multiI
            
        elif opt == "A":
            sos = self.sos_softplus(self.sos).view(self.init_row, self.init_col)
            mat = torch.matmul(sos, sos.T)
            mat = mat.view(-1)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            # output = torch.gather(mat_scaled.repeat(batch_size,self.k-1), dim=1, index=_input[:,1:self.k])
            output = mat_scaled[_input[:,1:self.k].flatten()].view(batch_size, self.k-1)
            output = mat_scaled[_input[:, 0]] * torch.prod(output, 1)
            output = output * self.multiA
            
        elif opt == "L":
            sos = self.sos_softplus(self.sos).view(self.init_row, self.init_col)
            mat = torch.matmul(sos.T, sos)
            mat = mat.view(-1)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            # output = torch.gather(mat_scaled.repeat(batch_size,self.k-1), dim=1, index=_input[:,1:self.k])
            output = mat_scaled[_input[:,1:self.k].flatten()].view(batch_size, self.k-1)
            output = mat_scaled[_input[:, 0]] * torch.prod(output, 1)
            output = output * self.multiL

        # output_scaled = output / torch.sqrt(torch.sum(torch.square(mat)) ** self.k)
        return output
    
    def get_sq_sum(self, opt):
        if opt == "I":
            mat = self.sos_softplus(self.sos)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            sq_sum = (torch.sum(torch.square(mat_scaled)) ** self.k) * (self.multiI ** 2)
        elif opt == "A":
            sos = self.sos_softplus(self.sos).view(self.init_row, self.init_col)
            mat = torch.matmul(sos, sos.T).view(-1)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            sq_sum = (torch.sum(torch.square(mat_scaled)) ** self.k)  * (self.multiA ** 2)
        elif opt == "L":
            sos = self.sos_softplus(self.sos).view(self.init_row, self.init_col)
            mat = torch.matmul(sos.T, sos).view(-1)
            mat_scaled = mat / torch.sqrt(torch.sum(torch.square(mat)))
            sq_sum = (torch.sum(torch.square(mat_scaled)) ** self.k)  * (self.multiL ** 2)
            
        return sq_sum
    
# Total model that includes permutation
class KroneckerFitting:
    def __init__(self, graph, init_row, init_col, k, device, sample_weight, saveflag, testflag, use_multi=True):
        self.init_row, self.init_col = init_row, init_col        
        self.k = k
        self.device = device
        self.graph = graph
        self.saveflag = saveflag
        self.testflag = testflag
        
        # Initialize device
        use_cuda = torch.cuda.is_available()
        self.i_device = torch.device("cuda:" + str(self.device[0]) if use_cuda else "cpu")  
        print(self.i_device)
        
        self.node_bases = init_row ** ((self.k-1) - torch.arange(self.k)).to(self.i_device)
        self.edge_bases = init_col ** ((self.k-1) - torch.arange(self.k)).to(self.i_device)
        self.num_row, self.num_col = init_row**k, init_col**k          
        self.sample_weight = sample_weight
        self.use_multi = use_multi
    
    # Initialize the deep learning model
    def init_model(self, data_type="double"):
        self.model = KroneckerModel(self.init_row, self.init_col, self.k, self.graph.sq_sum, self.graph.adj_sq_sum, self.graph.line_sq_sum, use_multi=self.use_multi, model_type=data_type)      
        if data_type == "double": self.model.double()
        elif data_type == "float": self.model.float()
        self.data_type = data_type
        # if torch.cuda.is_available() and len(self.device) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids = self.device)                        
        self.model = self.model.to(self.i_device)
        print(f"The number of params:{ sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def init_permutation(self):        
        self.node_perm = torch.randperm(self.num_row, dtype=torch.long).to(self.i_device)
        self.edge_perm = torch.randperm(self.num_col, dtype=torch.long).to(self.i_device)

    def set_permutation(self, row_perm_file, col_perm_file):                      
        # Read permutations from files
        with open(row_perm_file, 'r') as ff:            
            init_row_perm = [int(val) for val in ff]            
        with open(col_perm_file, 'r') as ff:
            init_col_perm = [int(val) for val in ff]        
                
        print(f'row min: {min(init_row_perm)}, \
              row max: {max(init_row_perm)}, row avg:{sum(init_row_perm) / len(init_row_perm)}')
        print(f'col min: {min(init_col_perm)}, \
              col max: {max(init_col_perm)}, row avg:{sum(init_col_perm) / len(init_col_perm)}')
        
        # Set the node permutatoin (node -> row)
        while len(init_row_perm) < self.num_row: init_row_perm.append(len(init_row_perm))
        while len(init_col_perm) < self.num_col: init_col_perm.append(len(init_col_perm))
            
        self.node_perm = torch.LongTensor(init_row_perm).to(self.i_device)
        self.edge_perm = torch.LongTensor(init_col_perm).to(self.i_device)        
        
    def L2_loss(self, is_train, batch_size, opt):        
        loss = self.model.get_sq_sum(opt)
        if opt == "I":
            search_space = self.graph.real_num_nonzero
        elif opt == "A":
            search_space = len(self.graph.adj_row_idx)
        elif opt == "L":
            search_space = len(self.graph.line_row_idx)
            
        for i in range(0, search_space, batch_size):    
            # Extract nodes and edges
            curr_batch_size = min(batch_size, search_space - i)
            # Convert to lstm inputs
            if opt == "I":
                nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
                nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)
                curr_val = torch.FloatTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
                # row_idx = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
                # col_idx = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
                row_idx = torch.div(self.node_perm[nodes].unsqueeze(1), self.node_bases, rounding_mode="trunc") % self.init_row
                col_idx = torch.div(self.edge_perm[edges].unsqueeze(1), self.edge_bases, rounding_mode="trunc") % self.init_col
                inputs = row_idx * self.init_col + col_idx
            elif opt == "A":
                nodes1, nodes2 = self.graph.adj_row_idx[i:i+curr_batch_size], self.graph.adj_col_idx[i:i+curr_batch_size]
                if self.saveflag:
                    curr_val = self.graph.load_values(i, opt)
                else:
                    tmp = (self.graph.inci_csr[nodes1,:].todense() @ self.graph.inci_csr[nodes2,:].T.todense())
                    curr_val = tmp.diagonal()
                nodes1, nodes2 = torch.LongTensor(nodes1).to(self.i_device), torch.LongTensor(nodes2).to(self.i_device)
                curr_val = torch.from_numpy(curr_val).to(self.i_device)
                # row_idx = self.node_perm[nodes1].unsqueeze(1) // self.node_bases % self.init_row
                # col_idx = self.node_perm[nodes2].unsqueeze(1) // self.node_bases % self.init_row
                row_idx = torch.div(self.node_perm[nodes1].unsqueeze(1), self.node_bases, rounding_mode="trunc") % self.init_row
                col_idx = torch.div(self.node_perm[nodes2].unsqueeze(1), self.node_bases, rounding_mode="trunc") % self.init_row
                inputs = row_idx * self.init_row + col_idx
            elif opt == "L":
                edges1, edges2 = self.graph.line_row_idx[i:i+curr_batch_size], self.graph.line_col_idx[i:i+curr_batch_size]  
                if self.saveflag:
                    curr_val = self.graph.load_values(i, opt)
                else:
                    tmp = (self.graph.inci_csr[:, edges1].T.todense() @ self.graph.inci_csr[:, edges2].todense())
                    curr_val = tmp.diagonal()
                edges1, edges2 = torch.LongTensor(edges1).to(self.i_device), torch.LongTensor(edges2).to(self.i_device) 
                curr_val = torch.from_numpy(curr_val).to(self.i_device)
                # row_idx = self.edge_perm[edges1].unsqueeze(1) // self.edge_bases % self.init_col
                # col_idx = self.edge_perm[edges2].unsqueeze(1) // self.edge_bases % self.init_col
                row_idx = torch.div(self.edge_perm[edges1].unsqueeze(1), self.edge_bases, rounding_mode="trunc") % self.init_col
                col_idx = torch.div(self.edge_perm[edges2].unsqueeze(1), self.edge_bases, rounding_mode="trunc") % self.init_col
                inputs = row_idx * self.init_col + col_idx
            
            samples = self.model(inputs, opt)     
            # print(samples.shape)
            # print(curr_val.shape)
            curr_loss = (torch.square(samples - curr_val) - torch.square(samples)).sum()                           
            loss += curr_loss.item()   
            
        # if is_train:
        #     loss.backward()

        return loss
    
    def write_matrix(self, batch_size, file_name, sv_print_flag=False):
        num_printed_entry = 0
        num_entry = self.graph.num_row * self.graph.num_col
        pbar = tqdm(total=num_entry)
        vals = np.zeros(num_entry)
        graph_row, graph_col, graph_val = np.array(self.graph.row_idx), np.array(self.graph.col_idx), np.array(self.graph.val)
        vals[graph_row*self.graph.num_col + graph_col] = graph_val
        
        _se = 0.
        sample_rows, sample_cols, sample_vals = [], [], []
        with torch.no_grad():
            with open(file_name, 'w') as f:            
                while num_printed_entry < num_entry:            
                    if batch_size > num_entry - num_printed_entry: 
                        batch_size = num_entry - num_printed_entry

                    # Build LSTM inputs
                    curr_rows, curr_cols = torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.i_device), torch.arange(num_printed_entry, num_printed_entry + batch_size, dtype=torch.long).to(self.i_device)                    
                    curr_vals = torch.tensor(vals[num_printed_entry:num_printed_entry  + batch_size]).to(self.i_device)
                    curr_rows, curr_cols = curr_rows//self.graph.num_col, curr_cols%self.graph.num_col                    
                    curr_rows, curr_cols = self.node_perm[curr_rows], self.edge_perm[curr_cols]            

                    # row_idx = curr_rows.unsqueeze(1) // self.node_bases % self.init_row # kro-rows
                    # col_idx = curr_cols.unsqueeze(1) // self.edge_bases % self.init_col # kro-cols   
                    row_idx = torch.div(curr_rows.unsqueeze(1), self.node_bases, rounding_mode="trunc") % self.init_row
                    col_idx = torch.div(curr_cols.unsqueeze(1), self.edge_bases, rounding_mode="trunc") % self.init_col  
                    
                    # Get lstm outputs
                    inputs = row_idx * self.init_col + col_idx 
                    samples = self.model(inputs, "I")
                    #print(samples.shape)
                    _se += torch.sum((samples - curr_vals)**2).item()
                    
                    samples = samples.detach().cpu().numpy()
                    for i in range(batch_size):
                        r, c, v = curr_rows[i], curr_cols[i], samples[i]
                        if random.random() < v:
                            sample_rows.append(r.cpu())
                            sample_cols.append(c.cpu())
                            sample_vals.append(v)
                            f.write(f'{c} {r} {v}\n')
                        # if ((i+num_printed_entry+1)%self.graph.num_col == 0):
                        #     f.write('\n')

                    num_printed_entry += batch_size            
                    pbar.update(batch_size)
        
            print(f'loss for the parts correpsond to real matrix {_se}')
        
            if sv_print_flag:
                criterion = torch.nn.MSELoss()
                sampled_inci = coo_matrix((sample_vals, (sample_rows, sample_cols)), shape=(self.num_row, self.num_col))
                _, sampled_svds,_ = scipy.sparse.linalg.svds(sampled_inci, k=min(min(self.num_row - 1, self.num_col - 1), 5000))
                sampled_svds = sorted(sampled_svds)[::-1]
                sampled_svds = torch.FloatTensor(sampled_svds)

                min_len = min(sampled_svds.shape[0], self.graph.svds.shape[0])
                loss = criterion(sampled_svds[:min_len], self.graph.svds[:min_len]).detach().item()

                print(sampled_svds[0], self.graph.svds[0])
                
                return loss
            
            else:
                
                return 0.0
        

    '''
        Compute the L2 loss in a naive manner
        batch_size: batch size for nonzeros       
    def loss_naive(self, batch_size):
        loss_sum = 0.0   
        if len(self.device) > 1: self.model.module.eval()
        else: self.model.eval()
        with torch.no_grad():
            # Handle zero early, build index for row
            curr_row = self.node_perm.unsqueeze(1) // self.node_bases % self.init_row
            curr_row_pad = self.init_row * torch.ones([self.num_row, self.second_k], dtype=torch.long).to(self.i_device)    
            curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)        
            # Assume entries are all zero, compute loss
            for i in tqdm(range(self.num_col)):
                curr_col = self.edge_perm[i].unsqueeze(-1) // self.edge_bases % self.init_col
                inputs = curr_row * self.init_col + curr_col                                                 
                outputs = self.model(inputs)            
                curr_loss = (torch.square(outputs)).sum()
                loss_sum += curr_loss.item()

            # Handle non-zero part            
            for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):
                # Extract nodes and edges
                curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
                nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
                nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                            
                curr_val = torch.FloatTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
                
                # Convert to row and col
                curr_row = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
                curr_col = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
                curr_row_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
                curr_row = torch.cat((curr_row, curr_row_pad), dim=-1)

                # Compute loss
                inputs = curr_row * self.init_col + curr_col      
                samples = self.model(inputs)
                curr_loss = (torch.square(samples - curr_val) - torch.square(samples)).sum()      
                    
                loss_sum += curr_loss.item()       
        return loss_sum 
    '''
    
    def sample_node_batches(self, batch_size):        
        '''
            Make mapping
            samples: permutation for random mapping
            pair_idx: node idx -> pair idx                
            mapping: node idx -> node idx
        '''
        # print("perm")
        start_time = time.time()
        sampled_n = self.num_row - 1
        # target_digit = 1 << int(random.random() * self.k)
        target_digit = np.random.randint(self.num_row)
        target_digit = target_digit & (-target_digit)
        if target_digit == 0: target_digit = 1
        # target_digit = self.num_row // 2
        # print("target", bin(target_digit))
        
        sampled_kro_rows = torch.randperm((sampled_n) // 2).to(self.i_device)
        # sampled_kro_rows = torch.arange((sampled_n) // 2).to(self.i_device)
        # print("  ".join([str(bin(t)) for t in sampled_kro_rows[:15]]))
        tmp = (sampled_kro_rows*2) - ((sampled_kro_rows*2) & ((2*target_digit) - 1)) 
        # print("  ".join([str(bin(t)) for t in tmp[:15]]))
        sampled_kro_rows = tmp +  (sampled_kro_rows & (target_digit - 1))
        # print("  ".join([str(bin(t)) for t in sampled_kro_rows[:15]]))
        if torch.all(((sampled_kro_rows & target_digit) == 0)) == False:
            print(bin(target_digit))
            print(sampled_kro_rows)
            for r in sampled_kro_rows:
                if (r & target_digit) != 0:
                    print(bin(r), bin(target_digit))
        assert torch.all(((sampled_kro_rows & target_digit) == 0))
        # sampled_kro_rows += target_digit * torch.ones(sampled_n // 2, dtype=torch.long).to(self.i_device)
        sampled_kro_rows += (target_digit * (torch.rand(sampled_n // 2).to(self.i_device) < 0.5).long())
#         print("  ".join([str(bin(t)) for t in sampled_kro_rows[:15]]))
#         print()
        
        # check = np.zeros(self.num_row, dtype=np.int64)
        # for si in range(sampled_n // 2):
        #     s = sampled_kro_rows[si]
        #     if s >= self.num_row:
        #         continue
        #     if check[s] > 0:
        #         a = si
        #         print(bin(a*2), bin(a*2 - ((a*2) & (2*target_digit - 1))), bin(a*2 - ((a*2) & (2*target_digit - 1)) + (a & (target_digit-1))), bin(a*2 - ((a*2) & (2*target_digit - 1)) + (a & (target_digit-1)) + target_digit))
        #         print(bin(sampled_kro_rows[a]), sampled_kro_rows[a], a)
        #         b = check[s]
        #         print(bin(b*2), bin(b*2 - ((b*2) & (2*target_digit - 1))), bin(b*2 - ((b*2) & (2*target_digit - 1)) + (b & (target_digit-1))), bin(b*2 - ((b*2) & (2*target_digit - 1)) + (b & (target_digit-1)) + target_digit))
        #         print(bin(sampled_kro_rows[b]), sampled_kro_rows[b], b)
        #     assert check[s] == 0
        #     check[s] = si
        # filtering
        sampled_kro_rows = sampled_kro_rows[sampled_kro_rows < sampled_n]
        sampled_kro_rows = sampled_kro_rows[((sampled_kro_rows^target_digit) + 1) < self.num_row]
        # for r in sampled_kro_rows:
        #     if r.item() >= self.num_row:
        #         print(bin(r))
        sampled_n = sampled_kro_rows.shape[0]
        if sampled_n == 0:
            return 0.
        
        # check = np.zeros(self.num_row)
        # for pi in range(self.num_row):
        #     if check[self.node_perm[pi]] > 0:
        #         print(self.node_perm[:pi], self.node_perm[pi])
        #     assert check[self.node_perm[pi]] == 0
        #     check[self.node_perm[pi]] += 1
        perm_inv = np.zeros(self.num_row)
        perm_inv[self.node_perm.cpu().numpy()] = np.arange(self.num_row)
        perm_inv = torch.LongTensor(perm_inv).to(self.i_device)
        
        for pi in range(self.num_row):
            if perm_inv[self.node_perm[pi]] != pi:
                print(self.node_perm[pi], pi, perm_inv[self.node_perm[pi]])
            assert perm_inv[self.node_perm[pi]] == pi
            
#         check = np.zeros(self.num_row, dtype=np.int64)
#         for si in range(sampled_n):
#             sa = sampled_kro_rows[si] + 1
#             sb = (sampled_kro_rows[si]^target_digit) + 1
#             if check[sa] != 0:
#                 print(bin(sa), bin(sb), si)
#                 ai = check[sa]
#                 psa = sampled_kro_rows[ai] + 1
#                 psb = (sampled_kro_rows[ai]^target_digit) + 1
#                 print(bin(psa), bin(psb), ai)
#             elif check[sb] != 0:
#                 print(bin(sa), bin(sb), si)
#                 ai = check[sb]
#                 psa = sampled_kro_rows[ai] + 1
#                 psb = (sampled_kro_rows[ai]^target_digit) + 1
#                 print(bin(psa), bin(psb), ai)
                
#             assert check[sa] == 0 and check[sb] == 0
#             check[sa] = si
#             check[sb] = si
        
        sampled_rows = perm_inv[sampled_kro_rows + 1]
        sampled_rows_mate = perm_inv[(sampled_kro_rows^target_digit) + 1]
        
#         check = np.zeros(self.num_row, dtype=np.int64)
#         for si in range(sampled_n):
#             s = sampled_rows[si]
#             assert check[s] == 0
#             check[s] = si
#             s = sampled_rows_mate[si]
#             if check[s] != 0:
#                 print(bin(sampled_rows[si]), bin(sampled_rows_mate[si]))
#                 print(bin(sampled_kro_rows[si] + 1), bin((sampled_kro_rows[si]^target_digit) + 1), bin(si))
#                 sa = check[s]
#                 print(bin(sampled_rows[sa]), bin(sampled_rows_mate[sa]))
#                 print(bin(sampled_kro_rows[sa] + 1), bin((sampled_kro_rows[sa]^target_digit) + 1), bin(sa))
#             assert check[s] == 0
#             check[s] = si
        
        #print(f'node prepare min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        h_fn = torch.randperm(self.num_col).to(self.i_device) + 1
        maxh = torch_scatter.scatter(h_fn[self.graph.col_idx], torch.LongTensor(self.graph.row_idx).to(self.i_device), dim=-1, dim_size=self.num_row, reduce='max')
        maxh = maxh[sampled_rows]
        #print(f'node min hashing: {time.time() - start_time}') 
        
        start_time = time.time()
        maxh_last = torch_scatter.scatter(torch.arange(sampled_n).to(self.i_device), maxh, dim=-1, dim_size=self.num_col + 1, reduce='max')
        maxh_count = torch.bincount(maxh, minlength=self.num_col + 1)
        odd_elements = maxh_last[(maxh_count % 2) > 0]
        maxh[odd_elements] = (self.num_col + 1)
        # maxh_count = torch.bincount(maxh, minlength=self.num_col + 1)
        maxh = maxh * (sampled_n) + torch.arange(sampled_n).to(self.i_device)
        
        unmatched_n = odd_elements.shape[0]
        if unmatched_n > 0:
            unmatched = torch.zeros(2*unmatched_n, dtype=torch.long).to(self.i_device)
            odd_elements = odd_elements[torch.randperm(unmatched_n)]
            unmatched[0::2], unmatched[1::2] = sampled_rows[odd_elements], sampled_rows_mate[odd_elements]

        start_time = time.time()
        _, sorted_idx = torch.sort(maxh)
        matched_n = sampled_n-unmatched_n
        samples = torch.zeros(2*matched_n, dtype=torch.long).to(self.i_device)
        samples[0::4], samples[1::4] = sampled_rows[sorted_idx[0:matched_n:2]], sampled_rows_mate[sorted_idx[1:matched_n:2]]
        samples[2::4], samples[3::4] = sampled_rows[sorted_idx[1:matched_n:2]], sampled_rows_mate[sorted_idx[0:matched_n:2]]
        if unmatched_n > 0:
            samples = torch.cat((samples, unmatched), dim=0)
        check = torch.unique(samples)
        assert check.shape[0] == 2*sampled_n
        
        pair_idx = torch.ones(self.num_row, dtype=torch.long).to(self.i_device)
        pair_idx *= (self.num_row - 1)
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx[samples] = torch.div(pair_idx[samples], 2, rounding_mode="trunc")
        mapping = torch.arange(self.num_row).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1) 
        
        if self.testflag:
            for i in range(self.num_row):
                if i not in samples:
                    assert pair_idx[i] == (self.num_row - 1)
                    assert mapping[i] == i
                else:
                    assert pair_idx[i] < (self.num_row - 1)
                    assert mapping[i] != i

        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)
        sampled_points = torch.rand(sampled_n, dtype=curr_dtype).to(self.i_device)
        prob_before = torch.zeros((self.num_row), dtype=curr_dtype).to(self.i_device)
        prob_after = torch.zeros((self.num_row), dtype=curr_dtype).to(self.i_device)

        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                   
            cols = self.edge_perm[edges]
            col_idx = torch.div(cols.unsqueeze(1), self.edge_bases, rounding_mode='trunc') % self.init_col
            
            curr_pair = pair_idx[nodes]
            curr_val = torch.FloatTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
            
            # Compute prob_before & prob_after                                   
            temp_list = [(self.node_perm[nodes], prob_before), (self.node_perm[mapping[nodes]], prob_after)]
            for (curr_rows, curr_prob) in temp_list:                                                                              
                row_idx = torch.div(curr_rows.unsqueeze(1), self.node_bases, rounding_mode='trunc') % self.init_row                                        
                inputs = row_idx * self.init_col + col_idx         
                outputs = self.model(inputs, "I")
                curr_prob.index_add_(0, curr_pair, -torch.square(outputs) + torch.square(outputs - curr_val))                                                      
        prob_ratios = prob_before - prob_after
        # if (2 * sampled_n) < self.num_row:
        #     if abs(prob_ratios[self.num_row - 1]) > 1e-12:
        #         print(abs(prob_ratios[self.num_row - 1]))
        #     assert abs(prob_ratios[self.num_row - 1]) < 1e-12
        prob_ratios = prob_ratios[:sampled_n]
        prob_ratios.clamp_(min=-(10**15))    
        prob_thre = prob_ratios.clone()
        final_decision = (prob_thre > 0.) # | (sampled_points <= (self.sample_weight * prob_thre).exp())
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        samples = samples.view(-1, 2)[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        self.node_perm[ll], self.node_perm[rr] = self.node_perm[rr].clone(), self.node_perm[ll].clone()
        
        if self.testflag:
            check = torch.zeros(self.num_row)
            for i in range(self.num_row):
                assert check[self.node_perm[i]] == 0
                check[self.node_perm[i]] = 1
                
        return prob_ratios[final_decision].sum().item()
    
    def sample_edge_batches(self, batch_size):
        start_time = time.time()
        sampled_n = self.num_col - 1
        target_digit = np.random.randint(self.num_col)
        target_digit = target_digit & (-target_digit)
        if target_digit == 0: target_digit = 1
        # target_digit = self.num_col // 2
                
        sampled_kro_cols = torch.randperm(sampled_n // 2).to(self.i_device)
        tmp = (sampled_kro_cols * 2) - ((sampled_kro_cols*2) & ((2*target_digit) - 1))
        sampled_kro_cols = tmp + (sampled_kro_cols & (target_digit - 1))
        assert torch.all(((sampled_kro_cols & target_digit) == 0))
        # sampled_kro_cols += target_digit * torch.ones(sampled_n // 2, dtype=torch.long)
        sampled_kro_cols += (target_digit * (torch.rand(sampled_n // 2).to(self.i_device) < 0.5).long())
        # filtering
        sampled_kro_cols = sampled_kro_cols[sampled_kro_cols < sampled_n]
        sampled_kro_cols = sampled_kro_cols[((sampled_kro_cols^target_digit) + 1) < self.num_col]
        sampled_n = sampled_kro_cols.shape[0]
        
        if sampled_n == 0:
            return 0.
        
        perm_inv = np.zeros(self.num_col)
        perm_inv[self.edge_perm.cpu().numpy()] = torch.arange(self.num_col)
        perm_inv = torch.LongTensor(perm_inv).to(self.i_device)
        
        for pi in range(self.num_col):
            if perm_inv[self.edge_perm[pi]] != pi:
                print(self.edge_perm[pi], pi, perm_inv[self.edge_perm[pi]])
            assert perm_inv[self.edge_perm[pi]] == pi
        
        sampled_cols = perm_inv[sampled_kro_cols + 1]
        sampled_cols_mate = perm_inv[(sampled_kro_cols^target_digit) + 1]
        
        start_time = time.time()
        h_fn = torch.randperm(self.num_row).to(self.i_device) + 1
        maxh = torch_scatter.scatter(h_fn[self.graph.row_idx], torch.LongTensor(self.graph.col_idx).to(self.i_device), dim=-1, dim_size=self.num_col, reduce='max')        
        maxh = maxh[sampled_cols]
            
        start_time = time.time()
        maxh_last = torch_scatter.scatter(torch.arange(sampled_n).to(self.i_device), maxh, dim=-1, dim_size=self.num_row + 1, reduce='max')
        maxh_count = torch.bincount(maxh, minlength=self.num_row + 1)
        odd_elements = maxh_last[(maxh_count % 2) > 0] 
        maxh[odd_elements] = (self.num_row + 1)  
        maxh = maxh * (sampled_n) + torch.arange(sampled_n).to(self.i_device) 
        
        unmatched_n = odd_elements.shape[0]
        if unmatched_n > 0:
            unmatched = torch.zeros(2*unmatched_n, dtype=torch.long).to(self.i_device)
            odd_elements = odd_elements[torch.randperm(unmatched_n)]
            unmatched[0::2], unmatched[1::2] = sampled_cols[odd_elements], sampled_cols_mate[odd_elements]
            
            
        staart_time = time.time()
        _, sorted_idx = torch.sort(maxh)        
        # samples: new idx -> cols
        matched_n = sampled_n-unmatched_n
        samples = torch.zeros(2*matched_n, dtype=torch.long).to(self.i_device)
        samples[0::4], samples[1::4] = sampled_cols[sorted_idx[0:matched_n:2]], sampled_cols_mate[sorted_idx[1:matched_n:2]]
        samples[2::4], samples[3::4] = sampled_cols[sorted_idx[1:matched_n:2]], sampled_cols_mate[sorted_idx[0:matched_n:2]]
        if unmatched_n > 0:
            samples = torch.cat((samples, unmatched), dim=0)
        # For debugging
        check = torch.unique(samples)
        assert check.shape[0] == 2 * sampled_n
    
        # pair_idx = torch.arange(self.num_col).to(self.i_device) #samples.clone()
        pair_idx = torch.ones(self.num_col, dtype=torch.long).to(self.i_device)
        pair_idx *= (self.num_col - 1)
        pair_idx[samples] = torch.arange(len(samples)).to(self.i_device)
        pair_idx[samples] = torch.div(pair_idx[samples], 2, rounding_mode="trunc")
        assert torch.all(pair_idx[samples] < (len(samples) // 2))
        mapping = torch.arange(self.num_col).to(self.i_device)
        mapping[samples] = torch.stack((samples[1::2], samples[0::2]), dim=1).view(-1)
        
        if self.testflag:
            for i in range(self.num_col):
                if i not in samples:
                    assert pair_idx[i] == (self.num_col - 1)
                    assert mapping[i] == i
                else:
                    assert pair_idx[i] < (self.num_col - 1)
                    assert mapping[i] != i

        start_time = time.time()
        if self.data_type == "double": curr_dtype = torch.double                       
        elif self.data_type == "float": curr_dtype = torch.float        
        else: assert(False)        
        sampled_points = torch.rand(sampled_n, dtype=curr_dtype).to(self.i_device)
        prob_before = torch.zeros((self.num_col), dtype=curr_dtype).to(self.i_device)
        prob_after = torch.zeros((self.num_col), dtype=curr_dtype).to(self.i_device)

        # Compute the change of loss                                
        # non zero part (No need to consider zero part because prob_before and prob_after are the same when all entries are zero)
        num_nnz = self.graph.real_num_nonzero
        for i in range(0, num_nnz, batch_size):
            # Build an input for the current batch
            curr_batch_size = min(batch_size, num_nnz - i)
            nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
            nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                   
            rows = self.node_perm[nodes]
            row_idx = torch.div(rows.unsqueeze(1), self.node_bases, rounding_mode='trunc') % self.init_row
            
            curr_pair = pair_idx[edges]
            curr_val = torch.FloatTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)
            
            # Compute prob_before & prob_after                                   
            temp_list = [(self.edge_perm[edges], prob_before), (self.edge_perm[mapping[edges]], prob_after)]
            for (curr_cols, curr_prob) in temp_list:                                                                              
                col_idx = torch.div(curr_cols.unsqueeze(1), self.edge_bases, rounding_mode='trunc') % self.init_col                                        
                inputs = row_idx * self.init_col + col_idx         
                outputs = self.model(inputs, "I")
                curr_prob.index_add_(0, curr_pair, -torch.square(outputs) + torch.square(outputs - curr_val))                                                      
        prob_ratios = prob_before - prob_after
        # if sampled_n * 2 < self.num_col:
        #     if abs(prob_ratios[self.num_col - 1]) > 1e-12:
        #         print(abs(prob_ratios[self.num_col - 1]))
        #     assert abs(prob_ratios[self.num_col - 1]) < 1e-12
        sampled_ratio = prob_ratios[:sampled_n]
        sampled_ratio = sampled_ratio.clamp_(min=-(10**15))
        final_decision = (sampled_ratio > 0.) # | (sampled_points <= (self.sample_weight*sampled_ratio).exp())
        
        start_time = time.time()
        if final_decision.long().sum().item() == 0:
            return 0.
        
        view_sample = samples.view(-1, 2)
        samples = view_sample[final_decision].view(-1)
        ll, rr = samples[0::2], samples[1::2]
        self.edge_perm[ll], self.edge_perm[rr] = self.edge_perm[rr].clone(), self.edge_perm[ll].clone()
        
        if self.testflag:
            check = torch.zeros(self.num_col)
            for i in range(self.num_col):
                assert check[self.edge_perm[i]] == 0
                check[self.edge_perm[i]] = 1
        
        return  sampled_ratio[final_decision].sum().item()

    '''
        Compute the L2 loss in an efficient manner   
        batch_size: batch size for nonzeros        
    def check_distribution(self, batch_size, save_file):        
        with torch.no_grad():
            # nnzs = list(zip(graph.adjcoo.row, graph.adjcoo.col))
            if len(self.device) > 1: sq_sum = (self.model.module.level_sq_sum)**(self.k + self.second_k)
            else: sq_sum = (self.model.level_sq_sum)**(self.k + self.second_k)
            sq_sum = sq_sum.item()   
            max_entry = max(self.graph.val)            
            loss_list = [0. for _ in range(max_entry + 1)]
            num_list = [0 for _ in range(max_entry + 1)]
            
            #for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):
            for i in tqdm(range(0, self.graph.real_num_nonzero, batch_size)):    
                # Extract nodes and edges
                curr_batch_size = min(batch_size, self.graph.real_num_nonzero - i)
                nodes, edges = self.graph.row_idx[i:i+curr_batch_size], self.graph.col_idx[i:i+curr_batch_size]
                nodes, edges = torch.LongTensor(nodes).to(self.i_device), torch.LongTensor(edges).to(self.i_device)                          
                curr_val = torch.FloatTensor(self.graph.val[i:i+curr_batch_size]).to(self.i_device)

                # Convert to lstm inputs
                row_idx = self.node_perm[nodes].unsqueeze(1) // self.node_bases % self.init_row
                col_idx = self.edge_perm[edges].unsqueeze(1) // self.edge_bases % self.init_col
                row_idx_pad = self.init_row * torch.ones([curr_batch_size, self.second_k], dtype=torch.long).to(self.i_device)        
                row_idx = torch.cat((row_idx, row_idx_pad), dim=-1)

                # Correct non-zero parts
                inputs = row_idx * self.init_col + col_idx
                samples = self.model(inputs)
                #print(f'sample shape: {samples.shape}, value sahpe: {curr_val.shape}')      
                sq_sum -= torch.square(samples).sum().item()
                
                for i in range(1, max_entry + 1):
                    curr_idx = curr_val == i
                    loss_list[i] += torch.square(samples[curr_idx] - curr_val[curr_idx]).sum().item()
                    num_list[i] += curr_idx.long().sum().item()
                            
            loss_list[0] = sq_sum
            num_list[0] = self.num_row * self.num_col - self.graph.real_num_nonzero
            with open(save_file + ".txt", 'w') as f:
                for i in range(max_entry + 1):            
                    #print(f'{loss_list[i]}')
                    f.write(f'{num_list[i]}\t{loss_list[i]}\n')
    '''
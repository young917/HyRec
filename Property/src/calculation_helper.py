import os
import sys
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from itertools import chain
import networkx as nx
from numpy.linalg import svd
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs, norm
from scipy.stats import skew, kendalltau
import snap
import time
import math
import copy

# ----------------------------------------------------------------------------------------------

def find_svs(inputpath, outputdir):
    outputpath = outputdir + "sv.txt"
    node2edges = defaultdict(list)
    nodename2index = {}
    node_index = 0
    number_of_nodes = 0
    number_of_edges = 0

    print(inputpath)

    with open(inputpath, "r") as f:
        for idx, line in enumerate(f.readlines()):
            line = line[:-1] # strip enter
            nodes = line.split(",")
            for i in range(len(nodes)):
                node = nodes[i]
                if node not in nodename2index:
                    nodename2index[node] = node_index
                    node_index += 1
                nodes[i] = nodename2index[node]
            for v in nodes:
                node2edges[int(v)].append(idx)
            number_of_edges += 1
    number_of_nodes = len(node2edges.keys())

    if number_of_edges == 0:
        print("Empty Input")
        return -1, -1, -1

    dim = min(number_of_edges, number_of_nodes)
    s = -1

    flag = False
    if "tags-" in inputpath:
        flag = True
        dim = min(dim - 1, 1000)
    elif "threads-" in inputpath:
        flag = True
        dim = min(dim - 1, 1000)
    elif "coauth-" in inputpath:
        flag = True
        dim = min(dim - 1, 500)

    if flag is False:
        # Compute full rank singular values
        try:
            incident_matrix = np.zeros(shape=(number_of_nodes, number_of_edges), dtype=np.byte)
            for v in node2edges.keys():
                for edge_idx in node2edges[v]:
                    incident_matrix[v,edge_idx] = 1          
            s = svd(incident_matrix, compute_uv=False)
            s = sorted(list(s), reverse=True)
            assert len(s) == dim
            nonzero_s = []
            for i in range(1,dim):
                assert s[i-1] >= s[i]
            for i in range(dim):
                if s[i] != 0:
                    nonzero_s.append(s[i])
            with open(outputpath, "w") as f:
                for _sv in nonzero_s:
                    f.write(str(_sv) + "\n")
        except:
            rows, cols = zip(*chain.from_iterable([[(v, edge_idx) for edge_idx in node2edges[v]] for v in node2edges.keys()]))
            nnz = len(rows)
            incident_matrix = coo_matrix((np.ones(nnz), (rows, cols)), shape=(number_of_nodes, number_of_edges))
            sum_of_squares = norm(incident_matrix, 'fro') ** 2
            rank = min(num , dim - 1)
            _, s, _ = svds(incident_matrix.tocsc(), k=rank)
            last_sv_square = sum_of_squares - sum([_s * _s for _s in s])
            last_sv = math.sqrt(last_sv_square)
            s = list(s)
            s.append(last_sv)
            assert len(s) == dim
            s = sorted(s, reverse=True)
            for i in range(1,dim):
                assert s[i-1] >= s[i]
            nonzero_s = []
            for i in range(dim):
                if s[i] != 0:
                    nonzero_s.append(s[i])
            with open(outputpath, "w") as f:
                for _sv in nonzero_s:
                    f.write(str(_sv) + "\n")

    else:
        # Compute top-dim singular values
        rows, cols = zip(*chain.from_iterable([[(v, edge_idx) for edge_idx in node2edges[v]] for v in node2edges.keys()]))
        nnz = len(rows)
        incident_matrix = coo_matrix((np.ones(nnz), (rows, cols)), shape=(number_of_nodes, number_of_edges))
        _, s, _ = svds(incident_matrix.tocsc(), k=dim)
        s = list(s)
        s = sorted(s, reverse=True)
        for i in range(1,dim):
            assert s[i-1] >= s[i]
        nonzero_s = []
        for i in range(dim):
            if s[i] != 0:
                nonzero_s.append(s[i])
        with open(outputpath, "w") as f:
            for _sv in nonzero_s:
                f.write(str(_sv) + "\n")

    return s, dim

def sv_dist(_list_sv, dim, answer_max_portion):
    list_sv = copy.deepcopy(_list_sv)
    singular_values = {}
    
    if answer_max_portion != -1:
        number_of_required_svs = math.ceil(dim * answer_max_portion)
        list_sv = list_sv[:number_of_required_svs]
    denom = sum([_s * _s for _s in list_sv])
    until = 0.0
    total = dim
    for idx in range(len(list_sv)):
        proportion = (idx + 1) / total
        until += list_sv[idx] ** 2
        singular_values[proportion] = until / denom

    return singular_values

# ----------------------------------------------------------------------------------------------

RESULTDIR = "../results/"

parser = argparse.ArgumentParser()
parser.add_argument('--inputpath', required=False)
parser.add_argument('--outputdir', required=False)


parser.add_argument('--effdiam', required=False, action='store_true')
parser.add_argument('--sv', required=False, action='store_true')


args = parser.parse_args()
outputdir = args.outputdir
print("Start " + outputdir)

appx_flag = False
appx_flag2 = False
for dataname in ["threads-ask-ubuntu", "tags-ask-ubuntu", "tags-math-sx"]:
    if dataname in args.inputpath:
        appx_flag = True
for dataname in ["coauth-DBLP-full", "threads-math-sx", "coauth-MAG-Geology-full"]:
    if dataname in args.inputpath:
        appx_flag2 = True

entire_nodeset = set([])
entire_number_of_hyperedges = 0
datapath = args.inputpath + ".txt"
datanode2index = {}
with open(datapath, "r") as f:
    for line in f.readlines():
        hyperedge = line[:-1].split(",")
        entire_number_of_hyperedges += 1
        for n in hyperedge:
            if int(n) not in datanode2index:
                datanode2index[int(n)] = len(datanode2index)
            nodeindex = datanode2index[int(n)]
            entire_nodeset.add(nodeindex)
entire_number_of_nodes = len(entire_nodeset)

if args.effdiam:
    inputpath = datapath
    
    pg = snap.TUNGraph.New()
    nodeset = set([])
    number_of_edges = 0

    start_time = time.time()
    with open(inputpath, "r") as f:
        for line in f.readlines():
            line = line[:-1]
            hyperedge = [int(_n) for _n in line.split(",")]
            number_of_edges += 1
            for i, n in enumerate(hyperedge):
                nodeindex = n
                nodeindex = datanode2index[n]
                hyperedge[i] = nodeindex
                if nodeindex not in nodeset:
                    nodeset.add(nodeindex)
                    pg.AddNode(nodeindex)
            # Add clique edges
            if len(hyperedge) != 1: 
                for i in range(0, len(hyperedge)-1):
                    for j in range(i+1, len(hyperedge)):
                        i1, i2 = min(hyperedge[i], hyperedge[j]), max(hyperedge[i], hyperedge[j])
                        ret = pg.AddEdge(i1, i2)
        num_nodes = len(nodeset)
        if appx_flag:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 1000, False)
        elif appx_flag2:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 100, False)
        else:
            effective_diameter = snap.GetBfsEffDiam(pg, num_nodes if num_nodes < 5000 else 5000, False)
        # Save Effective Diameter
        with open(outputdir + "effdiameter.txt", "w") as f:
            f.write(str(effective_diameter) + "\n")
        print("End EffDiameter")

if args.sv:               
    inputpath = datapath
    assert(os.path.isfile(inputpath))
    answer_s, answer_dim = find_svs(inputpath, outputdir)

    print("End Singular Value Dist")

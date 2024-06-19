import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from IPython.display import display
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import argparse
from utils import *

plt.rcParams.update({'font.size': 13})

# Setting -------------------------------------------------------------------------------------------------------------
columns = ["E", "V", 
 "deg score", "deg coef", "sz score", "sz coef",
 "pd score", "pd coef", "its score", "its coef", 
 "cc score", "cc coef", "cch score", "cch coef", 
 "dst score", "dst coef", "ov score", "ov coef",
 "lsv", "sv score", "sv coef", "eff"]

column_mapping = {
    "NumHedge": "E", "NumNode": "V",
    "degree": "deg", "pairdeg": "pd",
    "intersection": "its", "size": "sz",
    "clusteringcoef": "cc", "clusteringcoef_hedge": "cch",
    "density_dist": "dst", "overlapness_dist": "ov",
    "LargestSV": "lsv", "sv": "sv", 
    "effdiam": "eff"
}
xlabeldict = {
    "degree": "Node degree",
    "size": "Hyperedge size",
    "pairdeg": "Degree of node pairs",
    "intersection": "Intersection size",
    "sv": "Rank",
    "svt_u": "Rank",
    "svt_v": "Rank",
    "clusteringcoef_hedge": "Node degree",
    "density_dist": "# of nodes",
    "overlapness_dist": "# of nodes",
    
    "clusteringcoef": "v's degree in CE",
    "wcc": "Rank",
}
ylabeldict = {
    "degree": "PDF",
    "size": "PDF",
    "pairdeg": "PDF",
    "intersection": 'PDF',
    "sv": "Singular value",
    "svt_u": "Singular vector",
    "svt_v": "Singular vector",
    "clusteringcoef_hedge": "# of inter- \n secting pairs",
    "density_dist": "# of hyperedges",
    "overlapness_dist": r"$\sum$ hyperedge " + "\n sizes",
    
    "clusteringcoef": "# triangles at v in CE",
    "wcc": "size of connected component",
}
color = {
    "answer": "black",
    
    "HyperK": "#4daf4a",
    "hypercl": "#e6ab02",
    "hyperlap": "#377eb8",
    "hyperpa": "#984ea3",
    "thera": "#ff7f00",
    "hyperff": "#e41a1c"
}

# -------------------------------------------------------------------------------------------------------------------------
distset = ["degree", "size", "pairdeg", "intersection", "sv", 
           "clusteringcoef_hedge", "density_dist", "overlapness_dist"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str)
args = parser.parse_args()

namelist = [("answer", -1)]
namelist.append(("hyperlap", -1))    
namelist.append(("hypercl", -1))
if os.path.isfile("../results/hyperpa/{}/sv.txt".format(args.dataname)):
    namelist.append(("hyperpa", -1))
for target in ["hyperff", "thera", "HyperK"]:
    with open("ablation_result/{}.pkl".format(target), "rb") as f:
        result = pickle.load(f)
    if args.dataname in result:
        namelist.append((target, result[args.dataname]))

for distname in distset:
    outputpath = "figure/" + args.dataname + "/"
    if os.path.isdir(outputpath) is False:
        os.makedirs(outputpath)
    outputpath += distname + ".jpg"

    plt.figure(figsize=(5.5,5), dpi=100)
    for (name, idx) in namelist:
        print(args.dataname, name, idx)
        ret, dist = read_properties(args.dataname, name, idx)
        if name == "answer":
            ret_answer = ret
            dist_answer = dist 
            if distname == "intersection":
                print(dist_answer[distname].keys())

        # PLOT!
        target_dist = dist[distname]
        x = list(target_dist.keys())
        y = [target_dist[_x] for _x in x]
        
        if name in ["answer"]:
            plt.scatter(x, y, label=name, c=color[name], alpha=0.6, s=240)
        elif name != "HyperK":
            plt.scatter(x, y, label=name, c=color[name], alpha=0.7, s=100)
        else:
            plt.scatter(x, y, label=name, c=color[name], alpha=1.0, s= 100)
        
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    
    ax = plt.gca()
    ax.tick_params(labelcolor='#4B4B4B', labelsize=26)
    plt.xlabel(xlabeldict[distname], fontsize=33)
    plt.ylabel(ylabeldict[distname], fontsize=33)

    plt.savefig(outputpath, bbox_inches='tight')
    plt.show()
    plt.close()

print()
print()
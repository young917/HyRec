import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from IPython.display import display
from collections import defaultdict
import matplotlib
import pickle
import matplotlib.pyplot as plt
import os
import argparse
from utils import *

plt.rcParams.update({'font.size': 13})

# Setting -----------------------------------------------------------------------------------------------------------------
columns = ["E", "V", 
 "deg score", "deg coef", "sz score", "sz coef",
 "pd score", "pd coef", "its score", "its coef", 
 "cc score", "cc coef", "cch score", "cch coef", 
 "dst score", "dst coef", "ov score", "ov coef",
 "lsv", "sv score", "sv coef", "eff"]

property_list = ["NumHedge", "NumNode", "degree", "size", "pairdeg", "intersection",
             "clusteringcoef", "clusteringcoef_hedge", "density_dist", "overlapness_dist",
             "LargestSV", "sv", "effdiam"]

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
    "clusteringcoef_hedge": "Node degree",
    "density_dist": "# of nodes",
    "overlapness_dist": "# of nodes",
}

ylabeldict = {
    "degree": "PDF",
    "size": "PDF",
    "pairdeg": "PDF",
    "intersection": 'PDF',
    "sv": "Singular value",
    "clusteringcoef_hedge": "# of inter- \n secting pairs",
    "density_dist": "# of hyperedges",
    "overlapness_dist": r"$\sum$ hyperedge " + "\n sizes",
}

color = {
    1: {
        "answer": "#505050",
        "ext_HyperK": "#66a61e",
        "hyperpa": "#975a97",
        "thera": "#d37601",
        "hyperff": "#9d1641"
    },
    
    0: {
        "answer": "black",
        "ext_HyperK": "#4daf4a",
        "hyperpa": "#984ea3",
        "thera": "#ff7f00",
        "hyperff": "#e41a1c"
    }
}
styledict = {
    0: "dashed",
    1: "solid"
}

markerdict = {
#     1: "^",
    1: "o",
    0: "o"
}

alphadict = {
    1: 0.9,
    0: 1.0
}

# -----------------------------------------------------------------------------------------------------------------
fulldatalist = {
    "email-Enron-half": "email-Enron-full",
    "email-Eu-half": "email-Eu-full",
    "contact-primary-school-half": "contact-primary-school",
    "contact-high-school-half": "contact-high-school",
    "NDC-classes-half": "NDC-classes-full",
    "NDC-substances-half": "NDC-substances-full",
    "tags-ask-ubuntu-half": "tags-ask-ubuntu", 
    "tags-math-sx-half": "tags-math-sx",
    "threads-ask-ubuntu-half": "threads-ask-ubuntu", 
    "threads-math-sx-half": "threads-math-sx",
    "coauth-MAG-Geology-half": "coauth-MAG-Geology-full"
}
halfdatalist = {
    "email-Enron-full" : "email-Enron-half",
    "email-Eu-full" : "email-Eu-half",
    "contact-primary-school" : "contact-primary-school-half",
    "contact-high-school" : "contact-high-school-half",
    "NDC-classes-full" : "NDC-classes-half",
    "NDC-substances-full" : "NDC-substances-half",
    "tags-ask-ubuntu": "tags-ask-ubuntu-half",
    "tags-math-sx": "tags-math-sx-half",
    "threads-ask-ubuntu": "threads-ask-ubuntu-half",
    "threads-math-sx": "threads-math-sx-half",
    "coauth-MAG-Geology-full": "coauth-MAG-Geology-half"
}
def halfindex_2_fullindex(dataname, index):
    assert "half" in dataname

    check_tmp = {}
    d = pd.read_csv("../results/ext_HyperK/{}/output_list_half.txt".format(dataname))
    for irow, row in d.iterrows():
        model_index = row["modelIndex"]
        path = row["modelpath"].split("/")[-2]
        check_tmp[int(model_index)] = path
        
    ret = -1
    fulldata =  fulldatalist[dataname]
    d = pd.read_csv("../results/ext_HyperK/{}/output_list_ext.txt".format(fulldata))
    for irow, row in d.iterrows():
        model_index = row["modelIndex"]
        path = row["modelpath"].split("/")[-3]
        if path == check_tmp[int(index)]:
            ret = model_index
            break
            
    return ret

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, default="email-Eu-half", help="should be ended by half")
args = parser.parse_args()

namelist = [("answer", -1)]
if os.path.isfile("../results/hyperpa/{}/effdiameter.txt".format(args.dataname)):
    namelist.append(("hyperpa", -1))
    
for target in ["hyperff", "thera"]:
    with open("ablation_result/{}.pkl".format(target), "rb") as f:
        result = pickle.load(f)
    if args.dataname in result:
        namelist.append((target, result[args.dataname]))
for target in ["HyperK"]:
    with open("ablation_result/{}.pkl".format(target), "rb") as f:
        result = pickle.load(f)
    if args.dataname in result:
        namelist.append(("ext_" + target, result[args.dataname]))

distset = ["degree", "size", "pairdeg", "intersection", "sv", 
           "clusteringcoef_hedge", "density_dist", "overlapness_dist"]
for distname in distset:
    min_x = 1
    max_x = 0
    min_y = 1e+12
    max_y = 0

    for di, cur_dataname in enumerate([fulldatalist[args.dataname], args.dataname]):
        outputpath = "./figure/" + args.dataname + "/"
        if os.path.isdir(outputpath) is False:
            os.makedirs(outputpath)
        if di == 1:
            outputpath += distname + "_past.jpg"
        else:
            outputpath += distname + "_pred.jpg"
        
        print(outputpath)

        plt.figure(figsize=(5.5,5), dpi=100)
        for item in namelist:
            (name, idx) = item
                
            if di == 1:
                ret, dist = read_properties(args.dataname, name, idx)
            else:
                if name in ["hyperpa", "hyperff", "thera"]:
                    ret, dist = read_properties(args.dataname, "ext_" + name, idx)
                elif name == "ext_HyperK":
                    full_dataname = fulldatalist[args.dataname]
                    full_idx = halfindex_2_fullindex(args.dataname, idx)
                    ret, dist = read_properties(full_dataname, name, full_idx)
                else:
                    full_dataname = fulldatalist[args.dataname]
                    ret, dist = read_properties(full_dataname, name, idx)
                
            if name == "answer":
                ret_answer = ret
                dist_answer = dist 

            # PLOT!
            target_dist = dist[distname]
            x = list(target_dist.keys())
            y = [target_dist[_x] for _x in x]
            
            if di == 0 and max(x) > max_x:
                max_x = max(x)
            if distname == "sv" and di == 0:
                if max(y) > max_y:
                    max_y = max(y)
                if min(y) < min_y:
                    min_y = min(y)
            
            if name in ["answer"]:
                plt.scatter(x, y, label=name + "_" + str(di), c=color[di][name], alpha=alphadict[di], s=240, marker=markerdict[di], zorder=2)
            elif name != "ext_HyperK":
                plt.scatter(x, y, label=name + "_" + str(di), c=color[di][name], alpha=alphadict[di], s=100, marker=markerdict[di], zorder=2)
            else:
                plt.scatter(x, y, label=name + "_" + str(di), c=color[di][name], alpha=1.0, s= 100, marker=markerdict[di], zorder=2)
            
        ax = plt.gca()
        ax.set_xlim((min_x / 2, max_x * (2**0.5)))
        
        if distname == "sv":
            ax.set_ylim((min_y / 4, max_y * (2 ** 0.5)))
        
        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

        ax.tick_params(labelcolor='#4B4B4B', labelsize=26)
        plt.xlabel(xlabeldict[distname], fontsize=33)
        plt.ylabel(ylabeldict[distname], fontsize=33)

            
        plt.savefig(outputpath, bbox_inches='tight')
        plt.show()
        plt.close()

print()
print()
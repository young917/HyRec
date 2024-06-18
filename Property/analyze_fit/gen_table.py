import argparse
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import pickle
from utils import *

column_mapping = {
    "NumHedge": "E", "NumNode": "V",
    "degree": "deg", "pairdeg": "pd",
    "intersection": "its", "size": "sz",
    "clusteringcoef": "cc", "clusteringcoef_hedge": "cch",
    "density_dist": "dst", "overlapness_dist": "ov",
    "LargestSV": "lsv", "sv": "sv", 
    "effdiam": "eff"
}

def get_baseline_list(dataname):
    namelist = [("answer", -1), ("hyperlap", -1),  ("hypercl", -1)]
    
    if os.path.isfile("../results/hyperpa/{}/effdiameter.txt".format(dataname)):
        namelist.append(("hyperpa", -1))

    for target in ["hyperff", "thera", "HyperK"]:
        with open("ablation_result/{}.pkl".format(target), "rb") as f:
            result = pickle.load(f)
        namelist.append((target, result[dataname]))
        
    return namelist

def make_dist_table(dataname, outputdir, namelist=None):
    property_list = ["degree", "size", "pairdeg", "intersection","sv", 
                 "clusteringcoef_hedge", "density_dist", "overlapness_dist", "effdiam"] 
    
    if namelist is None:
        namelist = get_baseline_list(dataname)
    
    outputpath = outputdir + dataname + ".txt"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    columns = [column_mapping[prop] for prop in property_list]
    with open(outputpath, "w") as f:
        f.write(",".join(["AlgoName", "AlgoOpt"] + columns) + "\n")
        
    for name, modelindex in namelist:
        ret, dist = read_properties(dataname, name, modelindex)
        if name == "answer":
            ret_answer = ret
            dist_answer = dist   
            continue

        difflist = []
        for prop in property_list:
            
            if prop in ["NumHedge", "NumNode", "LargestSV", "effdiam"]: 
                diff = abs(dist[prop] - dist_answer[prop]) / dist_answer[prop]
            elif prop in ["degree", "size", "pairdeg", "intersection"]:
                diff = get_cumul_dist(dist_answer[prop], dist[prop])
                assert diff >= 0 and diff <= 1
            else:
                intersect_xs = sorted(list(set(list(dist_answer[prop].keys())).intersection(set(list(dist[prop].keys())))))
                if len(intersect_xs) == 0:
                    xs = list(dist[prop].keys())
                    answer_ys = [math.log2(1e-20) for _x in xs]
                    processed_ys = [math.log2(dist[prop][_x]) for _x in xs]
                else:
                    answer_ys = [math.log2(dist_answer[prop][_x]) for _x in intersect_xs]
                    processed_ys = [math.log2(dist[prop][_x]) for _x in intersect_xs]
                diff = get_rmse(answer_ys, processed_ys)
                
            difflist.append(str(diff))

        with open(outputpath, "a") as f:
            algoopt = str(modelindex)
            algoname = name
            f.write(",".join([algoname, algoopt] + difflist) + "\n")
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--outputdir", default="csv/sv_fit/", type=str)
    args = parser.parse_args()
    
    make_dist_table(args.dataname, args.outputdir)
    
    d = pd.read_csv(args.outputdir + args.dataname + ".txt")
    
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff'] #, "mod"] 
    
    d = pd.read_csv(args.outputdir + args.dataname + ".txt")
    target = d[evallist]
    d['avg'] = target.mean(axis=1)
    d.to_csv(args.outputdir + args.dataname + ".txt", index=False)
    
    # Make Ranking Result
    d = pd.read_csv(args.outputdir + args.dataname + ".txt")
    for ename in evallist:
        d[ename] = d[ename].abs().rank(method='min')
    ranks = d[evallist]
    d['avg'] = ranks.mean(axis=1)
    d.to_csv(args.outputdir + args.dataname + "_rank.txt", index=False)
    
    # Rank
    rd = pd.read_csv(args.outputdir + args.dataname + "_rank.txt")
    rd = rd.sort_values(by=["avg"], ascending=True)
    
    print(rd)
    
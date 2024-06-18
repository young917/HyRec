import argparse
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import math
import pickle
from utils import *
from tqdm import tqdm

column_mapping = {
    "NumHedge": "E", "NumNode": "V",
    "degree": "deg", "pairdeg": "pd",
    "intersection": "its", "size": "sz",
    "clusteringcoef": "cc", "clusteringcoef_hedge": "cch",
    "density_dist": "dst", "overlapness_dist": "ov",
    "LargestSV": "lsv", "sv": "sv", 
    "effdiam": "eff"
}

def get_directories(dataname, ablation_target):
    bset=[8, 12, 15]
    pset=[0.5, 0.7, 0.9]
    cset=[2.0, 6.0, 10.0]
    namelist = [("answer", -1)]

    if ablation_target == "hyperff":
        for _fname in os.listdir("../dataset/"):
            if dataname + "_ff" in _fname:
                fname = _fname[:-4]
                tmp = fname.split("_")[2:]
                paramname = "_".join(tmp)
                if os.path.isfile("../results/ext_hyperff/{}/{}/sv.txt".format(dataname, paramname)) and os.path.isfile("../results/ext_hyperff/{}/{}/effdiameter.txt".format(dataname, paramname)):
                    namelist.append(("hyperff", paramname))

    elif ablation_target == "thera":
        count = 0
        for _fname in os.listdir("../dataset/tr/"):
            if dataname + "_tr" in _fname:
                fname = _fname[:-4]
                tmp = fname.split("_")[2:]
                paramname = "_".join(tmp)
                b, p, c = int(tmp[0]), float(tmp[1]), float(tmp[2])
                if b not in bset:
                    continue
                if p not in pset:
                    continue
                if c not in cset:
                    continue
                if os.path.isfile("../results/thera/{}/{}/effdiameter.txt".format(dataname, paramname)) and os.path.isfile("../results/thera/{}/{}/sv.txt".format(dataname, paramname)):
                    if os.path.isfile("../results/ext_thera/{}/{}/effdiameter.txt".format(dataname, paramname)) and os.path.isfile("../results/ext_thera/{}/{}/sv.txt".format(dataname, paramname)):
                        namelist.append(("thera", paramname))
                        count += 1
        assert count <= len(bset) * len(pset) * len(cset)
        print("tr :", count, len(bset) * len(pset) * len(cset))

    elif ablation_target == "HyperK":
        d = pd.read_csv("../results/ext_HyperK/{}/output_list_half.txt".format(dataname))
        for irow, row in d.iterrows():
            opt = row["modelIndex"]
            if os.path.isfile("../results/ext_HyperK/{}/{}/effdiameter.txt".format(dataname, opt)):
                namelist.append(("ext_HyperK", int(opt)))
    
    return namelist


def make_ablation_table(dataname, ablation_target):
    namelist = get_directories(dataname, ablation_target)

    property_list = ["degree", "size", "pairdeg", "intersection",
             "clusteringcoef_hedge", "density_dist", "overlapness_dist",
             "sv", "effdiam"]
    # prepare
    outputdir = "csv/ablation_" + ablation_target + "/"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    outputpath = outputdir + dataname + ".txt"
    columns = [column_mapping[prop] for prop in property_list]
    with open(outputpath, "w") as f:
        f.write(",".join(["AlgoName", "AlgoOpt"] + columns) + "\n")
        
    for name, modelindex in tqdm(namelist):
        print(name, modelindex)
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
    
    return outputpath
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--ablation_target", type=str, help="[hyperff, thera, HyperK]")
    args = parser.parse_args()

    outputpath = make_ablation_table(args.dataname, args.ablation_target)
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff'] 

    # Make Diff Result
    d = pd.read_csv(outputpath)
    target = d[evallist]
    d['avg'] = target.mean(axis=1)

    # Make Ranking Result
    prefix = outputpath[:-4]
    d = pd.read_csv(outputpath)
    for ename in evallist:
        d[ename] = d[ename].abs().rank(method='min')
    ranks = d[evallist]
    d['avg'] = ranks.mean(axis=1)
    d.to_csv(prefix + "_rank.txt", index=False)

    # Total Eval
    # Rank
    rd = pd.read_csv(prefix + "_rank.txt")
    rd = rd.sort_values(by=["avg"], ascending=True)

    algoopt2sum = defaultdict(int)
    rorder = 1
    for irow, row in rd.iterrows():
        if args.ablation_target in row["AlgoName"]:
            algoopt = str(row["AlgoOpt"])
            algoopt2sum[algoopt] += rorder
        rorder += 1

    sortedkeys = sorted(list(algoopt2sum.keys()), key=lambda x: algoopt2sum[x])
    if os.path.isfile("ablation_result/{}.pkl".format(args.ablation_target)):
        with open("ablation_result/{}.pkl".format(args.ablation_target), "rb") as f:
            ablation_result = pickle.load(f)
    elif os.path.isdir("ablation_result/") is False:
        os.makedirs("ablation_result/")
        ablation_result = {}
    else:
        ablation_result = {}

    top_opt = sortedkeys[0]
    ablation_result[args.dataname] = top_opt
    
    with open("ablation_result/{}.pkl".format(args.ablation_target), "wb") as f:
        pickle.dump(ablation_result, f)
    
    
    
    
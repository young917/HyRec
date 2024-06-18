import argparse
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import math
import pandas as pd
import pickle
from tqdm import tqdm
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

def get_directories(dataname, ablation_target):
    bset=[8, 12, 15]
    pset=[0.5, 0.7, 0.9]
    cset=[2.0, 6.0, 10.0]
    namelist = []
    require_filelist = ["degree", "pairdeg", "intersection", "size"] +["clusteringcoef_hedge", "density_dist", "overlapness_dist", "sv", "effdiameter"]
    if ablation_target == "hkron_sv":
        d = pd.read_csv("../results_dup/hkron_sv/{}/output_list.txt".format(dataname))
        for irow, row in d.iterrows():
            opt = row["modelIndex"]
            existflag = True
            for fname in require_filelist:
                if os.path.isfile("../results_dup/hkron_sv/{}/{}/{}.txt".format(dataname, opt, fname)) is False:
                    existflag = False
            if existflag:
                namelist.append(("hkron_sv", int(opt)))
    
    return namelist


def make_ablation_table(dataname, ablation_target):
    baselist = [("answer", -1), ("hyperlap", -1),  ("cl", -1)]
    if os.path.isfile("../results_dup/hyperpa/{}/sv.txt".format(dataname)):
        baselist.append(("hyperpa", -1))
    with open("ablation_result/ff.pkl", "rb") as f:
        ff_res = pickle.load(f)
    with open("ablation_result/tr.pkl", "rb") as f:
        tr_res = pickle.load(f)
    if dataname in ff_res:
        baselist.append(("ff", ff_res[dataname]))
    if dataname in tr_res:
        baselist.append(("tr", tr_res[dataname]))

    namelist = get_directories(dataname, ablation_target)

    # property_list = ["NumHedge", "NumNode", "degree", "size", "pairdeg", "intersection",
            #  "clusteringcoef_hedge", "density_dist", "overlapness_dist",
            #  "sv", "effdiam"]
    property_list = ["degree", "size", "pairdeg", "intersection",
             "clusteringcoef_hedge", "density_dist", "overlapness_dist",
             "sv", "effdiam"]
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff'] 

    output_d = pd.read_csv("../results_dup/hkron_sv/{}/output_list.txt".format(dataname))
    idx2numparam = {}
    for irow, row in output_d.iterrows():
        idx = row["modelIndex"]
        _path = row["modelpath"]
        _row, _col = _path.split("/")[-2].split("_")[:2]
        numparam = int(_row) * int(_col)
        idx2numparam[idx] = numparam

    # prepare
    outputdir = "csv/ablation_" + ablation_target + "/"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    outputpath = outputdir + dataname + ".txt"
    columns = [column_mapping[prop] for prop in property_list]
    opt2result = {}

    for targetname, targetmodelindex in tqdm(namelist):

        with open(outputpath, "w") as f:
            f.write(",".join(["AlgoName", "AlgoOpt"] + columns) + "\n")
            
        for name, modelindex in baselist + [(targetname, targetmodelindex)]:
            ret, dist = read_properties(dataname, name, modelindex)
            if name == "answer":
                ret_answer = ret
                dist_answer = dist   
                continue
            difflist = []
            for prop in property_list:
                
                if prop in ["NumHedge", "NumNode", "LargestSV", "effdiam"]: #, "mod"]:
                    diff = abs(dist[prop] - dist_answer[prop]) / dist_answer[prop]
                    
#                 elif prop in ["degree", "size", "pairdeg", "intersection"]:
#                     diff = get_cumul_dist(dist_answer[prop], dist[prop])
#                     diff = get_rmse_dist(dist_answer[prop], dist[prop], set_length=True, normalize=False, logflag=True)
                    
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
        
        # rank and norm
        prefix = outputpath[:-4]
        d = pd.read_csv(outputpath)
        for col in evallist:
            if d[col].std() != 0:
                d[col] = (d[col] - d[col].mean()) / d[col].std()
        norms = d[evallist]
        d['avg'] = norms.mean(axis=1)
        d.to_csv(prefix + "_norm.txt", index=False)
        # Make Ranking Result
        d = pd.read_csv(outputpath)
        for ename in evallist:
            d[ename] = d[ename].abs().rank(method='min')
        ranks = d[evallist]
        d['avg'] = ranks.mean(axis=1)
        d.to_csv(prefix + "_rank.txt", index=False)

        # Norm
        nd = pd.read_csv(prefix + "_norm.txt")
        nd = nd.sort_values(by=["avg"], ascending=True)
        norder = 1
        for irow, row in nd.iterrows():
            if row["AlgoName"] == targetname:
                target_norm = norder #row["avg"]
            norder += 1
        # Rank
        rd = pd.read_csv(prefix + "_rank.txt")
        rd = rd.sort_values(by=["avg"], ascending=True)
        rorder = 1
        for irow, row in rd.iterrows():
            if row["AlgoName"] == targetname:
                # target_rank = rorder # row["avg"]
                if "threads" in dataname:
                    target_rank = row["avg"]
                else:
                    target_rank = rorder
            rorder += 1
        # opt2result[targetmodelindex] = (target_norm + target_rank)
        # opt2result[targetmodelindex] = target_rank
        opt2result[targetmodelindex] = (target_rank, idx2numparam[targetmodelindex])

    if os.path.isdir("ablation_result/{}/".format(ablation_target)) is False:
        os.makedirs("ablation_result/{}/".format(ablation_target))
    with open("ablation_result/{}/{}.pkl".format(ablation_target, dataname), "wb") as f:
        pickle.dump(opt2result, f)

    sorted_opt = sorted(list(opt2result.keys()), key=lambda x: opt2result[x])
    top_opts = sorted_opt[:5]

    print(opt2result)
    print(min(opt2result.values()))
    
    return top_opts
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--ablation_target", default="hkron_sv", type=str)
    args = parser.parse_args()

    top_opts = make_ablation_table(args.dataname, args.ablation_target)
    
    if os.path.isfile("ablation_result/{}.pkl".format(args.ablation_target)):
        with open("ablation_result/{}.pkl".format(args.ablation_target), "rb") as f:
            ablation_result = pickle.load(f)
    else:
        ablation_result = {}
    
    ablation_result[args.dataname] = top_opts
    
    with open("ablation_result/{}.pkl".format(args.ablation_target), "wb") as f:
        pickle.dump(ablation_result, f)
    
    
    
    
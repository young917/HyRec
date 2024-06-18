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

half2full = {
        "email-Enron-half": "email-Enron-full",
        "email-Eu-half": "email-Eu-full",
        "NDC-classes-half": "NDC-classes-full",
        "NDC-substances-half": "NDC-substances-full",
        "tags-ask-ubuntu-half": "tags-ask-ubuntu",
        "tags-math-sx-half": "tags-math-sx",
        "contact-high-school-half": "contact-high-school",
        "contact-primary-school-half": "contact-primary-school",
        "threads-ask-ubuntu-half": "threads-ask-ubuntu",
        "threads-math-sx-half": "threads-math-sx",
        "coauth-MAG-Geology-half": "coauth-MAG-Geology-full"
    }

full2half = {
        "email-Enron-full": "email-Enron-half",
        "email-Eu-full": "email-Eu-half",
        "NDC-classes-full": "NDC-classes-half",
        "NDC-substances-full": "NDC-substances-half",
        "tags-ask-ubuntu": "tags-ask-ubuntu-half",
        "tags-math-sx": "tags-math-sx-half",
        "contact-high-school": "contact-high-school-half",
        "contact-primary-school": "contact-primary-school-half",
        "threads-ask-ubuntu": "threads-ask-ubuntu-half",
        "threads-math-sx": "threads-math-sx-half",
        "coauth-MAG-Geology-full": "coauth-MAG-Geology-half"
    }

def get_baseline_list(dataname):
    half_dataname = full2half[dataname]
    namelist = [("answer", -1)]
    if os.path.isfile("../results/ext_hyperpa/{}/effdiameter.txt".format(half_dataname)):
        namelist.append(("ext_hyperpa", -1))
    # HyperFF and Thera
    for target in ["hyperff", "thera"]:
        with open("ablation_result/{}.pkl".format(target), "rb") as f:
            result = pickle.load(f)
        namelist.append(("ext_" + target, result[half_dataname]))
    for target in ["HyperK"]:
        with open("ablation_result/{}.pkl".format(target), "rb") as f:
            result = pickle.load(f)
        namelist.append(("ext_" + target, result[half_dataname]))
        
    return namelist

def make_dist_table(dataname, outputdir):
    half_dataname = full2half[dataname]
    property_list = ["degree", "size", "pairdeg", "intersection","sv", 
                 "clusteringcoef_hedge", "density_dist", "overlapness_dist", "effdiam"] #, "mod"]
    
    namelist = get_baseline_list(dataname)
    print(namelist)
    
    outputpath = outputdir + dataname + ".txt"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    columns = [column_mapping[prop] for prop in property_list]
    with open(outputpath, "w") as f:
        f.write(",".join(["AlgoName", "AlgoOpt"] + columns) + "\n")
        
    for name, modelindex in namelist:
        if name == "answer":
            ret, dist = read_properties(dataname, name, modelindex)
        elif "pa" in name:
            ret, dist = read_properties(half_dataname, name, modelindex)
        elif "ff" in name or "thera" in name:
            ret, dist = read_properties(half_dataname, name, modelindex)
        else:
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
    parser.add_argument("--outputdir", default="csv/ext/", type=str)
    args = parser.parse_args()
    
    make_dist_table(args.dataname, args.outputdir)
    
    d = pd.read_csv(args.outputdir + args.dataname + ".txt")
    
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff']
    
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
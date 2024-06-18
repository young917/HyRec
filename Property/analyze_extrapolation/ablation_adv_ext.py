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
        "coauth-MAG-Geology-half": "coauth-MAG-Geology-full",
        "coauth-MAG-History-half": "coauth-MAG-History-full",
        
        "email-Enron-half2": "email-Enron-full",
        "email-Eu-half2": "email-Eu-full",
        "contact-primary-school-half2": "contact-primary-school",
        "contact-high-school-half2": "contact-high-school",
        "NDC-classes-half2": "NDC-classes-full",
        "NDC-substances-half2": "NDC-substances-full",
        "tags-ask-ubuntu-half2": "tags-ask-ubuntu",
        "tags-math-sx-half2": "tags-math-sx",
        "threads-ask-ubuntu-half2": "threads-ask-ubuntu",
        "threads-math-sx-half2": "threads-math-sx",
        "coauth-MAG-Geology-half2": "coauth-MAG-Geology-full",
    }

def make_ablation_table_half(dataname, ablation_target):
    baselist = [("answer", -1)]
    if os.path.isfile("../results_dup/hyperpa/{}/sv.txt".format(dataname)):
        baselist.append(("hyperpa", -1))
    with open("ablation_result/ff.pkl", "rb") as f:
        ff_res = pickle.load(f)
    with open("ablation_result/tr.pkl", "rb") as f:
        tr_res = pickle.load(f)
    baselist.append(("ff", ff_res[dataname]))
    baselist.append(("tr", tr_res[dataname]))

    namelist = []
    d = pd.read_csv("../results_dup/hkron_sv_half/{}/output_list_half.txt".format(dataname))
    for irow, row in d.iterrows():
        opt = row["modelIndex"]
        if os.path.isfile("../results_dup/hkron_sv_half/{}/{}/sv.txt".format(dataname, opt)):
            namelist.append(("hkron_sv_half", int(opt)))

    property_list = ["degree", "size", "pairdeg", "intersection",
             "clusteringcoef_hedge", "density_dist", "overlapness_dist",
             "sv", "effdiam"]
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff'] 

    outputdir = "csv/ablation_" + ablation_target + "_half/"
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
        target_norm = -1
        for irow, row in nd.iterrows():
            if targetname in row["AlgoName"]:
                target_norm = norder #row["avg"]
            norder += 1
        assert target_norm > 0
        # Rank
        rd = pd.read_csv(prefix + "_rank.txt")
        rd = rd.sort_values(by=["avg"], ascending=True)
        rorder = 1
        target_rank = -1
        for irow, row in rd.iterrows():
            if targetname in row["AlgoName"]:
                target_rank = rorder # row["avg"]
            rorder += 1
        assert target_rank > 0
        # opt2result[targetmodelindex] = (target_norm + target_rank)
        opt2result[targetmodelindex] = target_rank

    if os.path.isdir("ablation_result_half/{}/".format(ablation_target)) is False:
        os.makedirs("ablation_result_half/{}/".format(ablation_target))
    with open("ablation_result_half/{}/{}.pkl".format(ablation_target, dataname), "wb") as f:
        pickle.dump(opt2result, f)

    sorted_opt = sorted(list(opt2result.keys()), key=lambda x: opt2result[x])

    print(opt2result)
    
    return opt2result

def make_ablation_table_ext(dataname, ablation_target):
    baselist = [("answer", -1)]
    if os.path.isfile("../results_dup/ext_hyperpa/{}/sv.txt".format(dataname)):
        baselist.append(("ext_hyperpa", -1))
    with open("ablation_result/ff.pkl", "rb") as f:
        ff_res = pickle.load(f)
    with open("ablation_result/tr.pkl", "rb") as f:
        tr_res = pickle.load(f)
    baselist.append(("ext_ff", ff_res[dataname]))
    baselist.append(("ext_tr", tr_res[dataname]))

    full_dataname = half2full[dataname]
    namelist = []
    d = pd.read_csv("../results_dup/hkron_sv_half/{}/output_list_ext.txt".format(full_dataname))
    for irow, row in d.iterrows():
        opt = row["modelIndex"]
        if os.path.isfile("../results_dup/hkron_sv_half/{}/{}/sv.txt".format(full_dataname, opt)):
            namelist.append(("hkron_sv_half", int(opt)))

    property_list = ["degree", "size", "pairdeg", "intersection",
             "clusteringcoef_hedge", "density_dist", "overlapness_dist",
             "sv", "effdiam"]
    evallist = ['deg', 'sz', 'pd', 'its', 'cch', 'dst', 'ov', 'sv', 'eff'] 

    outputdir = "./csv/ablation_" + ablation_target + "_half/"
    if os.path.isdir(outputdir) is False:
        os.makedirs(outputdir)
    outputpath = outputdir + full_dataname + ".txt"
    columns = [column_mapping[prop] for prop in property_list]
    opt2result = {}

    for targetname, targetmodelindex in tqdm(namelist):

        with open(outputpath, "w") as f:
            f.write(",".join(["AlgoName", "AlgoOpt"] + columns) + "\n")
            
        for name, modelindex in baselist + [(targetname, targetmodelindex)]:
            if name in ["ext_hyperpa", "ext_ff", "ext_tr"]:
                ret, dist = read_properties(dataname, name, modelindex)
            else:
                ret, dist = read_properties(full_dataname, name, modelindex)
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
        target_norm = -1
        for irow, row in nd.iterrows():
            if targetname in row["AlgoName"]:
                target_norm = norder #row["avg"]
            norder += 1
        assert target_norm > 0
        # Rank
        rd = pd.read_csv(prefix + "_rank.txt")
        rd = rd.sort_values(by=["avg"], ascending=True)
        rorder = 1
        target_rank = -1
        for irow, row in rd.iterrows():
            if targetname in row["AlgoName"]:
                # target_rank = rorder # row["avg"]
                if "threads" in dataname:
                    target_rank = row["avg"]
                else:
                    target_rank = rorder
            rorder += 1
        assert target_rank > 0
        # opt2result[targetmodelindex] = (target_norm + target_rank)
        opt2result[targetmodelindex] = target_rank


    if os.path.isdir("ablation_result_half/{}/".format(ablation_target)) is False:
        os.makedirs("ablation_result_half/{}/".format(ablation_target))
    with open("ablation_result_half/{}/{}.pkl".format(ablation_target, full_dataname), "wb") as f:
        pickle.dump(opt2result, f)

    sorted_opt = sorted(list(opt2result.keys()), key=lambda x: opt2result[x])

    print(opt2result)
    
    return opt2result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str)
    parser.add_argument("--ablation_target", default="hkron_sv", type=str)
    args = parser.parse_args()
    
    assert "half" in args.dataname
    full_dataname = half2full[args.dataname]
    
    opt2res_half = make_ablation_table_half(args.dataname, args.ablation_target)
    opt2res_ext = make_ablation_table_ext(args.dataname, args.ablation_target)

    print("Half")
    print(opt2res_half)
    print("Ext")
    print(opt2res_ext)

    result = []
    # result_indexes = []
    diridx2idxes = defaultdict(dict)
    half_d = pd.read_csv("../results_dup/hkron_sv_half/{}/output_list_half.txt".format(args.dataname))
    ext_d = pd.read_csv("../results_dup/hkron_sv_half/{}/output_list_ext.txt".format(full_dataname))
    for irow, row in half_d.iterrows():
        diridx = row["dirIndex"]
        idx = row["modelIndex"]
        if idx in opt2res_half.keys():
            if "half" not in diridx2idxes[diridx]:
                diridx2idxes[diridx]["half"] = []
            diridx2idxes[diridx]["half"].append(idx)
    for irow, row in ext_d.iterrows():
        diridx = row["dirIndex"]
        idx = row["modelIndex"]
        if diridx not in diridx2idxes.keys():
            continue
        if idx in opt2res_ext.keys():
            if "ext" not in diridx2idxes[diridx]:
                diridx2idxes[diridx]["ext"] = []
            diridx2idxes[diridx]["ext"].append(idx)
    for diridx in diridx2idxes.keys():
        if ("half" in diridx2idxes[diridx]) and ("ext" in diridx2idxes[diridx]):
            min_half = min([opt2res_half[opt] for opt in diridx2idxes[diridx]["half"]])
            min_ext = min([opt2res_ext[opt] for opt in diridx2idxes[diridx]["ext"]])
            min_half_pos = np.argmin([opt2res_half[opt] for opt in diridx2idxes[diridx]["half"]])
            min_ext_pos = np.argmin([opt2res_ext[opt] for opt in diridx2idxes[diridx]["ext"]])
            result.append((min_ext, min_half, diridx2idxes[diridx]["ext"][min_ext_pos], diridx2idxes[diridx]["half"][min_half_pos]))
            # result_indexes.append((diridx2idxes[diridx]["ext"][min_ext_pos], diridx2idxes[diridx]["half"][min_half_pos]))
    
    sorted_result = sorted(result)
    print("Result")
    print(sorted_result[:5])
    sorted_result_indexes = [(res[2], res[3]) for res in sorted_result[:5]]

    if os.path.isfile("ablation_result_half/{}.pkl".format(args.ablation_target)):
        with open("ablation_result_half/{}.pkl".format(args.ablation_target), "rb") as f:
            ablation_result = pickle.load(f)
    else:
        ablation_result = {}
    
    ablation_result[args.dataname] = [res[1] for res in sorted_result_indexes]
    ablation_result[full_dataname] = [res[0] for res in sorted_result_indexes]

    with open("ablation_result_half/{}.pkl".format(args.ablation_target), "wb") as f:
        pickle.dump(ablation_result, f)
    

    
    
    
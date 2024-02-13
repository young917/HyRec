import os
from collections import defaultdict
import scipy
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse.linalg
import argparse

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
}
datalist = ["email-Enron-half", "email-Eu-half",
            "contact-primary-school-half", "contact-high-school-half",
            "NDC-classes-half", "NDC-substances-half",
            "tags-ask-ubuntu", "threads-ask-ubuntu",
            "tags-math-sx", "threads-math-sx"]
recalculate_flag = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataname', default="")
parser.add_argument('--recalculate_flag', action="store_true")
args = parser.parse_args()

if len(args.dataname) > 0:
    datalist = [args.dataname]
recalculate_flag = args.recalculate_flag

for dataname in datalist:
    print(dataname)
    outputshfile = f"../../run/run_hyperk_{dataname}.sh"
    if os.path.isfile(outputshfile):
        os.remove(outputshfile)
    with open(outputshfile, "w") as f:
        f.write("cd ..\n")

    answer_dataname = fulldatalist[dataname]
    answer_hedgeset = set()
    answer_nodeset = set()
    with open("../../dataset/" + answer_dataname + ".txt", "r") as f:
        for line in f.readlines():
            hedge_str = line.rstrip().split(",")
            hedge_str = sorted(list(set([int(v) for v in hedge_str])))
            for v in hedge_str:
                answer_nodeset.add(v)
            hedge_str = ",".join([str(v) for v in hedge_str])
            answer_hedgeset.add(hedge_str)
    answerH = len(answer_hedgeset)
    answerV = len(answer_nodeset)

    outputfile = "./{}/output_list.txt".format(dataname)
    already_exist = {}
    # clear previous output
    if os.path.isfile(outputfile):
        os.remove(outputfile)
    elif os.path.isdir("./{}/".format(dataname)) is False:
        os.makedirs("./{}/".format(dataname))
    if recalculate_flag is False and os.path.isfile(outputfile):
        _d = pd.read_csv(outputfile)
        for irow, row in _d.iterrows():
            modelpath = row["modelpath"]
            modelindex = row["modelIndex"]
            paramname = modelpath.split("/")[-2]
            outputdir = "./{}/{}/".format(dataname, str(modelindex))
            if os.path.isfile(outputdir + "effdiameter.txt") and os.path.isfile(outputdir + "clusteringcoef_hedge.txt"):
                already_exist[paramname] = modelindex      
    else:
        with open(outputfile, "w") as f:
            f.write("modelIndex,modelpath,lossSV,lossSVeval,numH,numV,answerH,answerV\n")

    output_count = len(already_exist)
    for rootdir in ["../../../Model/result/"]
        if os.path.isdir(rootdir + dataname) is False:
            continue

        for dirname in os.listdir(rootdir + dataname):
            if dirname in already_exist:
                continue

            curdir = rootdir + dataname + "/" + dirname + "/"
            fulldir = rootdir + dataname + "/" + dirname + "/full/"
            print(curdir)

            tmp = dirname.split("_")
            lossSV, lossSV_eval = -1, -1

            if os.path.isdir(curdir) is False:
                continue
            if os.path.isdir(fulldir) is False:
                continue
            filelist = [fname for fname in os.listdir(curdir)]
            full_filelist = [fname for fname in os.listdir(fulldir)]
            flag = True
            if "log.txt" not in filelist:
                flag = False
            if "eval_log.txt" not in filelist:
                flag = False
            if "sampled_0.txt" not in full_filelist:
                flag = False
            if flag is False:
                continue
            # Valid Output!
            try:
                with open(curdir + "log.txt", "r") as f:
                    for line in f.readlines():
                        ep_str, loss_str = line.rstrip().split(", ")
                        ep = int(ep_str.split(":")[-1])
                        lossSV = float(loss_str.split(":")[-1])

                if os.path.isfile(fulldir + "final_eval_log.txt"):
                    with open(fulldir + "final_eval_log.txt", "r") as f:
                        for line in f.readlines():
                            ep_str, loss_str = line.rstrip().split(", ")
                            ep = int(ep_str.split(":")[-1])
                            lossSV_eval = float(loss_str.split(":")[-1])
                else:
                    with open(fulldir + "eval_log.txt", "r") as f:
                        for line in f.readlines():
                            ep_str, loss_str = line.rstrip().split(", ")
                            ep = int(ep_str.split(":")[-1])
                            lossSV_eval = float(loss_str.split(":")[-1])

            except:
                continue
                        
            curpath = fulldir + "sampled_0.txt"
            sample_vals, sample_rows, sample_cols = [], [], []
            hypergraph = defaultdict(list)
            entire_hedgeset = set()
            cur_nodeset = set()
            try:
                with open(curpath, "r") as f:
                    for line in f.readlines():
                        i, j, _ = line.rstrip().split(" ")
                        i, j = int(i), int(j)
                        sample_rows.append(j)
                        sample_cols.append(i)
                        sample_vals.append(1.0)
                        hypergraph[i].append(j)
            except:
                continue

            curpath = "./{}/{}/hypergraph".format(dataname, output_count)
            if os.path.isdir("./{}/{}/".format(dataname, output_count)) is False:
                os.makedirs("./{}/{}/".format(dataname, output_count))
            with open(curpath + ".txt", "w") as f:
                for hedgeidx in hypergraph.keys():
                    hedge = hypergraph[hedgeidx]
                    hedgeset = tuple(sorted(list(set(hedge))))
                    if hedgeset not in entire_hedgeset:
                        entire_hedgeset.add(hedgeset)
                        for _v in hedge:
                            cur_nodeset.add(_v)
                        hedge = [str(_v) for _v in hedge]
                        f.write(",".join(hedge) + "\n")
            print(len(hypergraph), len(entire_hedgeset))

            if len(entire_hedgeset) > 0:
                curpath = "./{}/{}/hypergraph".format(dataname, output_count)
                with open(outputshfile, "+a") as f:
                    f.write("./bin/Sampling --inputpath {} --outputdir ./results/hyperk/{}/{}/\n".format(curpath, dataname, output_count))
                    f.write("cd src\n")
                    f.write("python calculation_helper.py --inputpath {} --outputdir ../results/hyperk/{}/{}/ --sv --effdiam\n".format(curpath, dataname, output_count))
                    f.write("cd ../\n")
                with open(outputfile, "+a") as f:
                    # f.write("modelIndex,modelpath,numH,totalH,numV,lossSV,lossSVeval,answerH,answerV\n")
                    f.write("%d,%s,%.8f,%.8f,%d,%d,%d,%d\n" % (output_count, rootdir + dataname + "/" + dirname + "/", lossSV, lossSV_eval, len(entire_hedgeset), len(cur_nodeset), answerH, answerV))
                output_count += 1

    print(dataname, output_count)

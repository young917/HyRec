import os
from collections import defaultdict
import scipy
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse.linalg
import argparse

datalist = ["email-Enron-full", "email-Eu-full",
            "contact-primary-school", "contact-high-school",
            "NDC-classes-full", "NDC-substances-full",
            "tags-ask-ubuntu", "threads-ask-ubuntu",
            "tags-math-sx", "threads-math-sx",
            "coauth-MAG-Geology-full"]

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', default="")
parser.add_argument('--inputdir', default="../../Model/result/")
args = parser.parse_args()

if len(args.dataname) > 0:
    datalist = [args.dataname]

for dataname in datalist:
    outputshfile = f"../run/run_eval_HyRec_{dataname}.sh"
    if os.path.isfile(outputshfile):
        os.remove(outputshfile)
    with open(outputshfile, "w") as f:
        f.write("cd ..\n")

    outputfile = "./HyRec/{}/output_list.txt".format(dataname)
    if os.path.isfile(outputfile):
        os.remove(outputfile)
    elif os.path.isdir("./HyRec/{}/".format(dataname)) is False:
        os.makedirs("./HyRec/{}/".format(dataname))
    with open(outputfile, "w") as f:
        f.write("modelIndex,modelpath\n")

    output_count = 0 
    rootdir = args.inputdir
    for dirname in os.listdir(rootdir + dataname):
        curdir = rootdir + dataname + "/" + dirname + "/"
        if os.path.isdir(curdir) is False:
            print("No exist directory: ", curdir)
            continue
                    
        for outname in ["sampled_0.txt"]:
            if os.path.isfile(curdir + outname) is False:
                print("No ", curdir + outname)
                continue

            curpath = curdir + outname
            sample_vals, sample_rows, sample_cols = [], [], []
            hypergraph = defaultdict(list)
            numhedge = 0
            with open(curpath, "r") as f:
                for line in f.readlines():
                    i, j, _ = line.rstrip().split(" ")
                    i, j = int(i), int(j)
                    sample_rows.append(j)
                    sample_cols.append(i)
                    sample_vals.append(1.0)
                    hypergraph[i].append(j)

            curpath = "./HyRec/{}/{}/hypergraph".format(dataname, output_count)
            if os.path.isdir("./HyRec/{}/{}/".format(dataname, output_count)) is False:
                os.makedirs("./HyRec/{}/{}/".format(dataname, output_count))
            with open(curpath + ".txt", "w") as f:
                for hedgeidx in hypergraph.keys():
                    hedge = hypergraph[hedgeidx]
                    hedge = sorted(list(set(hedge)))
                    hedge = [str(_v) for _v in hedge]
                    f.write(",".join(hedge) + "\n")
                    numhedge += 1

            if numhedge > 0:
                curpath = "./results/HyRec/{}/{}/hypergraph".format(dataname, output_count)
                with open(outputshfile, "+a") as f:
                    f.write("./bin/Evaluation --inputpath {} --outputdir ./results/HyRec/{}/{}/ --dupflag\n".format(curpath, dataname, output_count))
                    f.write("cd src\n")
                    f.write("python calculation_helper.py --inputpath .{} --outputdir ../results/HyRec/{}/{}/ --sv --effdiam --dupflag\n".format(curpath, dataname, output_count))
                    f.write("cd ../\n")
                with open(outputfile, "+a") as f:
                    f.write("%d,%s\n" % (output_count, rootdir + dataname + "/" + dirname + "/"))
                output_count += 1
            else:
                print("No exist hyperedges: ", curdir)
    
    print(dataname, output_count)

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
    "coauth-MAG-Geology-half": "coauth-MAG-Geology-full"
}
datalist = ["email-Enron-half", "email-Eu-half",
            "contact-high-school-half", "contact-primary-school-half", 
            "NDC-classes-half", "NDC-substances-half",
            "tags-ask-ubuntu-half", "tags-math-sx-half",
            "threads-ask-ubuntu-half", "threads-math-sx-half",
            "coauth-MAG-Geology-half"]

def readwrite(curpath, dataname, count, outputshfile):
    sample_vals, sample_rows, sample_cols = [], [], []
    hypergraph = defaultdict(list)
    errorflag = False
    numhedge = 0
    with open(curpath, "r") as f:
        for line in f.readlines():
            if len(line.rstrip().split(" ")) != 3:
                print(curpath)
                errorflag = True
                break
            i, j, _ = line.rstrip().split(" ")
            i, j = int(i), int(j)
            sample_rows.append(j)
            sample_cols.append(i)
            sample_vals.append(1.0)
            hypergraph[i].append(j)
    if errorflag:
        return -1

    curpath = "./ext_HyperK/{}/{}/hypergraph".format(dataname, count)
    if os.path.isdir("./ext_HyperK/{}/{}/".format(dataname, count)) is False:
        os.makedirs("./ext_HyperK/{}/{}/".format(dataname, count))
    with open(curpath + ".txt", "w") as f:
        for hedgeidx in hypergraph.keys():
            hedge = hypergraph[hedgeidx]
            hedge = sorted(list(set(hedge)))
            hedge = [str(_v) for _v in hedge]
            f.write(",".join(hedge) + "\n")
            numhedge += 1

    curpath = "./results/ext_HyperK/{}/{}/hypergraph".format(dataname, count)
    with open(outputshfile, "+a") as f:
        f.write("./bin/Evaluation --inputpath {} --outputdir ./results/ext_HyperK/{}/{}/ --dupflag\n".format(curpath, dataname, count))
        f.write("cd src\n")
        f.write("python calculation_helper.py --inputpath .{} --outputdir ../results/ext_HyperK/{}/{}/ --sv --effdiam --dupflag --appxflag\n".format(curpath, dataname, count))
        f.write("cd ../\n")

    return numhedge

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', default="")
parser.add_argument('--inputdir', default="../../Model/result/")
args = parser.parse_args()

if len(args.dataname) > 0:
    datalist = [args.dataname]

for dataname in datalist:
    outputshfile_half = f"../run/run_eval_ext_HyperK_{dataname}.sh"
    if os.path.isfile(outputshfile_half):
        os.remove(outputshfile_half)
    with open(outputshfile_half, "w") as f:
        f.write("cd ..\n")
    full_dataname = fulldatalist[dataname]
    outputshfile_ext = f"../run/run_eval_ext_HyperK_{full_dataname}.sh"
    if os.path.isfile(outputshfile_ext):
        os.remove(outputshfile_ext)
    with open(outputshfile_ext, "w") as f:
        f.write("cd ..\n")

    outputfile_half = "./ext_HyperK/{}/output_list_half.txt".format(dataname)
    if os.path.isfile(outputfile_half):
        os.remove(outputfile_half)
    elif os.path.isdir("./ext_HyperK/{}/".format(dataname)) is False:
        os.makedirs("./ext_HyperK/{}/".format(dataname))
    with open(outputfile_half, "w") as f:
        f.write("modelIndex,modelpath,dirIndex\n")

    full_dataname = fulldatalist[dataname]
    outputfile_ext = "./ext_HyperK/{}/output_list_ext.txt".format(full_dataname)
    if os.path.isfile(outputfile_ext):
        os.remove(outputfile_ext)
    elif os.path.isdir("./ext_HyperK/{}/".format(full_dataname)) is False:
        os.makedirs("./ext_HyperK/{}/".format(full_dataname))
    with open(outputfile_ext, "w") as f:
        f.write("modelIndex,modelpath,dirIndex\n")

    output_count = 0
    output_count_half = 0
    output_count_ext = 0
    rootdir = args.inputdir

    if os.path.isdir(rootdir + dataname) is False:
        continue

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
            _nh = readwrite(curpath, dataname, output_count_half, outputshfile_half)  
            if _nh == -1:
                continue
            with open(outputfile_half, "+a") as f:
                f.write("%d,%s,%d\n" % (output_count_half, rootdir + dataname + "/" + dirname + "/", output_count))
            output_count_half += 1
        
        for outname_ext in ["full/sampled_0.txt"]:
            curpath_ext = curdir +  outname_ext
            if os.path.isfile(curpath_ext) is False:
                print("No ", curpath_ext)
                continue
            full_dataname = fulldatalist[dataname]
            readwrite(curpath_ext, full_dataname, output_count_ext, outputshfile_ext)
            with open(outputfile_ext, "+a") as f:
                f.write("%d,%s,%d\n" % (output_count_ext, rootdir + dataname + "/" + dirname + "/full/", output_count))
            output_count_ext += 1
        
        output_count += 1

    print(dataname, output_count_half, output_count_ext)

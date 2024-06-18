import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import os
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import math

def linearregression(X, Y, nolog=False):
    if len(X) == 0:
        return 0, 0, [0], 0
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y).reshape(-1, 1)
    if nolog is False:
        X = np.log2(X)
        Y = np.log2(Y)
    reg = LinearRegression().fit(X, Y)
    score = reg.score(X, Y)
    coef = reg.coef_
    assert len(coef) == 1
    coef = coef[0][0]
    intercept = reg.intercept_[0]
    pred = reg.predict(X).flatten()
    pred = np.exp2(pred)

    return score, coef, pred, intercept


# Read Properties ------------------------------------------------------------------------------------------------------

def read_properties(dataname, algoname, modelindex=-1):
    if "answer" == algoname:
        graphpath = "../dataset/" + dataname + ".txt"
        outputdir = "../results/{}/{}/".format(algoname, dataname)
    elif "hypercl" == algoname:
        graphpath = "../dataset/" + dataname + "_cl.txt"
        outputdir = "../results/{}/{}/".format(algoname, dataname)
    elif "hyperlap" == algoname:
        graphpath = "../dataset/" + dataname + "_lap.txt"
        outputdir = "../results/{}/{}/".format(algoname, dataname)
    elif "hyperpa" == algoname:
        graphpath = "../dataset/" + dataname + "_pa.txt"
        outputdir = "../results/{}/{}/".format(algoname, dataname)
    elif "ext_hyperpa" == algoname:
        graphpath = "../dataset/" + dataname + "_pa.txt"
        outputdir = "../results/{}/{}/".format(algoname, dataname)
    elif "thera" == algoname:
        graphpath = "../dataset/tr/{}_tr_{}.txt".format(dataname, modelindex)
        outputdir = "../results/thera/{}/{}/".format(dataname, modelindex)
    elif "ext_thera" == algoname:
        graphpath = "../dataset/tr/{}_tr_{}.txt".format(dataname, modelindex)
        outputdir = "../results/ext_thera/{}/{}/".format(dataname, modelindex)
    elif "hyperff" == algoname:
        graphpath = "../dataset/{}_ff_{}.txt".format(dataname, modelindex)
        outputdir = "../results/hyperff/{}/{}/".format(dataname, modelindex)
    elif "ext_hyperff" == algoname:
        graphpath = "../dataset/{}_ext_ff_{}.txt".format(dataname, modelindex)
        outputdir = "../results/ext_hyperff/{}/{}/".format(dataname, modelindex)
    else:
        graphpath = "../results/{}/{}/{}/hypergraph.txt".format(algoname, dataname, modelindex)
        outputdir = "../results/{}/{}/{}/".format(algoname, dataname, modelindex)
    
    if os.path.isfile(graphpath) is False or os.path.isdir(outputdir) is False:
        return

    return_dict = {}
    dist = {}
    print(graphpath)
    
    # Num Nodes & Num Edges
    numhedge = 0
    nodeset = set()
    with open(graphpath, "r") as f:
        for line in f.readlines():
            hedge = line.rstrip().split(",")
            for v in hedge:
                nodeset.add(int(v))
            numhedge += 1
    numnode = len(nodeset)
    return_dict["NumHedge"] = numhedge
    return_dict["NumNode"] = numnode  
    dist["NumHedge"] = numhedge
    dist["NumNode"] = numnode    
    
    for distname in ["degree", "pairdeg", "intersection", "size"]:
        dist[distname] = {}
        X = []
        with open(outputdir + distname + ".txt", "r") as f:
            for line in f.readlines():
                val, pdf = line.rstrip().split(",")
                val, pdf = float(val), float(pdf)
                if pdf == 0.0 or val == 0.0:
                    continue
                dist[distname][val] = pdf
                X.append(val)
        X = sorted(X)
        Y = [dist[distname][x] for x in X]
        
    for distname in ["clusteringcoef_hedge", "density_dist", "overlapness_dist"]:
        _dist = defaultdict(list)
        dist[distname] = {}
        X = []
        with open(outputdir + distname + ".txt", "r") as f:
            for line in f.readlines():
                val, pdf = line.rstrip().split(",")
                val, pdf = float(val), float(pdf)
                if val == 0.0 or pdf == 0.0:
                    continue
                # binning
                val = 2 ** int(math.log2(val))
                _dist[val].append(pdf)
        X = sorted(list(_dist.keys()))
        Y = []
        for x in X:
            # binning
            y = 2 ** np.mean([math.log2(_y) for _y in _dist[x]])
            assert y > 0
            dist[distname][x] = y
            if y > 0:
                Y.append(y)
            else:
                Y.append(1)
        with open(outputdir + distname + "_processed.txt", "w") as f:
            for i in range(len(X)):
                assert dist[distname][X[i]] > 0.0
                line = [str(X[i]), str(dist[distname][X[i]])]
                f.write(",".join(line) + "\n")
    
    with open(outputdir + "sv.txt", "r") as f:
        tmp = {}
        X = []
        lsv = 0
        for li, line in enumerate(f.readlines()):
            sv = float(line.rstrip())
            if li == 0:
                lsv = sv
            tmp[li + 1] = sv
            X.append(li + 1)
        X = sorted(X)
        if dataname not in ["tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu", "threads-math-sx", "coauth-MAG-Geology-full"]:
            X = X[:min(1000, int(len(X) * 0.5))]
            Y = [tmp[x] for x in X]
        elif dataname in ["tags-ask-ubuntu", "tags-math-sx", "threads-ask-ubuntu", "threads-math-sx"]:
            X = X[:1000]
            if len(X) < 1000:
                Y = [tmp[x] for x in X] + [1e-12 for _ in range(1000 - len(X))]
                X = list(range(1,1000+1))
            else:
                Y = [tmp[x] for x in X]
        elif dataname in [ "coauth-MAG-Geology-full"]:
            X = X[:500]
            assert len(X) == 500
            Y = [tmp[x] for x in X]
        else:
            assert False, "Invalid Dataname"

        dist["sv"] = {}
        for x,y in zip(X, Y):
            dist["sv"][x] = y
        dist["LargestSV"] = lsv

    # EffDiam
    with open(outputdir + "effdiameter.txt", "r") as f:
        effdiam = 0
        for line in f.readlines():
            effdiam = float(line.rstrip())
        return_dict["effdiam"] = effdiam
        dist["effdiam"] = effdiam

    return return_dict, dist

# Get Distance ------------------------------------------------------------------------------------------------------

def get_cdf(_dict):
    cumulated_x = sorted(list(_dict.keys()))
    cdf = {}
    cum = 0

    for _x in cumulated_x:
        cum += _dict[_x]
        cdf[_x] = cum
        assert cum < 1.1
        
    return cdf

def get_cumul_dist(dict_x1, dict_x2):
    cdf1 = get_cdf(dict_x1)
    x1 = list(cdf1.keys())
    cdf2 = get_cdf(dict_x2)
    x2 = list(cdf2.keys())
    
    cum1, cum2 = 0, 0
    maxdiff = 0
    for x in sorted(list(set(x1 + x2))):
        if x in x1:
            cum1 = cdf1[x]
        if x in x2:
            cum2 = cdf2[x]
        if abs(cum1 - cum2) > maxdiff:
            maxdiff = abs(cum1 - cum2)
    
    return maxdiff

def get_rmse(y1s, y2s):
    total = 0
    y1s = np.array(y1s)
    y2s = np.array(y2s)
    total = np.sum((y1s - y2s) ** 2)
    assert y1s.shape[0] > 0
    total /= y1s.shape[0]
    total = total ** 0.5
    
    return total


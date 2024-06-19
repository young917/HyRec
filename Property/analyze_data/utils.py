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
    if nolog is False:
        pred = np.exp2(pred)

    return score, coef, pred, intercept, reg


# Read Properties ------------------------------------------------------------------------------------------------------

def read_properties(dataname, outputdir, modelindex=-1):
    graphpath = "../dataset/" + dataname + ".txt"

    if os.path.isfile(graphpath) is False or os.path.isdir(outputdir) is False:
        return

    return_dict = {}
    dist = {}
    
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


def get_cdf(x, y):
    cumulated_x = []
    cumulated_y = []
    cum = 0
    
    sorted_idx = np.argsort(x)
    sorted_x = [x[i] for i in sorted_idx]
    sorted_y = [y[i] for i in sorted_idx]
    
    for _x, _y in zip(sorted_x, sorted_y):
        cum += _y
        assert cum < 1.1 and cum > 0
        cumulated_x.append(_x)
        cumulated_y.append(cum)
        
    return cumulated_x, cumulated_y


def get_odds_ratio(x, y):
    cdf_x, cdf_y = get_cdf(x, y)
    
    new_x, new_y = [], []
    for _x, _y in zip(cdf_x, cdf_y):
        if _y >= 1:
            break
        new_x.append(_x)
        new_y.append(_y / (1.0 - _y))
    
    return new_x, new_y

from scipy import stats
def get_statistics(model, intercept, coef, x, y):
    params = np.append(intercept, coef)
    
    x = np.array(x)
    x = np.log2(x)
    y = np.array(y)
    y = np.log2(y)
    
    
    prediction = model.predict(x.reshape(-1, 1))

    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=1)

    new_trainset = pd.DataFrame({"Constant": np.ones(x.shape[0])}).join(pd.DataFrame(x))
    
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(prediction, y)

    variance = MSE * (np.linalg.inv(np.dot(new_trainset.T, new_trainset)).diagonal()) 

    std_error = np.sqrt(variance)
    t_values = params / std_error
    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_trainset) - len(new_trainset.columns)))) for i in t_values]
    
    
    std_error = np.round(std_error, 3)
    t_values = np.round(t_values, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    statistics = pd.DataFrame()
    statistics["Coefficients"], statistics["Standard Errors"], statistics["t -values"], statistics["p-values"] = [params, std_error, t_values, p_values]
    
    return p_values[0], p_values[1]

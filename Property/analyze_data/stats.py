import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import powerlaw
from IPython.display import display
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import os
from utils import *

dataset = ["email-Eu-full"]

# data statistics
for dataname in dataset:
    print(dataname)
    for distname in ["degree", "size", "sv", "effdiam"]:
        ret2, ret = read_properties(dataname, "../results/answer/" + dataname + "/")
        
        print("Numnode", ret["NumNode"])
        print("Numhedge", ret["NumHedge"])
        
        if distname in ["degree", "size"]:
            dist = ret[distname]
            x = sorted(list(dist.keys()))
            y = [dist[_x] for _x in x]
            avg = 0
            max_val = 0
            for _x, _y in zip(x, y):
                avg += _x * _y
                if _y > 0:
                    max_val = _x
            print(distname + " avg.", avg)
        
        elif distname in ["sv"]:
            dist = ret[distname]
            x = sorted(list(dist.keys()))
            y = [dist[_x] for _x in x]
            print(distname, y[0])
        elif distname in ["effdiam"]:
            print(distname, ret["effdiam"])
    print()
    print()
    print()

# powerlaw
properties = ["pairdeg", "intersection", "sv"]
for prop in properties:
    print(prop)
    d = pd.read_csv("../results/answer/powerlaw_test_" + prop + ".csv")
    display(d)

# linear regression
distset = ["degree", "size", "pairdeg", "intersection", "sv", "clusteringcoef_hedge", "density_dist", "overlapness_dist"]
for distname in distset:
    print(distname)
    print()
    column = ["Dataname", "Score", "Coef", "CoefPValue", "InterceptPValue"]
    with open(f"../results/answer/p_value_{distname}.txt", "w") as f:
        f.write(",".join(column) + "\n")
    
    for dataname in dataset:
        ret2, ret = read_properties(dataname, "../results/answer/" + dataname + "/")

        dist = ret[distname]
        x = sorted(list(dist.keys()))
        y = [dist[_x] for _x in x]

        if distname in ["degree", "size"]:
            select_length = int(0.75 * len(x))
            x, y = get_odds_ratio(x, y)
            _x = x[:select_length]
            _y = y[:select_length]
            score, coef, pred, intercept, reg = linearregression(_x, _y)
            pval1, pval2 = get_statistics(reg, intercept, coef, _x, _y)
        else:
            score, coef, pred, intercept, reg = linearregression(x, y)
            pval1, pval2 = get_statistics(reg, intercept, coef, x, y)

        with open(f"../results/answer/p_value_{distname}.txt", "a") as f:
            line = ",".join([dataname, "%.3f" % (score), "%.3f" % (coef), "%.6f" % (pval1), "%.6f" % (pval2)])
            f.write(line + "\n")
            
    d = pd.read_csv(f"../results/answer/p_value_{distname}.txt")
    display(d)
    print()
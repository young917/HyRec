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

plt.rcParams.update({'font.size': 15})

# setting --------------------------------------------------
xlabeldict = {
    "degree": "Node degree",
    "size": "Hyperedge size",
    "pairdeg": "Degree of node pairs",
    "intersection": "Intersection size",
    "sv": "Rank",
    "svt_u": "Rank",
    "svt_v": "Rank",
    "clusteringcoef_hedge": "Node degree",
    "density_dist": "# of nodes",
    "overlapness_dist": "# of nodes",
    
    "clusteringcoef": "v's degree in CE",
    "wcc": "Rank",
}

ylabeldict = {
    "degree": "OddsRatio",
    "size": "OddsRatio",
    "pairdeg": "PDF",
    "intersection": 'PDF',
    "sv": "Singular value",
    "svt_u": "Singular vector",
    "svt_v": "Singular vector",
    "clusteringcoef_hedge": "# of inter- \n secting pairs",
    "density_dist": "# of hyperedges",
    "overlapness_dist": r"$\sum$ hyperedge sizes",
    
    "clusteringcoef": "# triangles at v in CE",
    "wcc": "size of connected component",
}

colordict = {
    "email-Enron-full": "#e41a1c",
    "email-Eu-full": "#e41a1c",
    "contact-high-school": "#377eb8",
    "contact-primary-school": "#377eb8",
    "NDC-classes-full": "#4daf4a",
    "NDC-substances-full": "#4daf4a",
    "tags-ask-ubuntu": "#984ea3",
    "tags-math-sx": "#984ea3",
    "threads-ask-ubuntu": "#ff7f00",
    "threads-math-sx": "#ff7f00",
    "coauth-MAG-Geology-full": "#e6ab02",
    "coauth-MAG-History-full": "#e6ab02"
}

markerdict = {
    "email-Enron-full": "o",
    "email-Eu-full": "o",
    "contact-high-school": "^",
    "contact-primary-school": "^",
    "NDC-classes-full": "D",
    "NDC-substances-full": "D",
    "tags-ask-ubuntu": "P",
    "tags-math-sx": "P",
    "threads-ask-ubuntu": "s",
    "threads-math-sx": "s",
    "coauth-MAG-Geology-full": "<",
    "coauth-MAG-History-full": "<"
    
}
# -----------------------------------------------------------------


dataset = ["email-Eu-full"]
distset = ["degree", "size", "pairdeg", "intersection", "sv", "clusteringcoef_hedge", "density_dist", "overlapness_dist"]

# version 1 =======================================================================================================
for distname in distset:
    outputpath = "./" + distname + "/"
    if os.path.isdir(outputpath) is False:
        os.makedirs(outputpath)

    for dataname in dataset:
        outputpath += dataname + "_1.jpg"
        ret2, ret = read_properties(dataname, "../results/answer/" + dataname + "/")

        dist = ret[distname]
        x = sorted(list(dist.keys()))
        y = [dist[_x] for _x in x]
        score, coef, pred, intercept, reg = linearregression(x, y)

        if distname in ["degree", "size"]:
            select_length = int(0.75 * len(x))
            x, y = get_odds_ratio(x, y)
            _x = x[:select_length]
            _y = y[:select_length]
            score, coef, pred, intercept, reg = linearregression(_x, _y)

        plt.figure(figsize=(6,4), dpi=100)
        plt.scatter(x, y, zorder=2)
        line1 = plt.plot(x[:pred.shape[0]], pred, color="red", zorder=2)

        # Plot Angle & Score
        x_mid = 2 ** ((math.log2(min(x)) + math.log2(max(x))) / 2)
        pred_x_mid = 2 ** reg.predict(np.array([[math.log2(x_mid)]]))[0][0]
        len_x_mid = math.log2(x_mid) - math.log2(min(x))
        y_mid = 2 ** ((math.log2(min(pred)) + math.log2(max(pred))) / 2)
        xlength = 2 ** (len_x_mid * 0.2)
        xlength2 = 2 ** (len_x_mid * 0.3)
        ylength = 2 ** ((math.log2(pred[-1]) - math.log2(min(pred))) * 0.2)

        ax = plt.gca()
        ax.hlines(y=pred_x_mid, xmin=x_mid, xmax=x_mid * xlength, linewidth=2, color='red')

        if pred[0] > pred[-1]: # decreasing
            plt.text(x_mid * (xlength2 / 1.2), pred_x_mid, r"$\bf{\rho}$ = %.2f" % (coef), color="red", weight="bold", fontsize=28)
            plt.text(min(x), min(min(pred), min(y)), "Goodness of fit\n" + r"$\bf{R^{2}}$ = %.2f" % (score), color="red", weight="bold", verticalalignment="bottom", fontsize=28)
        else: # incresing
            plt.text(x_mid * (xlength2 / 1.2), pred_x_mid, r"$\bf{\rho}$ = %.2f" % (coef), color="red", weight="bold", fontsize=28)
            plt.text(min(x), max(max(pred), max(y)), "Goodness of fit\n" + r"$\bf{R^{2}}$ = %.2f" % (score), color="red", weight="bold", verticalalignment="top", fontsize=28)

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

        ax.tick_params(labelcolor='#4B4B4B', labelsize=26)
        plt.xlabel(xlabeldict[distname], fontsize=33)
        plt.ylabel(ylabeldict[distname], fontsize=33)

        plt.savefig(outputpath, bbox_inches='tight')
        plt.show()
        plt.close()

# Version 2 =======================================================================================================
for distname in distset:
    outputpath = "./" + distname + "/"
    if os.path.isdir(outputpath) is False:
        os.makedirs(outputpath)

    for dataname in dataset:
        outputpath += dataname + "_2.jpg"

        ret2, ret = read_properties(dataname, "../results/answer/" + dataname + "/")

        dist = ret[distname]
        x = sorted(list(dist.keys()))
        y = [dist[_x] for _x in x]
        score, coef, pred, intercept, reg = linearregression(x, y)

        if distname in ["degree", "size"]:
            select_length = int(0.75 * len(x))
            x, y = get_odds_ratio(x, y)
            _x = x[:select_length]
            _y = y[:select_length]
            score, coef, pred, intercept, reg = linearregression(_x, _y)

        plt.figure(figsize=(6,4), dpi=100)
        plt.scatter(x, y, zorder=2)
        line1 = plt.plot(x[:pred.shape[0]], pred, color="red", zorder=2)

        # Plot Angle & Score
        x_mid = 2 ** ((math.log2(min(x)) + math.log2(max(x))) / 2)
        pred_x_mid = 2 ** reg.predict(np.array([[math.log2(x_mid)]]))[0][0]
        len_x_mid = math.log2(x_mid) - math.log2(min(x))
        y_mid = 2 ** ((math.log2(min(pred)) + math.log2(max(pred))) / 2)
        xlength = 2 ** (len_x_mid * 0.2)
        xlength2 = 2 ** (len_x_mid * 0.3)
        ylength = 2 ** ((math.log2(pred[-1]) - math.log2(min(pred))) * 0.2)

        ax = plt.gca()

        if pred[0] > pred[-1]: # decreasing
            plt.text(min(x), min(min(pred), min(y)), r"$y  \propto x^{%.2f}$" % (coef), color="black", weight="bold", verticalalignment="bottom", fontsize=28)
            plt.text(min(x), min(min(pred), min(y)), "Goodness of fit\n" + r"$\bf{R^{2}}$ = %.2f" % (score), color="red", weight="bold", verticalalignment="bottom", fontsize=28)
        else: # incresing
            plt.text(min(x), max(max(pred), max(y)), r"$y  \propto x^{%.2f}$" % (coef), color="black", weight="bold", verticalalignment="top", fontsize=28)
            plt.text(min(x), max(max(pred), max(y))*0.9, "Goodness of fit\n" + r"$\bf{R^{2}}$ = %.2f" % (score), color="red", weight="bold", verticalalignment="bottom", fontsize=28)

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

        ax.tick_params(labelcolor='#4B4B4B', labelsize=26)
        plt.xlabel(xlabeldict[distname], fontsize=33)
        plt.ylabel(ylabeldict[distname], fontsize=33)

        plt.savefig(outputpath, bbox_inches='tight')
        #plt.show()
        plt.close()
    
# coefficients =======================================================================================================

from matplotlib.ticker import FormatStrFormatter

for distname in distset:
    aggregate_coef = {}
    aggregate_score = {}
    outputpath = "./" + distname + "/"
    if os.path.isdir(outputpath) is False:
        os.makedirs(outputpath)
    
    for dataname in dataset:
        ret2, ret = read_properties(dataname, "../results/answer/" + dataname + "/")
        dist = ret[distname]
        x = sorted(list(dist.keys()))
        y = [dist[_x] for _x in x]
        score, coef, pred, intercept, reg = linearregression(x, y)

        if distname in ["degree", "size"]:
            select_length = int(0.75 * len(x))
            x, y = get_odds_ratio(x, y)
            _x = x[:select_length]
            _y = y[:select_length]
            score, coef, pred, intercept, reg = linearregression(_x, _y)
        
        aggregate_coef[dataname] = coef
        aggregate_score[dataname] = score

    minusflag = False
    plt.figure(figsize=(6,4), dpi=100)
    xs, ys, cs = [], [], []
    for i in range(len(dataset)):
        dataname = dataset[i]
        xs.append(i)
        ys.append(aggregate_coef[dataname])
        plt.scatter(i, aggregate_coef[dataname], c=colordict[dataname], marker=markerdict[dataname], s=500, zorder=2)
    
    plt.tick_params( axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax = plt.gca()
    ax.tick_params(labelcolor='#4B4B4B', labelsize=26)
    
    plt.ylim((min(ys)-0.5, max(ys)+0.5))
    plt.xlabel("Dataset", fontsize=33)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.ylabel(r"Slope $\rho$" , fontsize=33)
    plt.savefig(outputpath + "coef.jpg", bbox_inches='tight')
    plt.show()
    plt.close()

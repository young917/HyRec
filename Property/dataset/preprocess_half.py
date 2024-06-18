import sys
import os
import numpy as np
import tqdm
from collections import defaultdict
import math

def main():
    datasetlist = ["email-Enron-full", "email-Eu-full", "contact-high-school", "contact-primary-school", 
                    "NDC-classes-full", "NDC-substances-full", "tags-ask-ubuntu", "threads-ask-ubuntu",  
                    "tags-math-sx", "threads-math-sx", "coauth-MAG-Geology-full"]
    
    for dataset in datasetlist:
        print(dataset)

        path = "../rawdata/" + dataset + '/'

        hedge2node = [] 
        node2index = {}
        
        timelist = []
        sizelist = []
        time2freq = defaultdict(int)
        with open(path + dataset + "-times.txt", "r") as f:
            for line in f.readlines():
                ts = int(line[:-1])
                timelist.append(ts)
                time2freq[ts] += 1
        min_timestamp = min(timelist)
        max_timestamp = max(timelist)

        with open(path + dataset + "-nverts.txt", "r") as f:
            for line in f.readlines():
                size = int(line[:-1])
                sizelist.append(size)

        total_nodeset = set()
        with open(path + dataset + "-simplices.txt", "r") as f:
            for line in f.readlines():
                node = int(line[:-1])
                total_nodeset.add(node)
        
        hidx_timestamp = 0
        hsize_timestamp = sizelist[hidx_timestamp]
        tmp = []
        with open(path + dataset + "-simplices.txt", "r") as f:
            for line in f.readlines():
                assert hidx_timestamp < len(sizelist)
                node = int(line[:-1])
                tmp.append(node)
                if len(tmp) == hsize_timestamp:
                    tmp = sorted(list(tmp))
                    hedge = []
                    # process timestamp
                    ts = timelist[hidx_timestamp]
                    # node reindex
                    for v in tmp:
                        if v not in node2index:
                            node2index[v] = len(node2index)
                        hedge.append(node2index[v])
                    # add hedge
                    hedge = sorted(list(set(hedge)))
                    hedge2node.append(hedge)
                    
                    if len(node2index) >= (len(total_nodeset) * (1./2)):
                        break
                    
                    # step to the next timestamp
                    hidx_timestamp += 1
                    if hidx_timestamp < len(sizelist):
                        hsize_timestamp = sizelist[hidx_timestamp]
                        tmp = []

        print(len(sizelist), len(hedge2node), "%.3f" % (len(hedge2node) / len(sizelist)))
        print(len(total_nodeset), len(node2index), "%.3f" % (len(node2index) / len(total_nodeset)))

        if "-full" in dataset:
            datasetname = dataset[:-5]
        else:
            datasetname = dataset
        with open("./" + datasetname + "-half.txt", "w") as f:
            for nodes in hedge2node:
                line = ','.join([str(x) for x in nodes]) + '\n'
                f.write(line)
        print()


if __name__ == '__main__':
    main()
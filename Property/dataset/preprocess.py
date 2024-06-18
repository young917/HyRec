import sys
import os
import numpy as np
import tqdm
from collections import defaultdict

def main():
    datasetlist = ["email-Enron-full", "email-Eu-full", "contact-high-school", "contact-primary-school", 
                    "NDC-classes-full", "NDC-substances-full", "tags-ask-ubuntu", "threads-ask-ubuntu",  
                    "tags-math-sx", "threads-math-sx", "coauth-MAG-Geology-full"]

    for dataset in datasetlist:
        print(dataset)

        path = "../rawdata/" + dataset + '/'
        
        timelist = []
        sizelist = []
        with open(path + dataset + "-times.txt", "r") as f:
            for line in f.readlines():
                ts = int(line[:-1])
                timelist.append(ts)

        with open(path + dataset + "-nverts.txt", "r") as f:
            for line in f.readlines():
                size = int(line[:-1])
                sizelist.append(size)
        
        hedge2node = []
        hedge2timestamp = []
        node2index = {}

        hidx_timestamp = 0
        hsize_timestamp = sizelist[hidx_timestamp]
        tmp = []
        with open(path + dataset + "-simplices.txt", "r") as f:
            for line in f.readlines():
                assert hidx_timestamp < len(sizelist)
                node = int(line[:-1])
                tmp.append(node)
                if len(tmp) == hsize_timestamp:
                    ts = timelist[hidx_timestamp]
                    tmp = sorted(list(set(tmp)))
                    hedge = []
                    for v in tmp:
                        if v not in node2index:
                            node2index[v] = len(node2index)
                        hedge.append(node2index[v])
                    hedge = sorted(hedge)

                    hedge2timestamp.append(ts)
                    hedge2node.append(hedge)
                    
                    # step to the next timestamp
                    hidx_timestamp += 1
                    if hidx_timestamp < len(sizelist):
                        hsize_timestamp = sizelist[hidx_timestamp]
                        tmp = []
        
        assert len(hedge2node) == len(hedge2timestamp)
        sorted_index = sorted(range(len(hedge2node)), key=lambda x:hedge2timestamp[x])
        hedge2node = [hedge2node[idx] for idx in sorted_index]
        hedge2timestamp = [hedge2timestamp[idx] for idx in sorted_index]
        for ts_idx in range(1, len(hedge2timestamp)):
            assert hedge2timestamp[ts_idx - 1] <= hedge2timestamp[ts_idx]
                    
        with open("./" + dataset + ".txt", "w") as f:
            for hidx, nodes in enumerate(hedge2node):
                line = ','.join([str(x) for x in nodes]) + '\n'
                f.write(line)

        


if __name__ == '__main__':
    main()
import os
datasetlist = ["email-Enron-full", "email-Eu-full", "contact-high-school", "contact-primary-school", "NDC-classes-full", "NDC-substances-full", "tags-ask-ubuntu", "threads-ask-ubuntu", "tags-math-sx", "threads-math-sx", "coauth-MAG-Geology-full"]
halfdatasetlist = ["email-Enron-half", "email-Eu-half", "contact-high-school-half", "contact-primary-school-half", "NDC-classes-half", "NDC-substances-half", "tags-ask-ubuntu-half", "tags-math-sx-half", "threads-ask-ubuntu-half", "threads-math-sx-half", "coauth-MAG-Geology-half"]

for dataname in datasetlist + halfdatasetlist:
    hedge2node = []
    nodereindex = {}
    with open("./" + dataname + ".txt", "r") as f:
        for line in f.readlines():
            tmp = line.rstrip().split(",")
            hedge = set()
            for vstr in tmp:
                if vstr not in nodereindex:
                    nodereindex[vstr] = len(nodereindex)
                hedge.add(nodereindex[vstr])
            hedge = sorted(list(hedge))
            hedge2node.append(hedge)

    with open("../../Model/input/" + dataname + ".txt", "w") as f:
        # firstline = "#hedges" "#nodes"
        f.write(str(len(hedge2node)) + " " + str(len(nodereindex)) + "\n")
        # line = "e" "v" "1"
        for hi, hedge in enumerate(hedge2node):
            for vi in hedge:
                f.write(str(hi) + " " + str(vi) + " 1\n")
    print("Process " + dataname)

    
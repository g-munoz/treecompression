from glob import glob
import pandas as pd
import sys

treetypes = ["FSB", "RBp"]
methods = ["drop", "supp:1", "supp:2", "supp:inf"]

instancenames = [
    f.replace("instances/models/", "").replace(".mps.gz", "")
    for f in sorted(glob("instances/models/miplib3/*.mps.gz"))
]

cols = pd.MultiIndex.from_product([treetypes,methods])
compressdf = pd.DataFrame(index=instancenames, columns=cols)
timedf = pd.DataFrame(index=instancenames, columns=cols)

cols = pd.MultiIndex.from_product([treetypes,methods,["Root","Avg"]])
nodetimesdf = pd.DataFrame(index=instancenames, columns=cols)
nodetimesdict = {}

for tree in treetypes:
    for instance in instancenames:
        for method in methods:
            try:
                filename = f"results/{tree}/{method}/{instance}.out"
                print(filename)
                with open(filename) as origin_file:
                    for line in origin_file:
                        if line:
                            line = line.split(' ')
                            if line[0]=="SUMMARY:":
                                name = line[1]
                                orignodes = int(line[3])
                                newnodes = int(line[5])
                                time = float(line[12])

                                compressdf.at[instance, (tree,method)] = (1 - newnodes/orignodes) * 100
                                timedf.at[instance, (tree,method)] = time

                                nodetimesdf.at[instance,(tree,method,"Avg")] = nodetimesdict[instance,tree,method,"Sum"]/nodetimesdict[instance,tree,method,"Count"]
                            elif line[0]=="NODEINFO:":
                                nodeid = int(line[1])
                                nodetime = float(line[2])
                                if (instance,tree,method,"Sum") in nodetimesdict.keys():
                                    nodetimesdict[instance,tree,method,"Sum"] += nodetime
                                    nodetimesdict[instance,tree,method,"Count"] += 1
                                else:
                                    nodetimesdict[instance,tree,method,"Sum"] = nodetime
                                    nodetimesdict[instance,tree,method,"Count"] = 1

                                if nodeid == 0:
                                    nodetimesdf.at[instance,(tree,method,"Root")] = nodetime
            except:
                pass

with pd.ExcelWriter('viz/summary_miplib3.xlsx') as writer:  
    compressdf.to_excel(writer, sheet_name='CompressionRatios')
    timedf.to_excel(writer, sheet_name='Time')
    nodetimesdf.to_excel(writer,sheet_name='NodeTimes')

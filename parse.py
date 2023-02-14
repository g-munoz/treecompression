from glob import glob
import pandas as pd
import sys
import json
import logging

treetypes = ["RB"]
log_methods = ["drop", "supp:1", "supp:2", "supp:inf"]
json_methods = ["heuristic", "drop"]
methods = log_methods + json_methods

instancenames = [
    f.replace("instances/models/", "").replace(".mps.gz", "")
    for f in sorted(glob("instances/models/miplib2017/*.mps.gz"))
]

data = []

for tree in treetypes:
    for instance in instancenames:
        # Parse log files (produced by compress.py)
        for method in log_methods:
            filename = f"results/{tree}/{method}/{instance}.out"
            print(filename)
            try:
                with open(filename) as origin_file:
                    for line in origin_file:
                        if line:
                            line = line.split(' ')
                            if line[0]=="SUMMARY:":
                                name = line[1]
                                orignodes = int(line[3])
                                newnodes = int(line[5])
                                time = float(line[12])
                                data.append({
                                    "instance": instance,
                                    "tree": tree,
                                    "method": method,
                                    "compress": (1 - newnodes/orignodes) * 100,
                                    "time": time,
                                    "orignodes": orignodes,
                                    "newnodes": newnodes,
                                })
            except:
                logging.exception(f"Failed to process {filename}")

        # Parse JSON files (produced by heuristics.py)
        for method in json_methods:
            filename = f"results/{tree}/{method}/{instance}.stats.json"
            try:
                print(filename)
                with open(filename) as file:
                    stats = json.load(file)
                orignodes = stats["nodes_before"]
                newnodes = stats["nodes_after"]
                time = stats["time"]
                data.append({
                    "instance": instance,
                    "tree": tree,
                    "method": method,
                    "compress": (1 - newnodes/orignodes) * 100,
                    "time": time,
                    "orignodes": orignodes,
                    "newnodes": newnodes,
                })
            except:
                logging.exception(f"Failed to process {filename}")


data = pd.DataFrame(data)
data.to_csv("viz/results.csv", index=False)

# with pd.ExcelWriter('viz/summary_miplib3.xlsx') as writer:  
#     compressdf.to_excel(writer, sheet_name='CompressionRatios')
#     timedf.to_excel(writer, sheet_name='Time')

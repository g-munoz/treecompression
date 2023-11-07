from pydoc import plain
from gurobipy import *
import sys
import os
import pandas as pd
import numpy as np
import argparse
import json
import time

timelimit = 600

parser = argparse.ArgumentParser(description='Test Disjuctions')

# filename required
parser.add_argument('filename', #type=open,
                    help='MPS or LP Model')

parser.add_argument('--timelimit', type=int,
                    help='Solver time limit')

parser.add_argument('--disjfile', help='File with disjunctions')

parser.add_argument('--strongbranching', action='store_true',
                    help='Use strong branching')

parser.add_argument('--quiet', action='store_true',
                    help='quiet')

args = parser.parse_args()

modelname = args.filename
print(modelname)
plainmodel = read(modelname)
disjmodel = plainmodel.copy()

filename, file_extension = modelname.split(os.extsep,1)

f = open(args.disjfile)
disjdata = json.load(f)

if args.timelimit != None:
	timelimit = args.timelimit

if args.quiet:
    disjmodel.setParam("OutputFlag",0)
    plainmodel.setParam("OutputFlag",0)

if args.strongbranching:
    disjmodel.setParam("VarBranch",3)
    plainmodel.setParam("VarBranch",3)  

disjmodel.setParam("TimeLimit",timelimit)
disjmodel.setParam("Threads",1)
disjmodel.setParam("Presolve",0)

plainmodel.setParam("TimeLimit",timelimit)
plainmodel.setParam("Threads",1)
plainmodel.setParam("Presolve",0)


nodecounts = {}
nodedisjunctions = []
tol = 1E-5

variables = disjmodel.getVars()
for i in disjdata.keys():
    for node in disjdata[i]["Nodes"]:
        disj = np.array(disjdata[i]["Nodes"][node]["pi"])
        #disj.append(disjdata[i]["Nodes"][node]["pi0"])

        alreadyadded = False
        for d in nodedisjunctions:
            if np.linalg.norm(disj-d) < tol:
                  alreadyadded = True
                  #print("Repeated disjunction")
                  break
        
        if not alreadyadded:
            nodedisjunctions.append(disj)
            z = disjmodel.addVar(vtype=GRB.INTEGER, lb= -GRB.INFINITY)
            disjmodel.addConstr(z == np.dot(disj,variables))
        
#print("Disj Collection Size", len(nodedisjunctions))
#plainmodel.write("orig.lp")
#disjmodel.write("disjs.lp")

plainmodel.optimize()
if not args.quiet:
    print("\n========\n")
disjmodel.optimize()

print("INFO",modelname,plainmodel.NodeCount,disjmodel.NodeCount,plainmodel.MIPGap,disjmodel.MIPGap,plainmodel.Runtime,disjmodel.Runtime)

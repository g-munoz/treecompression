from gurobipy import *
import sys
import os
import numpy as np
import argparse
import json
import time

variance = 0.05
totalnumber = 10
parser = argparse.ArgumentParser(description='Test Disjuctions')

# filename required
parser.add_argument('filename', #type=open,
                    help='MPS or LP Model')

parser.add_argument('--variance', type=float,
                    help='Perturbation variance')

parser.add_argument('--totalnumber', type=int,
                    help='Number of new instances')


args = parser.parse_args()

modelname = args.filename
print(modelname)
plainmodel = read(modelname)

filename, file_extension = modelname.split(os.extsep,1)

if args.totalnumber != None:
	totalnumber = args.totalnumber

if args.variance != None:
    variance = args.variance
    #print("INFO",variance)

obj = np.array(plainmodel.getAttr("Obj",plainmodel.getVars()))
for iter in range(totalnumber):

    objnew = obj*(np.ones(obj.shape) + np.random.normal(0, variance, obj.shape))
    plainvars = plainmodel.getVars()

    for i in range(len(plainvars)):
         plainvars[i].Obj = objnew[i]

    plainmodel.update()
    plainmodel.write(filename+"_rnd"+str(iter)+".mps")


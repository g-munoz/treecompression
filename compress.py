from gurobipy import *
import pandas as pd
import random
import sys
import os
import numpy as np
import argparse
import json
import time
from queue import Queue
from disjunction import *

nodetimelimit = 300
globaltimelimit = 3600
restrictedsupp = False
starttime = 0

def downtreesearch(node, tree):

	nodecount = 1
	children = tree["nodes"][node]["children"]	
	if len(children) == 0:
		print("INFO: Reached leaf", node)
		return nodecount

	if time.time() - starttime < globaltimelimit: #if we still have some time left
		
		success = main(node,tree)
	else:
		success = False
		
	if success:
		print("INFO: Subtree rooted at", node, "compressed")
	else:
		for i in children:
			nodecount = nodecount + downtreesearch(i, tree) ##Ugly hack: when time limit is hit, it will still go down the tree, but just for adding nodes. 

	return nodecount
	
def uptreesearch(tree):
	Q = Queue()
	remaining_children = {}
	success_mem = set()

	for i in tree["nodes"]:
		remaining_children[i] = len(tree["nodes"][i]["children"]) # we store the number of children remaining to "succeed"
		if len(tree["nodes"][i]["children"]) == 0: #queue leaves of the tree
			#print("INFO: Enqueued",i)
			Q.put(i)
			
	while not Q.empty() and time.time() - starttime < globaltimelimit:
		node = Q.get() #this queue should only have nodes whose children all succeeded. We might want to change this
		
		print("INFO: Processing",node)
		success = False
		
		if len(tree["nodes"][node]["children"]) == 0: #leaves are compressed, so we don't run anything. This is a bit ugly
			success = True
		else:
			success = main(node,tree)
		if success:
			#print("INFO: Success",node)
			print("INFO: Subtree rooted at", node, "compressed")
			success_mem.add(node)
			
		#else:
		#	print("Failed",node)
		#	#input()
		
		parent = tree["nodes"][node]["parent"]
		if parent != None :
			remaining_children[parent] = remaining_children[parent] - 1 ## decrease the parent count 
			if remaining_children[parent] == 0: ## parent is added to queue if all its children are done
				Q.put(parent)

	nodecount = countupcompression(tree,str(0),success_mem)

	return nodecount
	
def countupcompression(tree,node,success_mem):

	nodecount = 1
	children = tree["nodes"][node]["children"]	
	if len(children) == 0:
		return nodecount

	if node not in success_mem:
		for i in children:
			nodecount = nodecount + countupcompression(tree,i,success_mem)
			
	return nodecount

def main(node_id, tree):
	#this function tries to compress the subtree rooted at node_id

	modelname = sys.argv[1]
	model = read(modelname)
	#filename, file_extension = os.path.splitext(modelname)
	filename, file_extension = modelname.split(os.extsep,1)

	#node_id = sys.argv[2]
	#print(tree["nodes"][str(node_id)])
	#parent = tree["nodes"][str(node_id)]["parent"]
	curr_node = str(node_id)

	#this while loop goes UP the tree collecting the branching bounds
	while True:
		if curr_node == "0":
			break
		var = model.getVarByName(tree["nodes"][curr_node]["branch_var"])
		lb = tree["nodes"][curr_node]["branch_lb"]
		ub = tree["nodes"][curr_node]["branch_ub"]
		
		model.addConstr(var <= ub)
		model.addConstr(var >= lb)
		
		curr_node = tree["nodes"][curr_node]["parent"]

	model.update()
	
	nodefilename = filename+"_node"+str(node_id)+".lp"
	#model.write(nodefilename)
	
	subtreesupp = tree["nodes"][node_id]["subtree_support"]
	subtreebound = tree["nodes"][node_id]["subtree_bound"]
	
	args = []
	#args.append(nodefilename)
	args.append(model)
	args.append(str(subtreebound))
	
	#restrict support if requires
	if restrictedsupp:
		#print("\n\nUsing support\n\n")
		for i in subtreesupp:
			args.append(str(i)) #we accumulate
	
	return findDisjunction(args, nodetimelimit)

parser = argparse.ArgumentParser(description='Run tree search')

# filename required
parser.add_argument('filename', #type=open,
                    help='MPS or LP Model')

# optional time arguments
parser.add_argument('--nodetime', type=int,
                    help='Time limit per node compression')
parser.add_argument('--globaltime', type=int,
                    help='Global time limit')

# switch for restricting support
parser.add_argument('--restrictedsupp', action='store_true',
                    help='Restrict to subtree support')
# switch for doing tree search from the bottom
parser.add_argument('--upsearch', action='store_true',
                    help='Do tree search from bottom to top')

args = parser.parse_args()

modelname = args.filename
model = read(modelname)
filename, file_extension = modelname.split(os.extsep,1)
jsonname = filename + ".tree.json"
f = open(jsonname)
tree = json.load(f)

restrictedsupp = args.restrictedsupp

if args.nodetime != None:
	nodetimelimit = args.nodetime
if args.globaltime != None:
	globaltimelimit = args.globaltime

starttime = time.time()

if not args.upsearch:
	nodecount = downtreesearch(str(0),tree)
else:
	nodecount = uptreesearch(tree)

print("SUMMARY:", modelname,"Compressed", len(tree["nodes"]), "to", nodecount, "Support restricted=", args.restrictedsupp, "Upsearch=", args.upsearch, "Time=", time.time() - starttime)


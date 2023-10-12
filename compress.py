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
drop = True
usesubtreebound = False

disjcoefbound = GRB.INFINITY
disjsuppsize = GRB.INFINITY

dropnodecount = 0
compressnodecount = 0

current_seed = 00000
disjsummary = {}

def canbeDropped(node,tree):
	nodebnd = float(tree["nodes"][node]["obj"])
	globalbnd = float(tree["nodes"]["0"]["subtree_bound"])

	if tree["sense"] == "min" and nodebnd >= globalbnd :
		return True
	if tree["sense"] == "max" and nodebnd <= globalbnd :
		return True

	return False

def downtreesearchDFS(node, tree):

	global compressnodecount
	global dropnodecount
	global current_seed
	global disjsummary
	#print("DFS visiting node ", node)
	nodecount = 1
	nodesvisited = 1
	
	children = tree["nodes"][node]["children"]	
	if len(children) == 0:
		print("INFO: Reached leaf", node)
		return nodecount, nodesvisited

	if drop and canbeDropped(node,tree):
		dropnodecount += 1
		return nodecount, nodesvisited

	if time.time() - starttime < globaltimelimit - nodetimelimit: #if we still have some time left	
		success, runtime, pi, pi0 = main(node,tree)
		print("NODEINFO:", node, runtime, success)

		if success:
			disjsummary[current_seed]["Nodes"][node] = {}
			disjsummary[current_seed]["Nodes"][node]['pi'] = pi.tolist()
			disjsummary[current_seed]["Nodes"][node]['pi0'] = pi0
	else:
		success = False
		nodesvisited = 0
		
	if success:
		print("INFO: Subtree rooted at", node, "compressed")
		compressnodecount += 1
		nodecount = nodecount + 2 #we add 2, since the compression is done via a disjunction that would create to children. Note that here we are not on a leaf.
	else:
		for i in children:
			nodecount_down, nodesvisited_down = downtreesearchDFS(i, tree) ##Ugly hack: when time limit is hit, it will still go down the tree, but just for adding nodes. 
			nodecount = nodecount + nodecount_down ##Ugly hack: when time limit is hit, it will still go down the tree, but just for adding nodes. 
			nodesvisited = nodesvisited + nodesvisited_down

	return nodecount, nodesvisited
	

def downtreesearchBFS(startnode, tree):

	global compressnodecount
	global dropnodecount

	nodecount = 0
	nodesvisited = 0
	
	Q = Queue()
	Q.put(startnode)

	while not Q.empty():
		node = Q.get()
		#print("BFS visiting node ", node)
		nodecount += 1
		nodesvisited += 1

		children = tree["nodes"][node]["children"]	
		if len(children) == 0:
			print("INFO: Reached leaf", node)
			continue

		if drop and canbeDropped(node,tree):
			dropnodecount += 1	
			continue

		if time.time() - starttime < globaltimelimit - nodetimelimit: #if we still have some time left	
			success, runtime, pi, pi0 = main(node,tree)
			print("NODEINFO:", node, runtime, success)

			if success:
				disjsummary[current_seed]["Nodes"][node] = {}
				disjsummary[current_seed]["Nodes"][node]['pi'] = pi.tolist()
				disjsummary[current_seed]["Nodes"][node]['pi0'] = pi0
		else:
			success = False
			nodesvisited -= 1
		
		if success:
			print("INFO: Subtree rooted at", node, "compressed")
			compressnodecount += 1
			nodecount = nodecount + 2 #we add 2, since the compression is done via a disjunction that would create to children. Note that here we are not on a leaf.
		else:
			for i in children:
				Q.put(i)

	return nodecount, nodesvisited

def uptreesearch(tree):
	Q = Queue()
	remaining_children = {}
	success_mem = set()
	nodesvisited = 0

	for i in tree["nodes"]:
		remaining_children[i] = len(tree["nodes"][i]["children"]) # we store the number of children remaining to "succeed"
		if len(tree["nodes"][i]["children"]) == 0: #queue leaves of the tree
			#print("INFO: Enqueued",i)
			Q.put(i)
			
	while not Q.empty() and time.time() - starttime < globaltimelimit - nodetimelimit:
		node = Q.get() #this queue should only have nodes whose children all succeeded. We might want to change this
		
		print("INFO: Processing",node)
		success = 0
		
		if len(tree["nodes"][node]["children"]) == 0: #leaves are compressed, so we don't run anything. This is a bit ugly
			success = 2
		else:
			canDrop = drop and canbeDropped(node,tree)
			if canDrop: 
				success = 2
			else:
				success, runtime, pi, pi0 = main(node,tree)
				if success:
					disjsummary[current_seed]["Nodes"][node] = {}
					disjsummary[current_seed]["Nodes"][node]['pi'] = pi.tolist()
					disjsummary[current_seed]["Nodes"][node]['pi0'] = pi0
			
		nodesvisited += 1
		
		if success:
			#print("INFO: Success",node)
			print("INFO: Subtree rooted at", node, "compressed")
			success_mem.add((node,success)) #there can be two types of success: drop or disjunction
			
		#else:
		#	print("Failed",node)
		#	#input()
		
		parent = tree["nodes"][node]["parent"]
		if parent != None :
			remaining_children[parent] = remaining_children[parent] - 1 ## decrease the parent count 
			if remaining_children[parent] == 0: ## parent is added to queue if all its children are done
				Q.put(parent)

	nodecount = countupcompression(tree,str(0),success_mem)

	return nodecount, nodesvisited
	
def countupcompression(tree,node,success_mem):
	global compressnodecount
	global dropnodecount
	nodecount = 1
	children = tree["nodes"][node]["children"]	
	if len(children) == 0:
		return nodecount

	if (node,1) in success_mem:
		nodecount = nodecount + 2 #in the successful case, non-leaf, the compression uses 2 extra nodes
		compressnodecount+= 1
	elif (node,2) in success_mem: #the drop case does no add
		dropnodecount += 1
	else: 
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

	if usesubtreebound:
		bound = tree["nodes"][node_id]["subtree_bound"]
	else:
		bound = tree["nodes"]["0"]["subtree_bound"] #use root node bound (loosest)
	
	args = []
	#args.append(nodefilename)
	args.append(model)
	args.append(str(bound))
	
	#restrict support if requires
	if restrictedsupp:
		#print("\n\nUsing support\n\n")
		for i in subtreesupp:
			args.append(str(i)) #we accumulate
	
	return findDisjunction(args, nodetimelimit, disjcoefbound, disjsuppsize, current_seed)

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

parser.add_argument('--nodrop', action='store_true',
                    help='Disable drop operation')

parser.add_argument('--usesubtreebound', action='store_true',
                    help='Use subtree bound instead of global dual bound')

parser.add_argument('--disjcoefbound', type=int,
                    help='Disjunction coefficient bound')

parser.add_argument('--disjsuppsize', type=int,
                    help='Disjunction support size')

parser.add_argument('--treename', help='Tree to compress')

parser.add_argument('--bfs', action='store_true',
                    help='Use BFS to search the tree')

args = parser.parse_args()

modelname = args.filename
print(modelname)
model = read(modelname)
filename, file_extension = modelname.split(os.extsep,1)

jsonname = filename + ".tree.json"

if args.treename != None:
	jsonname = args.treename

f = open(jsonname)
tree = json.load(f)

restrictedsupp = args.restrictedsupp

if args.nodetime != None:
	nodetimelimit = args.nodetime
if args.globaltime != None:
	globaltimelimit = args.globaltime

if args.disjcoefbound != None and args.disjcoefbound >= 0: #we use -1 to flag unbounded case
	disjcoefbound = args.disjcoefbound
if args.disjsuppsize != None and args.disjsuppsize >= 0: #we use -1 to flag unbounded case
	disjsuppsize = args.disjsuppsize

if args.nodrop:
	drop = False
if args.usesubtreebound:
	usesubtreebound = True

seeds = [11111, 22222, 12345, 321321, 987789]
#seeds = [11111, 22222]

for i in seeds:
	current_seed = i
	disjsummary[i] = {}
	disjsummary[i]["Nodes"] = {}

	starttime = time.time()
	nodesvisited = 0
	compressnodecount = 0
	dropnodecount = 0

	if not args.upsearch:
		if args.bfs:
			nodecount, nodesvisited = downtreesearchBFS(str(0),tree)
		else:
			nodecount, nodesvisited = downtreesearchDFS(str(0),tree)
	else:
		nodecount, nodesvisited = uptreesearch(tree)

	disjsummary[i]["CompTreeSize"] = nodecount
	print("SUMMARY:", modelname,"Compressed", len(tree["nodes"]), "to", nodecount, "Support restricted=", args.restrictedsupp, "Upsearch=", args.upsearch, "Time=", time.time() - starttime, "Nodes_Visited=",nodesvisited, "Disj/Drop Nodes",compressnodecount,dropnodecount )

disjunctionsout_name = filename + "disjunctions.tree.json" ## TODO: this name should also have the treename used
with open(disjunctionsout_name, 'w') as fp:
	json.dump(disjsummary, fp, indent=4)

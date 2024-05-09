from gurobipy import *
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
starttime = 0
drop = True
usesubtreebound = False

disjcoefbound = GRB.INFINITY
disjsuppsize = GRB.INFINITY

dropnodecount = 0
compressnodecount = 0

current_seed = 00000
newnodesid = 0

def canbeDropped(node,tree):
	nodebnd = float(tree["nodes"][node]["obj"])
	globalbnd = float(tree["nodes"]["0"]["subtree_bound"])

	if tree["sense"] == "min" and nodebnd >= globalbnd :
		return True
	if tree["sense"] == "max" and nodebnd <= globalbnd :
		return True

	return False

def addDisjunctionToTree(tree, node, compressedtree, pi, pi0, obj1, obj2):
	global newnodesid

	child1 = str(newnodesid)
	compressedtree["nodes"][child1] = {}
	compressedtree["nodes"][child1]["id"] = child1
	compressedtree["nodes"][child1]["parent"] = node
	compressedtree["nodes"][child1]["children"] = []
	compressedtree["nodes"][child1]["depth"] = tree["nodes"][node]["depth"] + 1
	compressedtree["nodes"][child1]["obj"] = obj1
	compressedtree["nodes"][child1]["branch_lhs"] = {}

	for i in range(len(pi)):
		if pi[i] <= 1E-3:
			continue
		varname = idToNameDictionary[i]
		compressedtree["nodes"][child1]["branch_lhs"][varname] = pi[i]

	compressedtree["nodes"][child1]["branch_lb"] = -math.inf
	compressedtree["nodes"][child1]["branch_ub"] = pi0

	child2 = str(newnodesid+1)
	compressedtree["nodes"][child2] = {}
	compressedtree["nodes"][child2]["id"] = child2
	compressedtree["nodes"][child2]["parent"] = node
	compressedtree["nodes"][child2]["children"] = []
	compressedtree["nodes"][child2]["depth"] = tree["nodes"][node]["depth"] + 1
	compressedtree["nodes"][child2]["obj"] = obj2
	compressedtree["nodes"][child2]["branch_lhs"] = compressedtree["nodes"][child1]["branch_lhs"]

	compressedtree["nodes"][child2]["branch_lb"] = pi0 + 1
	compressedtree["nodes"][child2]["branch_ub"] = math.inf

	newnodesid += 2

def downtreesearchDFS(node, tree, compressedtree):

	global compressnodecount
	global dropnodecount
	global current_seed
	#print("DFS visiting node ", node)
	nodecount = 1
	nodesvisited = 1

	#copy current node info in tree
	compressedtree["nodes"][node] = tree["nodes"][node]
	
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
			addDisjunctionToTree(tree, node, compressedtree, pi, pi0, obj1, obj2)
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
	

def downtreesearchBFS(startnode, tree,compressedtree):

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

		#copy current node info in tree
		compressedtree["nodes"][node] = tree["nodes"][node]

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
				addDisjunctionToTree(tree, node, compressedtree, pi, pi0, obj1, obj2)
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

	args.append(str(node_id))
	
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
#seeds = [11111]

for i in seeds:
	current_seed = i

	compressedtree = {}
	compressedtree["sense"] = tree["sense"]
	compressedtree["nodes"] = {}
	global newnodesid
	newnodesid = len(tree["sense"]["nodes"])

	starttime = time.time()
	nodesvisited = 0
	compressnodecount = 0
	dropnodecount = 0

	if args.bfs:
		nodecount, nodesvisited = downtreesearchBFS(str(0),tree,compressedtree)
	else:
		nodecount, nodesvisited = downtreesearchDFS(str(0),tree,compressedtree)
	
	print("SUMMARY:", modelname,"Compressed", len(tree["nodes"]), "to", nodecount, "Time=", time.time() - starttime, "Nodes_Visited=",nodesvisited, "Disj/Drop Nodes",compressnodecount,dropnodecount )

disjunctionsout_name = filename + "disjunctions.tree.json" ## TODO: this name should also have the treename used
with open(disjunctionsout_name, 'w') as fp:
	json.dump(disjsummary, fp, indent=4)

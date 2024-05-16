from pyscipopt import *
import random
import sys
import os
import numpy as np
import argparse
import json
import time
from queue import Queue
from disjunctionSCIP import *

nodetimelimit = 300
globaltimelimit = 3600

starttime = 0
drop = True
usesubtreebound = False

disjcoefbound = None
disjsuppsize = None

dropnodecount = 0
compressnodecount = 0

current_seed = 00000
newnodesid = 0

varNameDictionary = {}
idToNameDictionary = {}
originalBounds = {}

def getVarByName(name):
	global varNameDictionary
	return varNameDictionary[name]

def saveVarAttributes(model):
	global varNameDictionary
	global originalBounds
	global idToNameDictionary

	vars = model.getVars()
	for i in range(len(vars)):
		x = vars[i]
		varNameDictionary[x.name] = x
		idToNameDictionary[i] = x.name
		originalBounds[i] = [x.getLbGlobal(), x.getUbGlobal()]

def canbeDropped(node,tree):
	nodebnd = float(tree["nodes"][node]["obj"])
	globalbnd = float(tree["nodes"]["0"]["subtree_bound"])

	if tree["sense"] == "min" and nodebnd >= globalbnd :
		return True
	if tree["sense"] == "max" and nodebnd <= globalbnd :
		return True

	return False

def restoreOriginalBounds(model):
	vars = model.getVars()
	for i in range(len(vars)):
		x = vars[i]
		model.chgVarLb(x,originalBounds[i][0])
		model.chgVarUb(x,originalBounds[i][1])

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

	compressedtree["nodes"][node]["children"] = [child1, child2]
	if compressedtree["sense"] == "min":
		compressedtree["nodes"][node]["subtree_bound"] = min(obj1, obj1)
	else:
		compressedtree["nodes"][node]["subtree_support"] = max(obj1, obj1)
	compressedtree["nodes"][node]["subtree_support"] = [] #TODO FILL

	newnodesid += 2

def downtreesearchDFS(node, tree, compressedtree, model):

	global compressnodecount
	global dropnodecount
	global current_seed
	global newnodesid

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
		success, runtime, pi, pi0, obj1, obj2 = processNode(node,tree,model)
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
			nodecount_down, nodesvisited_down = downtreesearchDFS(i, tree, compressedtree,model) ##Ugly hack: when time limit is hit, it will still go down the tree, but just for adding nodes. 
			nodecount = nodecount + nodecount_down 
			nodesvisited = nodesvisited + nodesvisited_down

	return nodecount, nodesvisited
	
def downtreesearchBFS(startnode, tree, compressedtree, model):

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
			success, runtime, pi, pi0, obj1, obj2 = processNode(node,tree,model)
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

def processNode(node_id, tree, model):
	#this function tries to compress the subtree rooted at node_id

	curr_node = str(node_id)

	#this while loop goes UP the tree collecting the branching bounds
	while curr_node != "0":
		var = getVarByName(tree["nodes"][curr_node]["branch_var"])
		lb = tree["nodes"][curr_node]["branch_lb"]
		ub = tree["nodes"][curr_node]["branch_ub"]
		
		model.chgVarUb(var, ub)
		model.chgVarLb(var, lb)
		
		curr_node = tree["nodes"][curr_node]["parent"]

	if usesubtreebound:
		bound = tree["nodes"][node_id]["subtree_bound"]
	else:
		bound = tree["nodes"]["0"]["subtree_bound"] #use root node bound (loosest)
	
	args = []
	args.append(model)
	args.append(str(bound))
	args.append(str(node_id))
	
	success, runtime, pi, pi0, obj1, obj2 = findDisjunction(args, nodetimelimit, disjcoefbound, disjsuppsize, current_seed)

	restoreOriginalBounds(model)

	return success, runtime, pi, pi0, obj1, obj2

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
model = Model()
model.readProblem(modelname)

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

#seeds = [11111, 22222, 12345, 321321, 987789]
seeds = [111, 222]
#seeds = [111]

saveVarAttributes(model)

for i in seeds:
	current_seed = i

	compressedtree = {}
	compressedtree["sense"] = tree["sense"]
	compressedtree["nodes"] = {}

	newnodesid = len(tree["nodes"])

	starttime = time.time()
	nodesvisited = 0
	compressnodecount = 0
	dropnodecount = 0

	if args.bfs:
		nodecount, nodesvisited = downtreesearchBFS(str(0),tree,compressedtree,model)
	else:
		nodecount, nodesvisited = downtreesearchDFS(str(0),tree,compressedtree,model)

	print("SUMMARY:", modelname,"Compressed", len(tree["nodes"]), "to", nodecount, "Time=", time.time() - starttime, "Nodes_Visited=",nodesvisited, "Disj/Drop Nodes",compressnodecount,dropnodecount )

	treeoutname = filename + ".tree.compressed.s"+str(current_seed)+".json" 
	with open(treeoutname, 'w') as fp:
		json.dump(compressedtree, fp, indent=4)

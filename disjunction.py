from gurobipy import *
import pandas as pd
import random
import sys
import os
import numpy as np

from scipy.sparse import csr_matrix
#from scipy.sparse import csr_array

#import matplotlib.pyplot as plt

deltathresh = 1E-3

def thresholdcallbak(model, where):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst > deltathresh:
            model.terminate()

def formulateDisjunctionMIP(model,K,support,nodetimelimit):
	

	### getting necessary parameters ###
	n = model.numVars
	m = model.numConstrs
	
	intvars = set()
	constrs = model.getConstrs()
	
	objcoeff = 1
	if model.ModelSense == GRB.MAXIMIZE:
		objcoeff = -1

	#first we count the total number of constraints (double-count eqs)
	totalcons = 0
	cons_std_map = [[] for i in range(m)]
	varbd_std_map = [[-1,-1] for i in range(n)]
	
	for j in range(m):
		sense = constrs[j].getAttr("Sense")
		if sense == '>' or sense == '<':
			cons_std_map[j] = [totalcons]
			totalcons += 1
			
		else:
			cons_std_map[j] = [totalcons, totalcons+1]
			#print("aAAAAAA")
			totalcons += 2
	
	#print(cons_std_map)
	#exit(0)
	allvars = model.getVars()	
	
	for i in range(model.numVars):
		var = allvars[i]	
		if var.lb > -GRB.INFINITY:
			varbd_std_map[i][0] = totalcons
			totalcons += 1
		if var.ub < GRB.INFINITY:
			varbd_std_map[i][1] = totalcons
			totalcons += 1
			
		if var.vtype == GRB.INTEGER or var.vtype == GRB.INTEGER:
			intvars.add(var.index)
	
	### set-up of disj model ###
	M = GRB.INFINITY
	
	disj = Model()
	disj.setParam("OutputFlag",0)
	disj.setParam("TimeLimit",nodetimelimit)
	disj.setParam("DualReductions",0)
	
	p = [None for i in range(totalcons)]
	q = [None for i in range(totalcons)]
	pi = [None for i in range(n)]
	for i in range(totalcons):
		p[i] = disj.addVar(lb=0.0, name="p%d" %i)
		q[i] = disj.addVar(lb=0.0, name="q%d" %i)
	
	sL = disj.addVar(lb=0.0, name="sL")
	sR = disj.addVar(lb=0.0, name="sR")
	
	for i in range(n):
		pi[i] = disj.addVar(vtype=GRB.INTEGER, lb=-M, ub=M, name="pi%d"%i)
		
	pi0 = disj.addVar(vtype=GRB.INTEGER, lb=-M, ub=M, name="pirhs")
	
	delta = disj.addVar(lb=0.0, name="delta")


	### support constraint and opt sense

	for i in range(n):
		if i not in intvars or i not in support:
			disj.addConstr(pi[i] == 0)

	disj.setObjective(delta, GRB.MAXIMIZE)

	### p and q constraints ###
	
	# m.addConstr( - np.transpose(A) @ p - c @ sL - pi == np.zeros(nvars), name="pc" )
	# m.addConstr( - np.transpose(A) @ q - c @ sR + pi == np.zeros(nvars), name="qc" )
	for var in allvars:
		new_cons_disj_p = LinExpr()
		new_cons_disj_q = LinExpr()
		
		
		for cons in constrs:
			coeff = model.getCoeff(cons, var)
			sense = cons.getAttr("Sense")
			j = cons.index
			
			#print(coeff, p[cons_std_map[j][0]])
			if sense == '>':
				new_cons_disj_p += coeff*p[cons_std_map[j][0]]
				new_cons_disj_q += coeff*q[cons_std_map[j][0]]
			elif sense == '<':
				new_cons_disj_p += -coeff*p[cons_std_map[j][0]]
				new_cons_disj_q += -coeff*q[cons_std_map[j][0]]
			else:
				new_cons_disj_p += (coeff*p[cons_std_map[j][0]] - coeff*p[cons_std_map[j][1]])
				new_cons_disj_q += (coeff*q[cons_std_map[j][0]] - coeff*q[cons_std_map[j][1]])

		if varbd_std_map[var.index][0] != -1:
			new_cons_disj_p += (p[varbd_std_map[var.index][0]])
			new_cons_disj_q += (q[varbd_std_map[var.index][0]])
		
		if varbd_std_map[var.index][1] != -1:
			new_cons_disj_p += (-p[varbd_std_map[var.index][1]])
			new_cons_disj_q += (-q[varbd_std_map[var.index][1]])
		
		new_cons_disj_p += (-sL*var.Obj*objcoeff - pi[var.index])
		new_cons_disj_q += (-sR*var.Obj*objcoeff + pi[var.index])
		
		disj.addConstr(new_cons_disj_p == 0, name=(var.varname+"p"))
		disj.addConstr(new_cons_disj_q == 0, name=(var.varname+"q"))
		
	### pb and qb constraints ###
	# m.addConstr( - np.transpose(b) @ p - sL * K - pi0 >= delta, name="dd" )
	# m.addConstr( - np.transpose(b) @ q - sR * K + pi0 >= -1 + delta, name="dd2" )
	
	new_cons_disj_p = LinExpr()
	new_cons_disj_q = LinExpr()
		
	for cons in constrs:
		coeff = cons.rhs
		sense = cons.getAttr("Sense")
		j = cons.index

		if sense == '>':
			new_cons_disj_p += coeff*p[cons_std_map[j][0]]
			new_cons_disj_q += coeff*q[cons_std_map[j][0]]
		elif sense == '<':
			new_cons_disj_p += -coeff*p[cons_std_map[j][0]]
			new_cons_disj_q += -coeff*q[cons_std_map[j][0]]
		else:
			new_cons_disj_p += (coeff*p[cons_std_map[j][0]] - coeff*p[cons_std_map[j][1]])
			new_cons_disj_q += (coeff*q[cons_std_map[j][0]] - coeff*q[cons_std_map[j][1]])
	
	for var in allvars:			
		if varbd_std_map[var.index][0] != -1:
			new_cons_disj_p += (var.lb*p[varbd_std_map[var.index][0]])
			new_cons_disj_q += (var.lb*q[varbd_std_map[var.index][0]])
		
		if varbd_std_map[var.index][1] != -1:
			new_cons_disj_p += (-var.ub*p[varbd_std_map[var.index][1]])
			new_cons_disj_q += (-var.ub*q[varbd_std_map[var.index][1]])
			
	new_cons_disj_p += (-sL*K*objcoeff - pi0 - delta)
	new_cons_disj_q += (-sR*K*objcoeff + pi0 - delta + 1)
		
	disj.addConstr(new_cons_disj_p >= 0, name="pb")
	disj.addConstr(new_cons_disj_q >= 0, name="pq")
	
	#############################
	
	disj.update()

	#disj.write('disj.lp')
	
	disj.optimize(thresholdcallbak)
	
	if disj.status == 5:
		#print("\nUnbounded problem, probably the proposed bound was too weak\n")
		return 1, None, None
		
	if disj.objVal <= 1E-6:
		#print("\nNo disjuction found\n")
		return 0, None, None
	
	#print("rounded output", np.round(pi.X), np.round(pi0.X))
	solX = [pi[i].X for i in range(len(pi))]
	print("INFO: Disjunction", np.round(solX), np.round(pi0.X))
	return 1, np.round(solX), np.round(pi0.X)


def findDisjunction(args, nodetimelimit):
	
	#model_orig = read(args[0])
	model_orig = args[0]
	#support = set(range(model_orig.numvars))
	#model_orig.write("test.lp")
	#print("Building A,b...")
	#c,A,b,intvars = buildAb(model_orig)
	#print("done.")
	
	
	K = float(args[1])
	if math.isinf(K):
		print("Infeasible node considered compressed already")
		return True
	
	support = set()
	if len(args) <= 2:
		support = set(range(model_orig.numvars))
	else:
		for i in range(2, len(args)):
			var = model_orig.getVarByName(args[i])
			if not var == None:
				support.add(var.index)
	
	#print("Formulating disjunction MIP...")

	success, pi,pi0 = formulateDisjunctionMIP(model_orig,K,support,nodetimelimit)
	#print("done.")
	if success:
		print("Node with dual bound", K, "can be compressed")
	else:
		print("Node with dual bound", K, "could not be compressed")
		
	sanitycheck = False
	if sanitycheck:
		relaxed = model_orig.relax()
		disj1 = relaxed.copy()
		disj2 = relaxed.copy()
		
		relaxed.setParam("OutputFlag",0)
		disj1.setParam("OutputFlag",0)
		disj2.setParam("OutputFlag",0)
		
		varlist1 = disj1.getVars()
		varlist2 = disj2.getVars()
		
		disj1.addConstr(np.dot(pi,varlist1) <= pi0)
		disj2.addConstr(np.dot(pi,varlist2) >= pi0 + 1)
		
		relaxed.optimize()
		disj1.optimize()
		disj2.optimize()
		
		if disj1.status == 3 :
			obj1 = "Inf"
		else:
			obj1 = disj1.objVal
		
		if disj2.status == 3 :
			obj2 = "Inf"
		else:
			obj2 = disj2.objVal
		
		print("\n==========\n")
		#print("Statuses", relaxed.status, disj1.status, disj2.status)
		print("Rel/disj1/disj2", relaxed.objVal, obj1, obj2)
		print("\n==========\n")
		
		for i in range(model_orig.numVars):
			var = model_orig.getVars()[i]
			varindex = var.index
			if abs(pi[varindex])> 1E-5:
				print(var, pi[varindex])
		print("rhs", pi0)
		
	return success

#findDisjunction(sys.argv)

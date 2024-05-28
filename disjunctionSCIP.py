from pyscipopt import *
import pandas as pd
import random
import sys
import os
import numpy as np
import math

#from scipy.sparse import csr_matrix
#from scipy.sparse import csr_array

#import matplotlib.pyplot as plt

deltathresh = 1E-2

def formulateDisjunctionMIP(model,K,support,nodetimelimit, disjcoefbound, disjsuppsize, seed, node_id):
	
	### getting necessary parameters ###
	n = model.getNVars()
	m = model.getNConss()
	
	intvars = set()
	constrs = model.getConss()
	
	objcoeff = 1
	if model.getObjectiveSense() == 'maximize':
		objcoeff = -1

	#first we count the total number of constraints (double-count eqs)
	totalcons = 0
	cons_std_map = [[] for i in range(m)]
	varbd_std_map = [[-1,-1] for i in range(n)]
	
	for j in range(m):
		lhs = model.getLhs(constrs[j])
		rhs = model.getRhs(constrs[j])
		if lhs <= -model.infinity() or rhs >= model.infinity(): #In case of one-sided inequalities
			cons_std_map[j] = [totalcons]
			totalcons += 1
		else:
			cons_std_map[j] = [totalcons, totalcons+1]
			totalcons += 2
	
	allvars = model.getVars()	
	
	for i in range(n):
		var = allvars[i]	
		if var.getLbGlobal() > -model.infinity():
			varbd_std_map[i][0] = totalcons
			totalcons += 1
		if var.getUbGlobal() < model.infinity():
			varbd_std_map[i][1] = totalcons
			totalcons += 1
		
		if var.vtype() == 'INTEGER' or var.vtype() == 'BINARY':
			intvars.add(i)
	
	### set-up of disj model ###
	disj = Model()

	M = 0
	if disjcoefbound == None:
		M = disj.infinity()
	else:
		M = disjcoefbound

	disj.setRealParam("limits/time",nodetimelimit)
	disj.setIntParam("lp/threads",1)
	disj.setIntParam("randomization/permutationseed",seed)
	disj.setIntParam("display/verblevel",0)
	
	p = [None for i in range(totalcons)]
	q = [None for i in range(totalcons)]
	pi = [None for i in range(n)]
	for i in range(totalcons):
		p[i] = disj.addVar(lb=0.0, name="p%d" %i)
		q[i] = disj.addVar(lb=0.0, name="q%d" %i)
	
	sL = disj.addVar(lb=0.0, name="sL")
	sR = disj.addVar(lb=0.0, name="sR")
	
	for i in range(n):
		lhscoefbound = M
		if disjsuppsize == 1: #when support size is one, lhs coefficient should be <= 1
			lhscoefbound = 1
		pi[i] = disj.addVar(vtype='I', lb=-lhscoefbound, ub=lhscoefbound, name=("pi"+allvars[i].name))
		
	pi0 = disj.addVar(vtype='I', lb=-M, ub=M, name="pirhs")
	
	delta = disj.addVar(lb=0.0, name="delta")

	### support constraint and opt sense
 
	for i in range(n):
		if i not in intvars or i not in support:
			disj.addCons(pi[i] == 0)

	disj.setObjective(delta, 'maximize')

	### p and q constraints ###
	
	# m.addConstr( - np.transpose(A) @ p - c @ sL - pi == np.zeros(nvars), name="pc" )
	# m.addConstr( - np.transpose(A) @ q - c @ sR + pi == np.zeros(nvars), name="qc" )
 
	for i in range(n):
		var = allvars[i]
		new_cons_disj_p = Expr()
		new_cons_disj_q = Expr()
		
		for j in range(m):
			cons = constrs[j]
			linexp = model.getValsLinear(cons)

			coeff = 0
			if var.name in linexp.keys():
				coeff = linexp[var.name]

			lhs = model.getLhs(cons)
			rhs = model.getRhs(cons)
			
			if lhs > -model.infinity() and rhs < model.infinity(): #if inequality is two-sided
				new_cons_disj_p += (coeff*p[cons_std_map[j][0]] - coeff*p[cons_std_map[j][1]])
				new_cons_disj_q += (coeff*q[cons_std_map[j][0]] - coeff*q[cons_std_map[j][1]])
			elif rhs >= model.infinity(): #if inequality is >=
				new_cons_disj_p += coeff*p[cons_std_map[j][0]]
				new_cons_disj_q += coeff*q[cons_std_map[j][0]]
			else: #if inequality is <=
				new_cons_disj_p += -coeff*p[cons_std_map[j][0]]
				new_cons_disj_q += -coeff*q[cons_std_map[j][0]]
			
		if varbd_std_map[i][0] != -1:
			new_cons_disj_p += (p[varbd_std_map[i][0]])
			new_cons_disj_q += (q[varbd_std_map[i][0]])
		
		if varbd_std_map[i][1] != -1:
			new_cons_disj_p += (-p[varbd_std_map[i][1]])
			new_cons_disj_q += (-q[varbd_std_map[i][1]])
		
		new_cons_disj_p += (-sL*var.getObj()*objcoeff - pi[i])
		new_cons_disj_q += (-sR*var.getObj()*objcoeff + pi[i])
		
		disj.addCons(new_cons_disj_p == 0, name=(var.name+"p"))
		disj.addCons(new_cons_disj_q == 0, name=(var.name+"q"))
		
	### pb and qb constraints ###
	# m.addConstr( - np.transpose(b) @ p - sL * K - pi0 >= delta, name="dd" )
	# m.addConstr( - np.transpose(b) @ q - sR * K + pi0 >= -1 + delta, name="dd2" )
	
	new_cons_disj_p = Expr()
	new_cons_disj_q = Expr()
		
	for j in range(m):
		cons = constrs[j]

		lhs = model.getLhs(cons)
		rhs = model.getRhs(cons)

		if lhs > -model.infinity() and rhs < model.infinity(): #if inequality is two-sided
			new_cons_disj_p += (lhs*p[cons_std_map[j][0]] - rhs*p[cons_std_map[j][1]])
			new_cons_disj_q += (lhs*q[cons_std_map[j][0]] - rhs*q[cons_std_map[j][1]])
		elif rhs >= model.infinity(): #if inequality is >=
			new_cons_disj_p += lhs*p[cons_std_map[j][0]]
			new_cons_disj_q += lhs*q[cons_std_map[j][0]]
		else:
			new_cons_disj_p += -rhs*p[cons_std_map[j][0]]
			new_cons_disj_q += -rhs*q[cons_std_map[j][0]]
	
	for i in range(n):
		var = allvars[i]	
		if varbd_std_map[i][0] != -1:
			new_cons_disj_p += (var.getLbGlobal()*p[varbd_std_map[i][0]])
			new_cons_disj_q += (var.getLbGlobal()*q[varbd_std_map[i][0]])
		
		if varbd_std_map[i][1] != -1:
			new_cons_disj_p += (-var.getUbGlobal()*p[varbd_std_map[i][1]])
			new_cons_disj_q += (-var.getUbGlobal()*q[varbd_std_map[i][1]])
			
	new_cons_disj_p += (-sL*K*objcoeff - pi0 - delta)
	new_cons_disj_q += (-sR*K*objcoeff + pi0 - delta + 1)
		
	disj.addCons(new_cons_disj_p >= 0, name="pb")
	disj.addCons(new_cons_disj_q >= 0, name="pq")
	
	#############################
	
	if disjsuppsize != None:
		#print("\n\nI should add constraints for a support of size ", disjsuppsize)
		bigM = 1E5
		pi_nz = [None for i in range(n)]
		
		for i in range(n):
			pi_nz[i] = disj.addVar(vtype='B', name="pi_nz%d"%i)
			disj.addCons(pi[i] <= bigM*pi_nz[i], name="nz_ub%d"%i)
			disj.addCons(pi[i] >= -bigM*pi_nz[i], name="nz_lb%d"%i)

		disj.addCons(np.sum(pi_nz) <= disjsuppsize)
		#pi0_nz = disj.addVar(vtype='B', name="pirhs_nz")
		
	###################

	# The following emulates an early stopping callback
	disj.setObjlimit(deltathresh)
	disj.setIntParam("limits/bestsol",1)
	
	disj.optimize()

	if disj.getStatus() == "unbounded":
		#print("\nUnbounded problem, probably the proposed bound was too weak\n")
		return 1, None, None, disj.getSolvingTime()
		
	if disj.getObjVal() <= 1E-6:
		#print("\nNo disjuction found\n")
		return 0, None, None, disj.getSolvingTime()

	#print("rounded output", np.round(pi.X), np.round(pi0.X))
	solX = [disj.getVal(pi[i]) for i in range(len(pi))]
	print("INFO: Node", node_id, "Disjunction", np.round(solX), np.round(disj.getVal(pi0)))

	return 1, np.round(solX), np.round(disj.getVal(pi0)), disj.getSolvingTime()


def findDisjunction(args, nodetimelimit, disjcoefbound, disjsuppsize, seed, nameToIdDictionary):
	
	model_orig = args[0]
	
	K = float(args[1])

	obj1 = None
	obj2 = None
	
	if math.isinf(K):
		print("Infeasible node considered compressed already")
		return True
	
	support = set(range(model_orig.getNVars()))

	success, pi,pi0, runtime = formulateDisjunctionMIP(model_orig,K,support,nodetimelimit, disjcoefbound, disjsuppsize, seed, args[2])
	#print("done.")
	if success:
		print("Node",args[2],"with dual bound", K, "can be compressed")
	else:
		print("Node",args[2],"with dual bound", K, "could not be compressed")
		
	sanitycheck = True
	if success and sanitycheck:
		relaxed = Model(sourceModel=model_orig)
		n = relaxed.getNVars()

		vvvaars = model_orig.getVars()

		for var in relaxed.getVars():
			vtype = var.vtype()
			if vtype == 'INTEGER':
				relaxed.chgVarType(var, 'CONTINUOUS')
			elif vtype == 'BINARY':
				relaxed.chgVarType(var, 'CONTINUOUS')
				#The two bounds below are not explictly in SCIP for a binary variable
				#Careful here! since the variables may have modified bounds due to branching
				relaxed.chgVarLb(var, max(0,var.getLbGlobal()))

				relaxed.chgVarUb(var, min(1,var.getUbGlobal()))
			
		disj1 = Model(sourceModel=relaxed)
		disj2 = Model(sourceModel=relaxed)
		
		relaxed.setIntParam("display/verblevel",0)
		disj1.setIntParam("display/verblevel",0)
		disj2.setIntParam("display/verblevel",0)

		varlist1 = disj1.getVars()
		varlist2 = disj2.getVars()

		## Careful! SCIP may change the variable orders when duplicating a model, so we keep a dictionary
		## and create the disjunctions "by hand"
		ineq1 = Expr()
		ineq2 = Expr()

		for i in range(n):
			ineq1 += varlist1[i]*pi[nameToIdDictionary[varlist1[i].name]]
			ineq2 += varlist2[i]*pi[nameToIdDictionary[varlist2[i].name]]
	
		disj1.addCons(ineq1 <= pi0)
		disj2.addCons(ineq2 >= pi0 + 1)

		relaxed.optimize()
		disj1.optimize()
		disj2.optimize()
		
		if disj1.getStatus() == 'infeasible' :
			if disj1.getObjectiveSense() == 'maximize':
				obj1 = -disj1.infinity()
			else:
				obj1 = disj1.infinity()
		else:
			obj1 = disj1.getObjVal()
		
		if disj2.getStatus() == 'infeasible' :
			if disj2.getObjectiveSense() == 'maximize':
				obj2 = -disj2.infinity()
			else:
				obj2 = disj2.infinity()
		else:
			obj2 = disj2.getObjVal()

		if relaxed.getObjectiveSense() == 'maximize' and max(obj1,obj2) > K + 1E-4 :
			print("Warning (Max): Sanity check failed, not counting as success. Rel/disj1/disj2/K", relaxed.getObjVal(), obj1, obj2,K)
			success  = 0
		elif relaxed.getObjectiveSense() == 'minimize' and min(obj1,obj2) < K - 1E-4:
			print("Warning (Min): Sanity check failed, not counting as success. Rel/disj1/disj2/K", relaxed.getObjVal(), obj1, obj2,K)
			success  = 0

	return success, runtime, pi, pi0, obj1, obj2

#findDisjunction(sys.argv)

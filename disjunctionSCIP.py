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

deltathresh = 1E-3

def thresholdcallbak(model, where):
    if where == GRB.Callback.MIPSOL:
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)

        if objbst > deltathresh:
            model.terminate()

def formulateDisjunctionMIP(model,K,support,nodetimelimit, disjcoefbound, disjsuppsize, seed):
	
	### getting necessary parameters ###
	n = model.getNVars()
	m = model.getNConss()
	
	intvars = set()
	constrs = model.getConss()
	
	objcoeff = 1
	print(model.getObjectiveSense())
	if model.getObjectiveSense() == 'maximize':
		objcoeff = -1

	#first we count the total number of constraints (double-count eqs)
	totalcons = 0
	cons_std_map = [[] for i in range(m)]
	varbd_std_map = [[-1,-1] for i in range(n)]
	
	for j in range(m):
		#sense = constrs[j].getAttr("Sense")
		lhs = model.getLhs(constrs[j])
		rhs = model.getRhs(constrs[j])
		if lhs <= -model.infinity() or rhs >= model.infinity(): #In case of one-sided inequalities
			cons_std_map[j] = [totalcons]
			totalcons += 1
		else:
			cons_std_map[j] = [totalcons, totalcons+1]
			#print("aAAAAAA")
			totalcons += 2
	
	#print(cons_std_map)
	#exit(0)
	allvars = model.getVars()	
	
	for i in range(model.getNVars()):
		var = allvars[i]	
		if var.getLbGlobal() > -model.infinity():
			varbd_std_map[i][0] = totalcons
			totalcons += 1
		if var.getUbGlobal() < model.infinity():
			varbd_std_map[i][1] = totalcons
			totalcons += 1
			
		if var.vtype == 'I' or var.vtype == 'B':
			intvars.add(var.index)
	
	### set-up of disj model ###
	
	disj = Model()

	M = 0
	if disjcoefbound == None:
		M = disj.infinity()
	else:
		M = disjcoefbound

	#disj.hideOutput(True)
 
	disj.setRealParam("limits/time",nodetimelimit)
	disj.setIntParam("lp/threads",1)
	disj.setIntParam("randomization/permutationseed",seed)
	
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
		pi[i] = disj.addVar(vtype='I', lb=-lhscoefbound, ub=lhscoefbound, name="pi%d"%i)
		
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
			#print("\n",linexp,"\n")
			#print(allvars)
			#coeff = model.getCoeff(cons, var)
   
			coeff = 0
			if var.name in linexp.keys():
				coeff = linexp[var.name]

			lhs = model.getLhs(cons)
			rhs = model.getRhs(cons)

			#sense = cons.getAttr("Sense")
			#j = cons.index
			
			#print(coeff, p[cons_std_map[j][0]])
			if lhs > -model.infinity() and rhs < model.infinity(): #if inequality is two-sided
				new_cons_disj_p += (coeff*p[cons_std_map[j][0]] - coeff*p[cons_std_map[j][1]])
				new_cons_disj_q += (coeff*q[cons_std_map[j][0]] - coeff*q[cons_std_map[j][1]])
			elif rhs >= model.infinity(): #if inequality is >=
				new_cons_disj_p += coeff*p[cons_std_map[j][0]]
				new_cons_disj_q += coeff*q[cons_std_map[j][0]]
			else: #if inequality is >=
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
		#coeff = cons.rhs
		#sense = cons.getAttr("Sense")

		#j = cons.index

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

	#disj.update()

	#disj.write('disj.lp')
	
	#disj.optimize(thresholdcallbak)
	disj.optimize()

	if disj.getStatus() == "unbounded":
		#print("\nUnbounded problem, probably the proposed bound was too weak\n")
		return 1, None, None, disj.getSolvingTime()
		
	if disj.getObjVal() <= 1E-6:
		#print("\nNo disjuction found\n")
		return 0, None, None, disj.getSolvingTime()
	
	#print("rounded output", np.round(pi.X), np.round(pi0.X))
	solX = [pi[i].X for i in range(len(pi))]
	print("INFO: Disjunction", np.round(solX), np.round(pi0.X))
	return 1, np.round(solX), np.round(pi0.X), disj.getSolvingTime()


def findDisjunction(args, nodetimelimit, disjcoefbound, disjsuppsize, seed):
	
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
	
	#old variant that allowed supports to be restricted
	#support = set()
	#if len(args) <= 2:
	support = set(range(model_orig.getNVars()))
	#else:
	#	for i in range(2, len(args)):
	#		var = model_orig.getVarByName(args[i])
	#		if not var == None:
	#			support.add(var.index)
	
	#print("Formulating disjunction MIP...")

	success, pi,pi0, runtime = formulateDisjunctionMIP(model_orig,K,support,nodetimelimit, disjcoefbound, disjsuppsize, seed)
	#print("done.")
	if success:
		print("Node with dual bound", K, "can be compressed")
	else:
		print("Node with dual bound", K, "could not be compressed")
		
	sanitycheck = True
	if success and sanitycheck:
		relaxed = model_orig.relax()
		disj1 = relaxed.copy()
		disj2 = relaxed.copy()
		
		relaxed.hideOutput(True)
		disj1.hideOutput(True)
		disj2.hideOutput(True)
		
		varlist1 = disj1.getVars()
		varlist2 = disj2.getVars()
		
		disj1.addCons(np.dot(pi,varlist1) <= pi0)
		disj2.addCons(np.dot(pi,varlist2) >= pi0 + 1)
		
		relaxed.optimize()
		disj1.optimize()
		disj2.optimize()
		
		if disj1.status == 3 :
			if disj1.ModelSense == 'maximize':
				obj1 = -disj1.infinity()
			else:
				obj1 = disj1.infinity()
		else:
			obj1 = disj1.objVal
		
		if disj2.status == 3 :
			if disj2.ModelSense == 'maximize':
				obj2 = -disj2.infinity()
			else:
				obj2 = disj2.infinity()
		else:
			obj2 = disj2.objVal
		
		# print("\n==========\n")
		# #print("Statuses", relaxed.status, disj1.status, disj2.status)
		# print("Rel/disj1/disj2", relaxed.objVal, obj1, obj2)
		# print("\n==========\n")
		
		# for i in range(model_orig.numVars):
		# 	var = model_orig.getVars()[i]
		# 	varindex = var.index
		# 	if abs(pi[varindex])> 1E-5:
		# 		print(var, pi[varindex])
		# print("rhs", pi0)

		if relaxed.ModelSense == 'maximize' and max(obj1,obj2) > K + 1E-4 :
			print("Warning (Max): Sanity check failed, not counting as success. Rel/disj1/disj2/K", relaxed.objVal, obj1, obj2,K)
			success  = 0
		elif relaxed.ModelSense == 'minimize' and min(obj1,obj2) < K - 1E-4:
			print("Warning (Min): Sanity check failed, not counting as success. Rel/disj1/disj2/K", relaxed.objVal, obj1, obj2,K)
			success  = 0

	return success, runtime, pi, pi0

#findDisjunction(sys.argv)

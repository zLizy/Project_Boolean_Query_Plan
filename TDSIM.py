from sympy import *
from sympy.parsing.sympy_parser import parse_expr

#  Input: partial plan e{'p1-':1-s,'p1+':s}, Boolean expression Bxp,
#  		  assignament Asg, flag branch, cost budget b
#  Output: best plan
def TDSIM(e, Bxp, Asg, branch,Memo,count):

	if Asg in [Memo.keys()]:
		return Memo[Asg]

	bestcost = 0
	bestplan = []
	# visited.append()
	for p in getPredicates(Bxp):
		# global count
		count = count + 1
		e_parent = BuildPlans(p,e,branch) # e=['x1']
		# p = True
		e_plus, c_plus = TDSIM(e_parent, Bxp.subs({p:True}),Asg+(p.name+'+',),True,Memo,count)
		# p = False
		e_minus, c_minus = TDSIM(e_parent, Bxp.subs({p:False}),Asg+(p.name+'-',),False,Memo,count)
		cost = cost_dict[p.name] \
			+ selectivity_dict[p.name]*c_plus \
			+ (1-selectivity_dict[p.name])*c_minus
		if bestplan == [] or bestcost < -(cost):
			bestplan = [e_parent, e_plus, e_minus]
			bestcost = -cost

	Memo[Asg] = bestplan
	# print(Asg) 

	return Memo[Asg], -bestcost

def BuildPlans(p,e,branch):
	# Xe = visited
	# Xp = [item for item in [p.name] if item not in Xe] # [p] - Xe
	if e == 'scan':
		return p.name
	elif branch == True:
		return p.name
		# e=p.name+'+'
	else:
		return p.name
		# e[p.name]['-'] = {}
		# e.append(p.name+'-')
	# return e


def getPredicates(Bxp):
	predicates = Bxp.atoms()
	# print(predicates)
	if predicates=={True} or predicates=={False}:
		return []
	return predicates


if __name__ == '__main__':

	e='scans'
	Asg=tuple(['scan'])
	branch=True
	global Memo
	Memo={}
	count=0

	cost_dict = {'x1':4,'x2':5,'x3':10, 'x4':15}
	selectivity_dict = {'x1':0.7,'x2':0.6,'x3':0.5,'x4':0.4}

	# Bxp= ((x1 & x2)|(x1 & x3)|(x1 & x4))
	# Bxp= ((x1 & x2)|(x3 & x4))
	Bxp= '((x1 & (x2 | x3))|(x4 & x3)|(x1 & x4))'
	Bxp = parse_expr(Bxp)

	plan,cost = TDSIM(e,Bxp,Asg,branch)
	print(plan,cost,count)
	# ((x1 & x2)|(x1 & x3)|(x1 & x4))
	# with/without memo: 145
	# ((x1 & x2)|(x3 & x4))
	# with/without memo: 88


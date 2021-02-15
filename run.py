import random
import timeit
import os.path
from util import *
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
import argparse
import matplotlib.pyplot as plt



def run(args):

	
	time = 0
	countModel = 0
	countPareto = 0
	# assignmentList = []
	tempList = []
	tempQuery = []

	# get model repository
	df = getRandomRepository(args.m,args.n,args.flag)
	# get Pareto summary
	df_pareto = getParetoSummary(df,args.m,args.n,args.flag)

	tasks = df.columns.tolist()
	models = df.index.tolist()

	# synthesize queries
	queryList = getQueries(args.nquery,args.n)

	for query in queryList:
		# print(query)
		flag = False

		T,steps = getSteps(str(query))
		# print(T)

		for t in T:
			if not t in tasks:
				print('Process fails due to lack of model solving task ' + t)
				flag = True
				break
		if flag: continue
		column=T #+['cost']
		# print(df.head())

		df_selected = df.loc[:,column]
		# print(df_selected.head())
		df_selected.dropna(how='all',inplace=True)
		df_selected = df_selected.fillna(0)
		# print(df_selected.head())
		row_selected = df_selected.index


		M = df_selected.index.tolist()
		
		cost = df.loc[row_selected,'cost']
		Accuracy = df_selected[T].to_numpy()
		# compute runtime
		start = timeit.default_timer()
		optimizer = Optimizer(steps,M,T,Accuracy,cost,args.bound)
		# task:model
		assignment = optimizer.optimize()
		end = timeit.default_timer()
		time += end-start

		if assignment != {}:
			countModel += len(assignment)
			count, temp = paretoSummary(df_pareto, assignment)
			countPareto += count
			if temp != '':
				tempList.append(temp)
				tempQuery.append(str(query))
		
		# assignmentList.append(assignment)
	
	
	writeSummary(args,countModel,countPareto)
	writeQuery(tempList,tempQuery,args.flag)
	return time



	
	

if __name__ == '__main__':
	
	# python run.py -m 100 -n 40 -flag 0 -nquery 100 -constraint cost -bound 200
	# python run.py -m 100 -n 40 -nquery 50 -constraint cost -bound 200
	# python run.py -m 200 -n 40 -nquery 50 -constraint cost -bound 200
	# python run.py -m 200 -n 40 -nquery 100 -constraint cost -bound 200
	# python run.py -m 200 -n 40 -nquery 150 -constraint cost -bound 200
	# python run.py -m 200 -n 40 -nquery 200 -constraint cost -bound 200
	# python run.py -m 200 -n 80 -nquery 50 -constraint cost -bound 200
	# python run.py -m 200 -n 20 -nquery 200 -constraint cost -bound 200
	'''
	Configurations
	'''
	parser = argparse.ArgumentParser(description = 'Description')
	parser.add_argument('-m', help='Number of models, enter it',default=100, type=int)
	parser.add_argument('-n', help='Number of tasks, enter it', default=40, type=int)
	parser.add_argument('-nquery', help='Number of queries, enter it', default=100, type=int)
	parser.add_argument('-flag', help='high accuracy model is costly', default=1, type=int)
	parser.add_argument('-constraint', help='Type of constraint, enter accuracy/cost', default='cost', type=str)
	parser.add_argument('-bound', help='constraint bound, enter it',default=40, type=float)
	args = parser.parse_args()

	
	time = run(args)
	filepath = 'simulation/run_summary.csv'
	if os.path.isfile(filepath):
		df = pd.read_csv(filepath,index_col=0)
	else: df = pd.DataFrame(columns=['M','N','#query','time'])
	df = df.append({'M':args.m,'N':args.n,'#query':args.nquery,'time':time},ignore_index=True)
	df = df.sort_values(by=['M','N','#query'])
	df.index = range(len(df))
	df.to_csv('simulation/run_summary.csv')

import random
import timeit
import os.path
from util import *
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
import argparse
import matplotlib.pyplot as plt
from baselines.baseline_c_cost import getBaseline1
from baselines.baseline_c_accuracy_reverse import getBaseline2



def run(args):

	countModel = 0
	countPareto = 0
	tempList = []
	tempQuery = []

	# get model repository
	df = getRandomRepository(args.mdist)
	tasks = ['T'+str(idx) for idx in range(len(df.columns)-1)]
	df.columns = tasks+['cost']
	print('len of df is: '+str(len(df.columns)))

	models = ['M'+str(idx) for idx in range(len(df.index))]
	df.index = models

	# get Pareto summary
	df_pareto = getParetoSummary(df,args.mdist)
	
	# synthesize queries
	queryList = getQueries(len(tasks),args.qdist)
	idx = 1
	time = 0
	data_process_time = 0

	# get Pareto model only
	if args.approach == 'baseline':
		df = getParetoModelOnly(df,df_pareto,args.mdist)


	for query in queryList:

		if idx %1 != 0:
			print('query '+ str(idx))
		else:
			print('###################')
			print('query '+ str(idx))
			T,steps = getSteps(str(query))

			# data process
			start = timeit.default_timer()
			df_selected = df[T]
			df_selected = df_selected.dropna(how='all').fillna(0)
			row_selected = df_selected.index
			M = df_selected.index.tolist()
			cost = df.loc[row_selected,'cost']
			Accuracy = df_selected.to_numpy()
			end = timeit.default_timer()
			data_process_time = end-start


			# compute runtime
			# ## optimizer
			if args.approach == 'optimizer':
				start = timeit.default_timer()
				optimizer = Optimizer(steps,M,T,Accuracy,cost,args.constraint,args.bound)
				# task:model
				assignment,_A,_C = optimizer.optimize()
				end = timeit.default_timer()
				time = end-start
				writeIntermediateParetoSummary(args,df_pareto,query,T,assignment,_A,_C,time,data_process_time)
			else:
				Cost = np.array([[cost[i] if Accuracy[i,j] !=0 else 500 for j in range(len(T))] for i in range(len(M))])
				if args.constraint == 'cost':
					## baseline1
					start = timeit.default_timer()
					# task:model
					flag,_A,_C,assignment = getBaseline1(steps,M,T,Cost,Accuracy,args.bound,selected_model={})
					end = timeit.default_timer()
					time = end-start
					writeIntermediateParetoSummary(args,df_pareto,query,T,assignment,_A,_C,time,data_process_time,approach='baseline')
				else:
					## baseline2
					start = timeit.default_timer()
					from baselines.baseline_c_accuracy import getBaseline2
					flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,args.bound,start,selected_model={})
					
					# if _A ==0 and not flag:
					# 	print('next')
					# 	start_ = timeit.default_timer()
					# 	from baselines.baseline_c_accuracy_reverse import getBaseline2
					# 	flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,args.bound,start_,selected_model={})
					# 	if _A ==0 and not flag:
					# 		print('next')
					# 		start_ = timeit.default_timer()
					# 		from baselines.baseline_c_accuracy import getBaseline2
					# 		flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,args.bound,start_,selected_model={})
					# 		if _A ==0 and not flag:
					# 			print('next')
					# 			from baselines.baseline_c_accuracy_reverse import getBaseline2
					# 			flag,_A,_C,assignment = getBaseline2(steps,M,T,Cost,Accuracy,args.bound,start_,selected_model={},second=True)
					
					end = timeit.default_timer()
					time = end-start
					print(flag,_A,_C)
					writeIntermediateParetoSummary(args,df_pareto,query,T,assignment,_A,_C,time,data_process_time,approach='baseline')
		idx+=1	
		# assignmentList.append(assignment)

	
	# writeSummary(args,countModel,countPareto)
	# writeQuery(tempList,tempQuery,args.mdist,args.qdist)
	return time


if __name__ == '__main__':
	
	# python run.py -mdist uniform -qdist uniform -constraint cost -bound 200 -approach baseline
	# python run.py -mdist uniform -qdist power_law -constraint cost -bound 200 -approach baseline
	# python run.py -mdist power_law -qdist uniform -constraint cost -bound 200
	# python run.py -mdist power_law -qdist power_law -constraint cost -bound 200

	# python run.py -mdist uniform -qdist uniform -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist uniform -qdist power_law -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist power_law -qdist uniform -constraint accuracy -bound 0.95 -approach baseline
	# python run.py -mdist power_law -qdist power_law -constraint accuracy -bound 0.95 -approach baseline
	
	# python run.py -m 200 -n 20 -nquery 200 -constraint cost -bound 200
	'''
	Configurations
	'''
	parser = argparse.ArgumentParser(description = 'Description')
	parser.add_argument('-m', help='Number of models, enter it',default=100, type=int)
	parser.add_argument('-n', help='Number of tasks, enter it', default=40, type=int)
	parser.add_argument('-nquery', help='Number of queries, enter it', default=100, type=int)
	parser.add_argument('-qdist', help='query distribution', default='uniform', type=str)
	parser.add_argument('-mdist', help='model repository distribution', default='uniform', type=str)
	parser.add_argument('-flag', help='high accuracy model is costly', default=1, type=int)
	parser.add_argument('-constraint', help='Type of constraint, enter accuracy/cost', default='cost', type=str)
	parser.add_argument('-approach', help='Type of approach, enter optimizer/baseline', default='optimizer', type=str)
	parser.add_argument('-bound', help='constraint bound, enter it',default=40, type=float)
	args = parser.parse_args()

	
	time = run(args)
	# filepath = 'repository/run_summary.csv'
	# if os.path.isfile(filepath):
	# 	df = pd.read_csv(filepath,index_col=0)
	# else: df = pd.DataFrame(columns=['mdist','qdist','constraint','bound','time'])
	# df = df.append({'mdist':args.mdist,'qdist':args.qdist,'constraint':args.constraint,'bound':args.bound,'time':time},ignore_index=True)
	# df = df.sort_values(by=['mdist','qdist','constraint'])
	# df.index = range(len(df))
	# df.to_csv('repository/run_summary.csv')

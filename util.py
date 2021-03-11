import os.path
import random
import pandas as pd
from simulation.synthesize_random import generateRandomRepository
from sympy.parsing.sympy_parser import parse_expr
from optimization.parser import *
from simulation.querygenerator import Expression
from simulation.checkPareto import checkPareto
from optimization.optimizer import *


def getDF(path):
	df = pd.read_csv(path,index_col=0)
	return df

def getParetoSummary(df,mflag):
	# method = mapFlag(flag)

	# path = 'repository/model_pareto_'+str(M)+'_'+str(N)+'_'+method+'.csv'
	path = 'repository/model_pareto_'+mflag+'.csv'
	if os.path.isfile(path):
		df_pareto = getDF(path)
	else:
		df_pareto = checkPareto(df,mflag)
		df_pareto.to_csv(path)
	return df_pareto


def getRandomRepository(mdist):
	'''
	Get synthesized model repository.
	The result is stored in model_repository.csv
	--mflag: uniform / power_law
	'''

	model_task_dict = {}
	# filepath = 'repository/model_repository_'+str(M)+'_'+str(N)+'_'+method+'.csv'
	filepath = 'repository/model_repository_'+mdist+'.csv'
	if os.path.isfile(filepath):
		print ("Model repository exist")
		df = getDF(filepath)
	else:
		print ("File not exist")
		print('synthesizing model repository')
		df = generateRandomRepository(mdist)
	
	return df


def getQueries(N,qdist='uniform'):
	'''
	Query generator
	parameter: 
	N - number of tasks
	qdist - distribution:
			uniform
			gaussian
			power_law
			conjunction
			disjunction
	'''

	filepath = 'repository/query_'+qdist+'.csv'
	if os.path.isfile(filepath):
		print ("Query file exist")
		queryList = list(getDF(filepath)['query'])
	else:
		print ("Synthesizing queries")
		queryList = createQueries(N,qdist)
	return queryList


def createQueries(N,qdist='uniform'):

	num_exp = 5
	num_iter = 10
	if qdist == 'conjunction':
		conj_pro = 1
	elif qdist == 'disjunction':
		conj_pro = 0
	else:
		conj_pro = 0.5

	queryList = []
	
	df = pd.DataFrame(columns=['query','#predicate','#&','#|','#()','ratio_&','ratio_|'])
	for i in range(1,num_exp+1):
		# factor for number of predicate in a query
		# maximum #predicate is 2^6 = 64
		for j in range(num_iter):
			num_pred = pow(2,i)
			query = str(Expression(N,num_pred,conj_pro,qdist))
			# print(query)
			queryList.append(query)
			num_and = len(query.split('&'))-1
			num_or = len(query.split('|'))-1
			num_par = len(query.split('('))-1
			item = {'query':query,'#predicate':pow(2,i),\
				'#&':num_and,'#|':num_or,'#()':num_par, \
				'ratio_&':round(num_and/(num_pred-1),4), \
				'ratio_|':round(num_or/(num_pred-1),4)}
			df = df.append(item,ignore_index=True)
	df.to_csv('repository/query_'+qdist+'.csv')
	return queryList


def paretoSummary(df,assignment):

	countPareto = 0
	countNP = 0
	pareto = ''
	non_pareto = ''
	model = ''
	for t,m in assignment.items():
		# if m not in df.index: 
		# 	countPareto += 1
		# 	continue
		if df.loc[m,t] == 1:
			countPareto += 1
			pareto += t+'.'+m
			pareto += '|'
		else: 
			countNP += 1
			non_pareto += t+'.'+m
			non_pareto += '|'
		model += t+'.'+m
		model += '|'
	pareto = pareto[:-1]
	non_pareto = non_pareto[:-1]
	model = model[:-1]
	return countPareto,countNP,pareto,non_pareto,model
	

def mapFlag(flag):
	if flag: 
		method = 'max'
	else: 
		method = 'median'
	return method

def createdf(columns):
	df = pd.DataFrame(columns=columns)
	return df

def getSteps(query):
	'''
	Parse query into steps:
	car & red | bus & yellow
	-->
	car&red,s0
	bus&yellow,s1
	s0|s1,s2
	'''
	result = parse(query)
	steps,i = getSequence(result,0,'')
	Bxp = parse_expr(query)
	predicates = [str(p) for p in list(Bxp.atoms())]

	return predicates,steps

def writeResults(args,query,T,all_model,pareto,non_pareto,count_p,count_np,_A,_C,time,data_process_time,approach):
	
	filepath = 'repository/run_summary_'+args.constraint+'_'+args.mdist+'_'+args.qdist+'_'+approach+'.csv'
	# filepath = 'repository/test_.csv'
	if os.path.isfile(filepath):
		df_summary = pd.read_csv(filepath,index_col=0)
	else: 
		df_summary = pd.DataFrame(columns=['query','#predicate','mdist','qdist','constraint','bound','approach','selected_model',
			'pareto_model','#pareto','non_pareto_model','#non_pareto','pareto_ratio','accuracy','cost','optimization_time','data_process_time'])

	# if not all_model:
	# 	df_summary = df_summary.append({'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
	# 	'constraint':args.constraint,'bound':args.bound,'approach':approach},ignore_index=True)
	# else:
	df_summary = df_summary.append({'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
		'constraint':args.constraint,'bound':args.bound,'approach':approach,'selected_model':all_model,'pareto_model':pareto,
		'#pareto':count_p,'non_pareto_model':non_pareto,'#non_pareto':count_np,'pareto_ratio':round(count_p/len(T),4),
		'accuracy':_A,'cost':_C,'optimization_time':round(time,4),'data_process_time':round(data_process_time,4)},ignore_index=True)
	df_summary = df_summary.sort_values(by=['#predicate','query','qdist','mdist','constraint','approach'])
	df_summary.index = range(len(df_summary))
	df_summary.to_csv(filepath)

def writeIntermediateParetoSummary(args,df_pareto,query,T,assignment,_A,_C,time,data_process_time,approach='optimizer'):
	if assignment != {}:  #{t:m}
		# countModel += len(assignment)
		count_p, count_np, pareto, non_pareto, all_model = paretoSummary(df_pareto, assignment)
		# countPareto += count_p
	# if all_model == '':
	# 	tempList.append(non_pareto)
	# 	tempQuery.append(str(non_pareto))
	else:
		all_model=pareto=non_pareto=''
		count_p=count_np=0

	writeResults(args,query,T,all_model,pareto,non_pareto,count_p, count_np,_A,_C,time,data_process_time,approach)
	# return countModel

def writeSummary(args,countModel,countPareto):

	path = 'repository/model_pareto_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['#predicates','#pareto','ratio','mdist','qdist'])
	
	if countModel:
		df_summary = df_summary.append(\
		{'#predicates':int(countModel),'#pareto':int(countPareto),\
		'ratio':round(countPareto/countModel,4),'mdist':args.mdist,'qdist':args.qdist},\
		ignore_index=True)
	else:
		df_summary = df_summary.append(\
		{'#predicates':int(countModel),'#pareto':int(countPareto),\
		'ratio':0,'mdist':args.mdist,'qdist':args.qdist},\
		ignore_index=True)
	df_summary = df_summary.sort_values(by=['mdist','qdist','#predicates'])
	df_summary.to_csv(path)
	return

def writeQuery(tempList,tempQuery,mdist,qdist):
	# print(tempList)
	# print(tempQuery)
	path = 'repository/non_pareto_query_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['query','predicate','mdist','qdist'])
	df_temp = createdf(['query','predicate','flag'])
	df_temp['query'] = tempQuery
	df_temp['predicate'] = tempList
	df_temp['mdist'] = [mdist]*len(tempQuery)
	df_temp['qdist'] = [qdist]*len(tempQuery)
	df_summary = df_summary.append(df_temp)
	df_summary.index = range(len(df_summary))
	# df_summary = df_summary.sort_values(by=['M','N','#query'])
	df_summary.to_csv(path)
	return
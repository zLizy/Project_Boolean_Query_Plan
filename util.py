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

def getParetoSummary(df,M,N,flag):
	method = mapFlag(flag)

	path = 'repository/model_pareto_'+str(M)+'_'+str(N)+'_'+method+'.csv'
	if os.path.isfile(path):
		df_pareto = getDF(path)
	else:
		df_pareto = checkPareto(df,M,N)
	return df_pareto


def getRandomRepository(M,N,flag):
	'''
	Get synthesized model repository.
	The result is stored in model_repository.csv
	'''
	method = mapFlag(flag)

	model_task_dict = {}
	filepath = 'repository/model_repository_'+str(M)+'_'+str(N)+'_'+method+'.csv'
	if os.path.isfile(filepath):
	    print ("File exist")
	    df = getDF(filepath)
	else:
	    print ("File not exist")
	    print('synthesizing model repository')
	    df = generateRandomRepository(M,N,8,method)
	
	return df


def getQueries(num_exp,N):
	'''
	Query generator
	parameter: 
	num_exp - number of expressions
	N - number of tasks
	'''
	
	queryList = []
	for i in range(num_exp):
		queryList.append(Expression(N-1,pow(2,random.randint(1,5))))
	return queryList



def paretoSummary(df,assignment):

	countPareto = 0
	temp = ''
	for t,m in assignment.items():
		# if m not in df.index: 
		# 	countPareto += 1
		# 	continue
		if df.loc[m,t] == 1:
			countPareto += 1
		else: 
			temp += t+'.'+m
			temp += '|'
	temp = temp[:-1]
	return countPareto,temp
	

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
	# print(result)
	steps,i = getSequence(result,0,'')
	# print(steps)
	Bxp = parse_expr(query)
	predicates = [str(p) for p in list(Bxp.atoms())]
	# print(predicates)
	return predicates,steps


def writeSummary(args,countModel,countPareto):
	# print(countModel,countPareto)
	method = mapFlag(args.flag)

	path = 'repository/model_pareto_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['M','N','#query','#predicates','#pareto','ratio','flag'])
	df_summary = df_summary.append(\
		{'M':int(args.m),'N':int(args.n),'#query':int(args.nquery),\
		'#predicates':int(countModel),'#pareto':int(countPareto),\
		'ratio':round(countPareto/countModel,4),'flag':method},\
		ignore_index=True)
	df_summary = df_summary.sort_values(by=['M','N','#query'])
	df_summary.to_csv(path)
	return

def writeQuery(tempList,tempQuery,method):
	# print(tempList)
	# print(tempQuery)
	path = 'repository/non_pareto_query_summary.csv'
	if os.path.isfile(path):
		df_summary = getDF(path)
	else:
		df_summary = createdf(['query','predicate','flag'])
	df_temp = createdf(['query','predicate','flag'])
	df_temp['query'] = tempQuery
	df_temp['predicate'] = tempList
	df_temp['flag'] = [method]*len(tempQuery)
	df_summary = df_summary.append(df_temp)
	df_summary.index = range(len(df_summary))
	# df_summary = df_summary.sort_values(by=['M','N','#query'])
	df_summary.to_csv(path)
	return
import os.path
import random
import pandas as pd
import numpy as np
from simulation.synthesize_sampling import generateRandomRepository
from sympy.parsing.sympy_parser import parse_expr
from sympy import *
from optimization.parser import *
from simulation.querygenerator import Expression
from simulation.checkPareto import checkPareto
from optimization.optimizer import *


def is_blank(x):
	return not pd.isnull(x)

def getDF(path):
	df = pd.read_csv(path,index_col=0)
	return df

def getParetoSummary(df,mdist='uniform',a=5,synthetic=True):
	# method = mapFlag(flag)

	if synthetic:
		if mdist == 'power_law':
			path = 'repository/model_pareto_'+mdist+'_a='+str(a)+'.csv'
		else:
			path = 'repository/model_pareto_'+mdist+'.csv'
	if not synthetic:
		path = 'repository/yolo_pareto.csv'
	# if os.path.isfile(path):
	# 	df_pareto = getDF(path)
	df_pareto = checkPareto(df,path)
	df_pareto.to_csv(path)
	return df_pareto

def getSelectivity(T,synthetic):
	if synthetic:
		path = 'repository/selectivity.csv'
		if os.path.isfile(path):
			df = getDF(path)
		else:
			df = pd.DataFrame(columns=['class','selectivity'])
			df['class'] = T
			df['selectivity'] = np.random.rand(len(T))
			df.to_csv(path)
	else:
		df = getDF('repository/coco_selectivity.csv')
	return dict(zip(df['class'], df['selectivity']))


def getRandomRepository(mdist,a):
	'''
	Get synthesized model repository.
	The result is stored in model_repository.csv
	--mflag: uniform / power_law
	'''
		# write index to file
	# filepath = 'repository/index.csv'
	# if not os.path.isfile(filepath):
	# 	df_index = pd.DataFrame()
	# 	df_index['model']=df.index.tolist()
	# 	df_index['new_name'] = models
	# 	df_index.to_csv(filepath)

	if mdist == 'power_law':
		filepath = 'repository/model_repository_'+mdist+'_a='+str(a)+'.csv'
	else:
		filepath = 'repository/model_repository_'+mdist+'.csv'
	if os.path.isfile(filepath):
		print ("Model repository exist")
		df = getDF(filepath)
	else:
		print ("File not exist")
		print('synthesizing model repository')
		df = generateRandomRepository(mdist,a)
	
	return df

def getYOLORepository():
	return getDF('repository/model_stats_f1.csv')

def getParetoModelOnly(df,df_pareto,mdist,a,synthetic=True):
	if mdist == 'power_law':
		path = 'repository/pareto_only_model_'+mdist+'_a='+str(a)+'.csv'
	else:
		path = 'repository/pareto_only_model_'+mdist+'.csv'
	if not synthetic:
		path = 'repository/pareto_only_yolo.csv'
	
	# if os.path.isfile(path):
	# 	df_new = getDF(path)
	# 	return df_new
	indexer = df_pareto.applymap(is_blank)
	df_new = pd.DataFrame(columns=df.columns,index=df.index)
	df_new[indexer] = df
	df_new['cost'] = df['cost']
	print('cost' in df_new.columns)
	df_new.to_csv(path)
	return df_new

def getQueries(N=1,qdist='uniform',synthetic=1):
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
	if synthetic:
		# filepath = 'repository/query_'+qdist+'.csv'
		filepath = 'simulation/DNF_query_'+qdist+'_synthetic.csv'
		if os.path.isfile(filepath):
			print ("Query file exist")
			queryList = list(getDF(filepath)['query'])
		else:
			print ("Synthesizing queries")
			queryList = createQueries(N,qdist)
	else:

		queryList = []
		filepath='simulation/query_gt.csv'
		if os.path.isfile(filepath):
			print ("Query file exist")
			df = getDF(filepath)
			queryList = list(df[df['form']=='dnf']['query'])
		# uni_idx = [11,35,37]
		# pow_idx = [2,5,13,15,27,22,26]
		# _idx = [uni_idx,pow_idx]
		# qdist = ['uniform','power_law']
		# for i,dist in enumerate(qdist):
		# 	filepath = 'simulation/query_'+qdist[i]+'_gt.csv'
		# 	if os.path.isfile(filepath):
		# 		print ("Query file exist")
		# 		queryList += list(getDF(filepath)['query'][_idx[i]])

		# dnf_uni_idx = [19,25]
		# dnf_pow_idx = [11,20]
		# _idx = [dnf_uni_idx,dnf_pow_idx]
		# qdist = ['uniform','power_law']
		# for i,dist in enumerate(qdist):
		# 	filepath = 'simulation/query_dnf_'+qdist[i]+'_gt.csv'
		# 	if os.path.isfile(filepath):
		# 		print ("Query file exist")
		# 		queryList += list(getDF(filepath)['query'][_idx[i]])
		print(queryList)
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
		if df.loc[m,t.replace('_',' ')] == 1:
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
	# parse query into lists of operations
	result = parse(query)
	steps,i = getSequence(result,0,'')
	# parse query into expression
	Bxp = parse_expr(query)
	predicates = [str(p) for p in list(Bxp.atoms())]

	return predicates,steps

def writeResults(args,query_idx,query,T,all_model,pareto,non_pareto,count_p,count_np,_A,_C,_time,data_process_time,approach):
	
	# if args.mdist == 'power_law':
	# 	filepath = 'output/summary_'+args.constraint+'_'+args.mdist+'_'+ \
	# 		args.qdist+'_'+approach+'_a='+str(args.a)+'.csv'
	if args.order:
		if args.synthetic:
			filepath = 'output/synthetic/summary_order_'+args.constraint+'_'+args.mdist+'_'+args.qdist+'_'+approach+'.csv'
		else:
			filepath = 'output/coco/summary_order_'+args.constraint+'_'+approach+'.csv'
			
	else:
		if args.synthetic:
			filepath = 'output/synthetic/summary_'+args.constraint+'_'+args.mdist+'_'+args.qdist+'_'+approach+'.csv'
		else:
			filepath = 'output/coco/summary_'+args.constraint+'_'+approach+'.csv'

	# filepath = 'repository/test.csv'
	if os.path.isfile(filepath):
		df_summary = pd.read_csv(filepath,index_col=0)
	else: 
		df_summary = pd.DataFrame(columns=['query_index','qgetSelectivityuery','#predicate','mdist','qdist','constraint','bound','approach','selected_model',
			'pareto_model','#pareto','non_pareto_model','#non_pareto','pareto_ratio','accuracy','cost','optimization_time','data_process_time'])

	# if not all_model:
	# 	df_summary = df_summary.append({'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
	# 	'constraint':args.constraint,'bound':args.bound,'approach':approach},ignore_index=True)
	# else:
	df_summary = df_summary.append({'query_index':query_idx,'query':query,'#predicate':len(T),'mdist':args.mdist,'qdist':args.qdist,
		'constraint':args.constraint,'bound':args.bound,'approach':approach,'selected_model':all_model,'pareto_model':pareto,
		'#pareto':count_p,'non_pareto_model':non_pareto,'#non_pareto':count_np,'pareto_ratio':round(count_p/len(T),4),
		'accuracy':_A,'cost':_C,'optimization_time':round(_time,4),'data_process_time':round(data_process_time,4)},ignore_index=True)
	df_summary = df_summary.sort_values(by=['#predicate','query','qdist','mdist','constraint','approach'])
	df_summary.index = range(len(df_summary))
	df_summary.to_csv(filepath)

def writeIntermediateParetoSummary(args,query_idx,df_pareto,query,T,assignment,_A,_C,time,data_process_time,approach='optimizer'):
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

	if args.record:
		writeResults(args,query_idx,query,T,all_model,pareto,non_pareto,count_p, count_np,_A,_C,time,data_process_time,approach)
	else:
		s = '_A,_C,time,data_process_time,approach'.replace(',','\t')
		print(s)
		print(_A,_C,time,data_process_time,approach)
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

def getPredicates(Bxp):
	predicates = Bxp.atoms()
	# print(predicates)
	if predicates=={True} or predicates=={False}:
		return []
	return predicates

def buildPath(name,assignment,Bxp,order,df_config,atom,parent_answered,path,flag=True):

	# print(Bxp)
	plan = []
	
	# if Bxp == True or Bxp == False:
	# 	return []
	predicates = list(getPredicates(Bxp))

	if predicates != []:
	# if order != []:
		# p = list(getPredicates(Bxp))[0]
		p = [symbols(o) for o in order if symbols(o) in predicates][-1]
		
		if atom == []:
			atom.append((p,))
		
		e_parent = p.name # e=['x1']
		model = assignment[e_parent]

		# print('model',model,'parent_answered',parent_answered)
		
		if model not in parent_answered:
			i = int(re.search( r'\d+', model).group())
			# extendScript(df_config[df_config['index']==i],atom,path)
			extendScript(name,i,atom,list(assignment.keys()),path)

		# print(str(p)+'=True',Bxp.subs({p:True}))
		# print('order',order)
		
		# p = True
		e_plus = buildPath(name,assignment,Bxp.subs({p:True}),order,df_config,atom+[(p,'>')],parent_answered+[model],path,flag=True)

		# p = False
		# print(str(p)+'=False',Bxp.subs({p:False}))
		# print('order',order)
		e_minus = buildPath(name,assignment,Bxp.subs({p:False}),order,df_config,atom+[(p,'=')],parent_answered+[model],path,flag=False)
		

		plan = [e_parent, e_plus, e_minus]


	return plan

def extendScript(name,model_index,atom,T,path):
	# DROP TABLE IF EXISTS coco.result;
	# SELECT * FROM coco.`images`
	# ORDER BY `images`.`id`  ASC
	# LIMIT 10
	# TO RUN hebafer/yolov3-sqlflow:latest
	# CMD "yolov3_detect_variant.py",
	#     "--dataset=coco",
	#     "--model=yolov3",
	#     "--latency=0.05",
	#     "--lag=100",
 	#    "--tasks=1,2,3,4,5"
	# INTO result;
	# SELECT * FROM coco.images 
	# ORDER BY images.image_id ASC 
	# LIMIT 10 
	# TO RUN hebafer/sqlflow-experiments:latest 
	# CMD "yolov3_detect_variant.py",
	# "--experiment_index=100" INTO result;

	s = "{sqlflow}\n{select}{where}\nTO RUN lizyliz/sqlflow-experiments:latest\n"
	s += "CMD \"detect.py\",\n\"--experiment_index={index}\"\nINTO {table};\n"
	s += "{merge}"

	# if flag == True: no filter condition
	if len(atom) == 1:
		sqlflow = 'DROP TABLE IF EXISTS coco_val.'+name+';\nDROP TABLE IF EXISTS coco_val.intermediate;\n' 
		where = ''
		select = 'SELECT * FROM coco_val.images'
		table = name
		merge = ''
	else:
		# print('atom',atom[1:])
		sqlflow = ''
		select = 'SELECT * FROM coco_val.'+name+'\n'
		where = 'WHERE '
		# print(atom)
		for (a,flag) in atom[1:]:
			where += f'{a}{flag}0 AND '
		where = where[:-5]
		table = 'intermediate'
		merge = 'UPDATE coco_val.'+name+' t, coco_val.intermediate s\nSET '
		for t in T:
			merge+= 't.'+ t + '=s.' + t + ','
		merge = merge[:-1]
		merge += '\nWHERE t.image_id = s.image_id;\n'
		# merge = 'REPLACE INTO coco_val.'+name+' SELECT * FROM coco_val.intermediate WHERE '+name+'.image_id=intermediate.image_id;\n'
		# merge = 'REPLACE INTO coco_val.'+name+' SELECT * FROM coco_val.intermediate;\n'

	# model = row['model'].values[0]
	# latency = row['latency'].values[0]
	# lag = row['lag'].values[0]
	# tasks =','.join([item for item in row['tasks'].values[0].replace('[','').replace(']','').replace('\n','').split(' ') if item != ''])
	# print(s.format(where=where,latency=latency,lag=lag))
	with open(path,'a') as f:
		# f.write(s.format(sqlflow=sqlflow,select=select,where=where,model=model,latency=latency,lag=lag,tasks=tasks))
		f.write(s.format(sqlflow=sqlflow,select=select,where=where,index=str(model_index),table=table,merge=merge))
		# f.write('\n')

	# return s.format(sqlflow=sqlflow,where=where,latency=latency,lag=lag)

def normalScript(name,assignment,path):
	s = "{sqlflow}\n{select}\nTO RUN lizyliz/sqlflow-experiments:latest\n"
	s += "CMD \"detect.py\",\n\"--experiment_index={index}\"\nINTO {table};\n"
	s += "{merge}"

	models = list(set(assignment.values()))
	T = list(assignment.keys())

	with open(path,'w') as f:
		for i,model in enumerate(models):
			model_index = int(re.search( r'\d+', model).group())
			if i == 0:
				sqlflow = 'DROP TABLE IF EXISTS coco_val.'+name+';\nDROP TABLE IF EXISTS coco_val.intermediate;\n' 
				select = 'SELECT * FROM coco_val.images'
				table = name
				merge = ''
			else:
				# print('atom',atom[1:])
				sqlflow = ''
				select = 'SELECT * FROM coco_val.'+name
				where = ''
				table = 'intermediate'
				merge = 'UPDATE coco_val.'+name+' t, coco_val.intermediate s\nSET '
				for t in T:
					merge+= 't.'+ t + '=s.' + t + ','
				merge = merge[:-1]
				merge += '\nWHERE t.image_id = s.image_id;\n'
			# f.write(s.format(sqlflow=sqlflow,select=select,where=where,model=model,latency=latency,lag=lag,tasks=tasks))
			f.write(s.format(sqlflow=sqlflow,select=select,index=str(model_index),table=table,merge=merge))

def writeScript(assignment,query,idx,pre_order,Etype,constraint,bound,order=False):

	if not assignment:
		return 0
	Bxp = parse_expr(query)
	# print(Bxp)
	config_path = 'repository/model_config_task_all.csv'
	df_config = getDF(config_path)

	if constraint == 'cost':
			bound = int(bound)
	else:
		bound = int(bound*100)

	if not os.path.exists('script/'):
		os.makedirs('script')

	if order:
		name = Etype+'_'+str(idx)+'_order_'+constraint+'_'+str(bound)
		path = 'script/script_'+name+'.sql'
	else:
		name = Etype+'_'+str(idx)+'_overlap_'+constraint+'_'+str(bound)
		path = 'script/script_'+name+'.sql'

	# print('======= Tree =====')
	if order:
		plan = buildPath(name,assignment,Bxp,pre_order,df_config,[],[],path,flag=True)
		print(plan)
	else:
		normalScript(name,assignment,path)
	
	




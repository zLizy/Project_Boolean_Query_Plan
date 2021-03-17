import pandas as pd

def checkPareto(df,mflag):

	map_dict = {}
	columns = list(df.columns)[:-1]
	for c in columns:
		data = df.loc[:,[c,'cost']]
		data = data.dropna()
		map_dict[c] = TwoDimensionsPD(data,c)
	print('len:', len(map_dict))

	cost = df['cost']

	df_pareto_b = pd.DataFrame.from_dict(map_dict, orient='index')
	df_pareto_b = df_pareto_b.T

	# add missing models
	rest = [i for i in df.index if i not in df_pareto_b.index]
	# print(rest)
	df_rest = pd.DataFrame(columns=columns,index=rest)
	df_pareto_b = df_pareto_b.append(df_rest)

	df_pareto_b.to_csv('repository/model_pareto_'+mflag+'.csv')
	# df_pareto.fillna(0)
	return df_pareto_b

def TwoDimensionsPD(data,taskName):
	map_dict = {}

	sorted_data = data.sort_values(by='cost')
	index = list(sorted_data.index)

	map_dict[index[0]] = 1

	cutt_off = sorted_data[taskName][0]
	for i in range(1, len(sorted_data)):
		if sorted_data[taskName][i] > cutt_off:
			cutt_off = sorted_data[taskName][i]
			map_dict[index[i]] = 1
	return map_dict


if __name__ == '__main__':

	filepath = '../model_repository.csv'
	df = pd.read_csv(filepath,index_col=0)
	df_pareto = checkPareto(df)
	print(df_pareto.head())
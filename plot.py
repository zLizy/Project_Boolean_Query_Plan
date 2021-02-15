import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
import pandas as pd

def plotSummary():
	'''
	columns: T1,T2,...,cost
	'''
	filepath = 'model_repository.csv'
	df = pd.read_csv(filepath,index_col=0)
	print(df.head())

	plt.figure()
	andrews_curves(df.loc[:,:-1], "cost");
	# Accuracy = df.loc[]to_numpy()
	# fig, ax = plt.subplots()
	# ax.scatter(x, y, s, c, marker=verts)

if __name__ == '__main__':
	plotSummary()
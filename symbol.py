import pandas
import numpy
from sympy import *


if __name__ == '__main__':
	e={}
	x1,x2,x3,x4 = symbols('x1,x2,x3,x4')
	E = ((x1 & x2)|(x1 & x3)|(x1 & x4)).subs({x1:True})
	# >>> E
	# x2 | x3 | x4
	E.atoms()
	# {x2, x3, x4}
	Bxp="(True&x2)||(True&x3)||(True&x4)"
	exp = '((x1 & x2)|(x1 & x3)|(x1 & x4))'
	print(eval(string.replace('&','and').replace('||','or').replace('!','not')))
	# TDACB(e,)
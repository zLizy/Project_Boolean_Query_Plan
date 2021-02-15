import numpy as np
import random
import math



class Expression(object):
    OPS = ['&', '|']

    GROUP_PROB = 0.3
    NEGATION_PROB = 0.1

    MIN_NUM, MAX_NUM = 0, 40

    def __init__(self, maxIndex, maxNumbers, _maxdepth=None, _depth=0):
        """
        maxNumbers has to be a power of 2
        """
        if _maxdepth is None:
            _maxdepth = math.log(maxNumbers, 2) - 1

        if _depth < _maxdepth and random.randint(0, _maxdepth) > _depth:
            self.left = Expression(maxIndex,maxNumbers, _maxdepth, _depth + 1)
        else:
            self.left = 'T'+str(random.randint(Expression.MIN_NUM, maxIndex))
            # if random.random() < Expression.NEGATION_PROB:
            # 	self.left = '!{0}'.format(self.left)

        if _depth < _maxdepth and random.randint(0, _maxdepth) > _depth:
            self.right = Expression(maxIndex,maxNumbers, _maxdepth, _depth + 1)
        else:
            self.right = 'T'+str(random.randint(Expression.MIN_NUM, maxIndex))
            # if random.random() < Expression.NEGATION_PROB:
            # 	self.right = '!{0}'.format(self.right)

        self.grouped = random.random() < Expression.GROUP_PROB
        self.operator = random.choice(Expression.OPS)

    def __str__(self):
        s = '{0!s} {1} {2!s}'.format(self.left, self.operator, self.right)
        if self.grouped:
            return '({0})'.format(s)
        else:
            return s



if __name__ == '__main__':

	M = 100
	N = 40
	num_ops = 20
	num_exp = 10

	for i in range(num_exp):
		print(Expression(N,pow(2,random.randint(1,5))).__str__())
		print(str(Expression(N,pow(2,random.randint(1,5)))))






query = "car & red | bus & yellow"
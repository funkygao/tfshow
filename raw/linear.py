#!/usr/bin/env python
# -*- coding: utf-8 -*-

from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

f = lambda x: x # 激活函数

class LinearUnit(Perceptron):
	def __init__(self, input_num):
		Perceptron.__init__(self, input_num, f)

def generate_samples():
	'''
	生成5个人的样本数据：工龄，与月工资
	'''
	return [[5], [3], [8], [1.4], [10.1]], [5500, 2300, 7600, 1800, 11400]

if __name__ == '__main__':
	l = LinearUnit(1)
	train_x, train_y = generate_samples()
	l.train(train_x, train_y, 10, .01)
	print l
	print 'Work 3.4 years, monthly salary = %.2f' % l.predict([3.4])
	print 'Work 15 years, monthly salary = %.2f' % l.predict([15])
	print 'Work 1.5 years, monthly salary = %.2f' % l.predict([1.5])
	print 'Work 6.3 years, monthly salary = %.2f' % l.predict([6.3])

	# plot
	x = np.arange(0, 13)
	y = [l.predict([x_]) for x_ in x]	
	plt.plot(x, y, label='linear')
	x = map(lambda x_:x_[0], train_x)	
	plt.plot(x, train_y, 'ro', label='sample %d'%len(x))
	plt.legend()
	plt.show()
    
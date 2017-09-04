#!/usr/bin/env python
# -*- coding: utf-8 -*-

from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

f = lambda x: x

class LinearUnit(Perceptron):
	def __init__(self, input_num):
		Perceptron.__init__(self, input_num, f)

def generate_samples():
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

	x = np.arange(0, 13)
	y = []
	for i in range(len(x)):
		y.append(l.predict([x[i]]))
	plt.plot(x, y, label='linear')
	x, y = [], []
	for m in train_x:
		x.append(m[0])
	for m in train_y:
		y.append(m)
	plt.plot(x, y, 'ro', label='sample %d'%len(x))
	plt.legend()
	plt.show()
    
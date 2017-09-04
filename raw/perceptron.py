#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

class Perceptron(object):
	def __init__(self, input_num, activator):
		'''
		初始化感知器，设置输入参数个数，以及激活函数
		'''
		self.activator = activator
		self.W = np.zeros(input_num)
		self.B = .0

	def __str__(self):
		return 'weights:%s\tbias:%f' % (self.W, self.B)

	def predict(self, input_vec):
		return self.activator(
			reduce(lambda a, b: a+b, 
				map(lambda(x, w): x*w, 
					zip(input_vec, self.W))) 
			+ self.B)

	def train(self, input_vecs, labels, epochs, rate):
		for i in range(epochs):
			self._one_step(input_vecs, labels, rate)

	def _one_step(self, input_vecs, labels, rate):
		'''
		一次迭代，把所有的训练数据过一遍
		'''
		samples = zip(input_vecs, labels)
		for (input_vec, label) in samples:
			output = self.predict(input_vec)
			# update weights
			self._update_weights(input_vec, output, label, rate)

	def _update_weights(self, input_vec, output, label, rate):
		delta = label-output
		self.W = map(
			lambda(x, w): w+rate*delta*x, 
			zip(input_vec, self.W))
		self.B += rate*delta

def f(x):
	'''
	AND的激活函数
	'''
	return 1 if x > 0 else 0

def generate_samples():
	input_vecs = [[1,1], [0,0], [1,0], [0,1]]
	labels = [1, 0, 0, 0]
	return input_vecs, labels

def train_and_perceptron():
	'''
	使用AND表训练感知器
	'''
	p = Perceptron(2, f)
	input_vecs, labels = generate_samples()
	p.train(input_vecs, labels, 10, .1)
	return p

if __name__ == '__main__':
	and_perception = train_and_perceptron()
	print and_perception

	print '1&0=%d' % and_perception.predict([1, 0])
	print '0&0=%d' % and_perception.predict([0, 0])
	print '0&1=%d' % and_perception.predict([0, 1])
	print '1&1=%d' % and_perception.predict([1, 1])


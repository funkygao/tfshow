#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

class Network(object):
	"""Network is the neutral network"""
	def __init__(self, layer_units):
		'''layer_units=[2, 3, 1] means input layer 2, hidden layer 3, output layer 1'''
		self.layer_units = layer_units		
		self.b = [np.random.randn(x, 1)/2 for x in layer_units[1:]]
		self.W = [np.random.randn(x, y)/10 for x, y in zip(layer_units[:-1], layer_units[1:])]

	def __str__(self):
		return '{0}  <=W\n{1}  <=b'.format(self.W, self.b)		

	def shape(self):
		return 'W:{0}, b:{1}'.format([w.shape for w in self.W], [b.shape for b in self.b])

	def feedforward(self, a):
		'''Return the output of the network if 'a' is input.'''
		for b, w in zip(self.b, self.W):
			a = sigmoid(w.T.dot(a)+b)
		return a

	def train(self, train_data, epochs, mini_batch_size, learning_rate, test_data=None):
		'''Train the neural network using mini-batch stochastic gradient descent.
		train_data is a list of tuples (x, y)'''
		n = len(train_data)
		for epoch in xrange(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, learning_rate)
		if test_data:
			print 'Epoch {0}: {1}/{2}'.format(epoch, self.evaluate(test_data), len(test_data))
		else:
			print 'Epoch {0} complete'.format(epoch)

	def update_mini_batch(self, train_data, learning_rate):
		nabla_b = [np.zeros(b.shape) for b in self.b]
		nabla_w = [np.zeros(w.shape) for w in self.W]
		for x, y in train_data:
			delta_nabla_b, delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.W = [w - (learning_rate/len(train_data))*nw for w, nw in zip(self.W, nabla_w)]
		self.b = [b - (learning_rate/len(train_data))*nb for b, nb in zip(self.b, nabla_b)]


	def dump(self):
		print self
		print self.shape()

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

if __name__ == '__main__':
	n = Network([3, 5, 2])
	n.dump()
	

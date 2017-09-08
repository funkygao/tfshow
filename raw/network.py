#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import mnist_loader
import pickle
import matplotlib.pyplot as plt

def sigmoid(z):
	'''when z is a vector or ndarray, numpy automatically applies sigmoid elementwise'''
	return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z) * (1. - sigmoid(z))

class Network(object):
	def __init__(self, layer_units):
		self.layer_units = layer_units
		self.W = [np.random.randn(y, x)/3 for x, y in zip(layer_units[:-1], layer_units[1:])]
		# first layer in input layer, omits to set any biases for those neurons
		self.b = [np.random.randn(y, 1)/10 for y in layer_units[1:]]
		self.activator = sigmoid

	@property
	def shapes(self):
		return [w.shape for w in self.W], [b.shape for b in self.b]	

	def predict(self, x):
		for W, b in zip(self.W, self.b):
			x = self.activator(np.dot(W, x) + b)
		return np.argmax(x)

	def train(self, train_data, epochs, mini_batch_size, learning_rate, test_data=None):
		for epoch in range(epochs):
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, len(train_data), mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, learning_rate)
			print 'Epoch {0}: complete'.format(epoch)

	def update_mini_batch(self, mini_batch, learning_rate):
		nabla_b = [np.zeros(b.shape) for b in self.b]
		nabla_W = [np.zeros(W.shape) for W in self.W]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_W = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_W = [nw+dnw for nw, dnw in zip(nabla_W, delta_nabla_W)]
		self.W = [w-((learning_rate/len(mini_batch)))*nw for w, nw in zip(self.W, nabla_W)]
		self.b = [b-((learning_rate/len(mini_batch)))*nb for b, nb in zip(self.b, nabla_b)]

	def backprop(self, x, y):
		pass

	def save(self, fn):
		params = {'W': self.W, 'b': self.b}
		outfile = open(fn, 'wb')
		pickle.dump(params, outfile)
		outfile.close()

	def load(self, fn):
		infile = open(fn, 'rb')
		params = pickle.load(infile)
		infile.close()
		self.W = params['W']
		self.b = params['b']

	def evaluate(self, test_data):
		test_results = [(self.predict(x), y) for (x, y) in test_data]
		return sum([int(x==y) for (x, y) in test_results])

	def evaluate_and_print_errors(self, test_data):
		test_results = [(self.predict(x), y) for (x, y) in test_data]
		idx = 0
		for (x, y) in test_results:
			if x != y:
				self.draw_sample(test_data, idx)
			idx+=1

	def draw_sample(self, train_data, idx):
		dim = int(np.sqrt(len(train_data[0][0])))
		sample = train_data[idx][0].reshape(dim, dim)
		for y in range(dim):
			for x in range(dim):
				if sample[y, x] > .1:
					print 'X',
				else:
					print '.',
			print


#=================
# Configurations
#=================
layer_units = [784, 20, 10]
epochs, mini_batch_size, learning_rate = 10, 20, 1.0

if __name__ == '__main__':
	train_data, validation_data, test_data = mnist_loader.load_data_wrapper() # 50K, 10K, 10K
	
	net = Network(layer_units)	
	print 'layers: {0}, shapes: {1}'.format(layer_units, net.shapes)
	
	net.train(train_data, epochs, mini_batch_size, learning_rate, test_data)
	print net.predict(np.random.randn(784))

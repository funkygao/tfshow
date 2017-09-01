#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FC:
	def __init__(self, in_num, out_num, lr=0.01):
		self._in_num = in_num
		self._out_num = out_num
		self.w = np.random.randn(out_num, in_num)*10
		self.b = np.zeros(out_num)
		self.lr = lr

	def _sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def _tanh(self, z):
		return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

	def forward_feed(self, in_data):
		return self._sigmoid(np.dot(self.w, in_data) + self.b)

def draw3D(X, Y, Z, angle=(40, -45)):
    fig = plt.figure(figsize=(15,7))
    ax = Axes3D(fig)
    ax.view_init(angle[0], angle[1])
    ax.plot_surface(X,Y,Z,rstride=1, cstride=1, cmap='rainbow')

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
X_f = X.flatten()
Y_f = Y.flatten()
data = zip(X_f, Y_f)

fc = FC(2, 1)
Z1 = np.array([fc.forward_feed(d) for d in data])
Z1 = Z1.reshape((100, 100))

draw3D(X, Y, Z1,(40, -45))

plt.show()


draw3D(X, Y, Z1,(40, -45))
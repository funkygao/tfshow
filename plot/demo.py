#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import math

x = np.asarray(range(1, 102))
y1 = np.random.randn(101)
y2 = np.linspace(-1, 1, 101)
# sigmoid的导数 y=sigmoid(x)，则y`=y(1-y)
y3 = map(lambda y:y*(1-y), x)
sigmoid = map(lambda x:1/(1+math.e**(-x)), x)

plt.plot(x, y1, 'ro', label='randn')
plt.plot(x, y2, label='linspace')
#plt.plot(x, y3, label='simoid`')
#plt.plot(x, sigmoid, label='sigmoid')
plt.legend()
plt.show()
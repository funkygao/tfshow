#!/usr/bin/env python
#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.asarray(range(1, 102))
y1 = np.random.randn(101)
y2 = np.linspace(-1, 1, 101)

plt.plot(x, y1, 'r', label='randn')
plt.plot(x, y2, label='linspace')
plt.legend()
plt.show()
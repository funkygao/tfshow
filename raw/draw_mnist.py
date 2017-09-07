#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mnist_loader
import numpy as np

train_data, _, _ = mnist_loader.load_data_wrapper()
for n in range(1000):
    sample = train_data[n][0].reshape(28, 28)
    for y in range(28):
        for x in range(28):
            if sample[y][x] > .1:
                print 'X',
            else:
                print '.',
        print
    print np.argmax(train_data[n][1])

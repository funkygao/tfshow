#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import time
import os

#===============
# Configurations
#===============
enable_matplot = False

if enable_matplot:
    plt.ion()
    figure = plt.figure()
    ax1 = plt.subplot(211)
    ax1.set_title('mnist image')

train_data, _, _ = mnist_loader.load_data_wrapper()
dim = int(np.sqrt(len(train_data[0][0])))

for n in range(len(train_data)):    
    img = train_data[n][0].reshape(dim, dim)

    time.sleep(.1)

    # draw on matplot
    if enable_matplot:
        ax1.matshow(img, cmap=plt.get_cmap('gray'))
        figure.canvas.draw()
        continue

    os.system('clear')
    for y in range(dim):
        for x in range(dim):
            if img[y][x] > .1:
                print 'X',
            else:
                print '.',
        print
    print n+1, np.argmax(train_data[n][1])

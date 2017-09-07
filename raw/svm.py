#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mnist_loader
from sklearn import svm

train_data, validation_data, test_data = mnist_loader.load_data()

# train
classifier = svm.SVC()
classifier.fit(train_data[0], train_data[1])

# test
predictions = [int(a) for a in classifier.predict(test_data[0])]
num_correct = sum(int(a==y) for a, y in zip(predictions, test_data[1]))

print '{0} of {1} values correct.'.format(num_correct, len(test_data[1]))
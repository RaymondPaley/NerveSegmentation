# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 06:38:37 2016

@author: Abecedarian
"""

import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import leaky_rectify, softmax
import scipy.misc as misc
import numpy as np
import os
import csv

#Set up Test Data
test_folder = '/Users/Abecedarian/Desktop/Nerve_Segmentation/test'
dir_list_test = os.listdir(test_folder)
dir_list_test = [x for x in dir_list_test if 'tif' in x]

test_num = len(dir_list_test)



X_test = np.empty((1, 1, 420,580))

        
# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.ivector('y')

# create a small convolutional neural network

network = lasagne.layers.InputLayer((None, 1, 420, 580), input_var)
network = lasagne.layers.Conv2DLayer(network, 32, (5, 5),stride = 2,
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 32, (3, 3), 
                                     nonlinearity=leaky_rectify)

network = lasagne.layers.Pool2DLayer(network, (3, 3), stride = 2, mode='max')

network = lasagne.layers.Conv2DLayer(network, 64, (5, 5),stride = 2,
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 64, (3, 3), 
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 64, (3, 3), 
                                     nonlinearity=leaky_rectify)                                 

network = lasagne.layers.Pool2DLayer(network, (3, 3), stride = 2, mode='max')

network = lasagne.layers.Conv2DLayer(network, 128, (3, 3),
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 128, (3, 3), 
                                     nonlinearity=leaky_rectify)
network = lasagne.layers.Conv2DLayer(network, 128, (3, 3), 
                                     nonlinearity=leaky_rectify)                                 

network = lasagne.layers.Pool2DLayer(network, (3, 3), stride = 2, mode='max')

network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    128, nonlinearity=leaky_rectify,
                                    W=lasagne.init.Orthogonal())
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    10, nonlinearity=softmax)

## Loading the parameters
with np.load('/Users/Abecedarian/Desktop/Nerve_Segmentation/model_random_chunks.npz') as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(network, param_values)

## Make a submission 
test_prediction = lasagne.layers.get_output(network, deterministic=True)
predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1),allow_input_downcast = True)

with open('/Users/Abecedarian/Desktop/Nerve_Segmentation/Submission_0_1_ver4.csv','w') as csvfile:
    imagesub = csv.writer(csvfile, delimiter='\t')
    imagesub.writerow(['img','pixels'])
    for i in range(test_num):
        imageNum = dir_list_test[i].split('.')[0]
        X_test[0,0] = misc.imread(test_folder+os.sep+dir_list_test[i])        
        rle = ''
        if predict_fn(X_test[0:1])[0]:
            rle = '1 243600'    
        imagesub.writerow([imageNum,rle])
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:48:23 2016

@author: Abecedarian
"""

import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid
import scipy.misc as misc
import time
import numpy as np
import os
import sys
import csv
from re import sub

#Set up Test Data
test_folder = '/Images/test'
#test_folder = '/Users/tiruviluamala/Desktop/UltrasoundNerveSegmentation/test'
dir_list_test = os.listdir(test_folder)
dir_list_test = [x for x in dir_list_test if 'tif' in x]

test_num = len(dir_list_test)

base_filt = 8
down_sample = 1.0
x_dim = int(420*down_sample)
y_dim = int(580*down_sample)

X_test = np.empty((1, 1, x_dim,y_dim))

#Create mechanism for working through minibatches
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
# create Theano variables for input and target minibatch
input_var = T.tensor4('X')
target_var = T.tensor3('y')

# Implement the U-net architecture

batch_size = 2

Inp = lasagne.layers.InputLayer((None, 1, x_dim, y_dim), input_var)

Conv1a = lasagne.layers.Conv2DLayer(Inp, base_filt, (3,3), nonlinearity=leaky_rectify)
Conv1b = lasagne.layers.Conv2DLayer(Conv1a, base_filt, (3,3), nonlinearity=leaky_rectify, pad = 'full')

Pool1 = lasagne.layers.Pool2DLayer(Conv1b, (2, 2), mode='max')

Conv2a = lasagne.layers.Conv2DLayer(Pool1, 2*base_filt, (3,3),nonlinearity=leaky_rectify)
Conv2b = lasagne.layers.Conv2DLayer(Conv2a, 2*base_filt, (3,3),nonlinearity=leaky_rectify, pad = 'full')

Pool2 = lasagne.layers.Pool2DLayer(Conv2b, (2, 2), mode='max')

Conv3a = lasagne.layers.Conv2DLayer(Pool2, 4*base_filt, (3,3),nonlinearity=leaky_rectify)
Conv3b = lasagne.layers.Conv2DLayer(Conv3a, 4*base_filt, (3,3),nonlinearity=leaky_rectify, pad = 'full')

Pool3 = lasagne.layers.Pool2DLayer(Conv3b, (2, 2), mode='max')

Conv4a = lasagne.layers.Conv2DLayer(Pool3, 8*base_filt, (3,3),nonlinearity=leaky_rectify)
Conv4b = lasagne.layers.Conv2DLayer(Conv4a, 8*base_filt, (3,3),nonlinearity=leaky_rectify, pad = 'full')

Pool4 = lasagne.layers.Pool2DLayer(Conv4b, (2, 2), mode='max')

Conv5a = lasagne.layers.Conv2DLayer(Pool4, 16*base_filt, (3,3),nonlinearity=leaky_rectify, pad = 1)
Conv5b = lasagne.layers.Conv2DLayer(Conv5a, 8*base_filt, (3,3),nonlinearity=leaky_rectify, pad = 'full')

UpConv1 = lasagne.layers.TransposedConv2DLayer(Conv5b, 8*base_filt, filter_size = (2,2), stride = 2, nonlinearity=leaky_rectify)

#merge layer here merges bottom of U
merge4 = lasagne.layers.ConcatLayer([Conv4b,UpConv1], 1, cropping = [None, None, 'center', 'center'])
Conv6a = lasagne.layers.Conv2DLayer(merge4, 8*base_filt, (3,3), nonlinearity=leaky_rectify, pad = 1)
Conv6b = lasagne.layers.Conv2DLayer(Conv6a, 4*base_filt, (3,3), nonlinearity=leaky_rectify, pad = 'full')

UpConv2 = lasagne.layers.TransposedConv2DLayer(Conv6b, 4*base_filt, filter_size = (2,2), stride = 2, nonlinearity=leaky_rectify)

#Merge layer here merges Next one up from bottom merge layer
merge3 = lasagne.layers.ConcatLayer([Conv3b,UpConv2], 1, cropping = [None, None, 'center', 'center'])

Conv7a = lasagne.layers.Conv2DLayer(merge3, 4*base_filt, (3,3), nonlinearity=leaky_rectify, pad = 1)
Conv7b = lasagne.layers.Conv2DLayer(Conv7a, 2*base_filt, (3,3), nonlinearity=leaky_rectify, pad = 'full')

UpConv3 = lasagne.layers.TransposedConv2DLayer(Conv7b, 2*base_filt,filter_size = (2,2), stride = 2, nonlinearity=leaky_rectify)

#merge layer here merges next one up from merge3
merge2 = lasagne.layers.ConcatLayer([Conv2b,UpConv3], 1, cropping = [None, None, 'center', 'center'])
Conv8a = lasagne.layers.Conv2DLayer(merge2, 2*base_filt, (3,3), nonlinearity=leaky_rectify, pad = 1)
Conv8b = lasagne.layers.Conv2DLayer(Conv8a, base_filt, (3,3), nonlinearity=leaky_rectify, pad = 'full')

UpConv4 = lasagne.layers.TransposedConv2DLayer(Conv8b, base_filt, filter_size = (2,2),stride = 2, nonlinearity=leaky_rectify)

#merge layer here merges top of U
merge1 = lasagne.layers.ConcatLayer([Conv1b,UpConv4], 1, cropping = [None, None, 'center', 'center'])
Conv9a =lasagne.layers.Conv2DLayer(merge1, base_filt, (3,3), nonlinearity=leaky_rectify)
Conv9b = lasagne.layers.Conv2DLayer(Conv9a, base_filt, (3,3), nonlinearity=leaky_rectify, pad = 'full')

#Tweaking the output layer    
network  = lasagne.layers.Conv2DLayer(Conv9b, 2, (1,1), nonlinearity = leaky_rectify)

network_compressed = lasagne.layers.Conv2DLayer(network, 1, (1,1), nonlinearity = sigmoid)

network_sig = lasagne.layers.SliceLayer(network_compressed, indices = 0, axis = 1)

# create loss function
prediction = lasagne.layers.get_output(network_sig)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
  
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001,
                                            momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast = True)

# load previously obtained params

#with np.load('/Images/model_Unet3.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#
#lasagne.layers.set_all_param_values(network, param_values)

#Set up Training Data
train_folder = '/Images/train'
#train_folder = '/Users/Abecedarian/Desktop/Nerve_Segmentation/train'
dir_list = os.listdir(train_folder)
dir_list = [x for x in dir_list if not('mask' in x)]


#Train in Chunks
debug_fn = theano.function([input_var], prediction, allow_input_downcast = True)

train_num = 100

X_train = np.empty((train_num, 1, x_dim, y_dim))
y_train = np.empty((train_num, x_dim, y_dim))

with open('/Images/errors.csv','w') as csvfile:
    error = csv.writer(csvfile, delimiter=',')
    error.writerow(['Train Error'])    
    
    for passNumber in range(40):
        np.random.seed(passNumber)       
        shuffled = np.random.choice(range(len(dir_list)), train_num, replace = False)
          
        for i in range(train_num):        
            X_train[i,0] = misc.imresize(misc.imread(train_folder+os.sep+dir_list[shuffled[i]]),size = down_sample)
            y_train[i] = 1.0*(misc.imresize(misc.imread(train_folder + os.sep+dir_list[shuffled[i]].split('.')[0] + '_mask.tif'), size = down_sample) > 0)
               
#        print(debug_fn(X_train[0:1]).shape, y_train[0:1].shape)       
        # train network
        num_epochs = 1
        
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
        
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            
        ## use trained network for predictions
        error.writerow([train_err / train_batches])
        
        ## save updated parameters        
        np.savez('/Images/model_Unet3.npz', *lasagne.layers.get_all_param_values(network))

## Saving

border_percent = 0.5

predict_fn = theano.function([input_var], prediction, allow_input_downcast = True)

def RLE(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > border_percent)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])
        
with open('/Images/Submission_RLEs.csv','w') as csvfile:
    imagesub = csv.writer(csvfile, delimiter=',')
    imagesub.writerow(['img','Prediction'])

    start_time = time.time()    
    for i in range(test_num):
        imageNum = dir_list_test[i].split('.')[0]
        X_test[0,0] = misc.imresize(misc.imread(test_folder+os.sep+dir_list_test[i]), size = down_sample)        
                
        imagesub.writerow([imageNum, RLE(predict_fn(X_test[0:1])[0])])
        print(time.time() - start_time)
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:25:42 2016

@author: Abecedarian
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:48:23 2016

@author: Abecedarian
"""

import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import leaky_rectify, softmax
import scipy.misc as misc
import time
import numpy as np
import os
import sys
import csv
#import matplotlib.pyplot as plt

#Set up Test Data
test_folder = '/Images/test'
#test_folder = '/Users/tiruviluamala/Desktop/UltrasoundNerveSegmentation/test'
dir_list_test = os.listdir(test_folder)
dir_list_test = [x for x in dir_list_test if 'tif' in x]

test_num = len(dir_list_test)



X_test = np.empty((100, 1, 42,58))

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
target_var = T.ivector('y')

# create a small convolutional neural network

network = lasagne.layers.InputLayer((None, 1, 42, 58), input_var)

network = lasagne.layers.Conv2DLayer(network, 32, (5, 5),
                                     nonlinearity=leaky_rectify)

network = lasagne.layers.Pool2DLayer(network, (2, 2), mode='max')

network = lasagne.layers.Conv2DLayer(network, 64, (5, 5),
                                     nonlinearity=leaky_rectify)
                                     
#network = lasagne.layers.Pool2DLayer(network, (3, 3), mode='max')
                          
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    128, nonlinearity=leaky_rectify,
                                    W=lasagne.init.Orthogonal())
                                    
network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                    2, nonlinearity=softmax)

# create loss function
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
        network, lasagne.regularization.l2)

# create parameter update expressions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0001,
                                            momentum=0.9)

# compile training function that updates parameters and returns training loss
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast = True)

# load previously obtained params

with np.load('/Images/model_random_chunks_binaryBlocksTrimmed.npz') as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]

lasagne.layers.set_all_param_values(network, param_values)

#Set up Training Data
train_folder = '/Images/train'
#train_folder = '/Users/Abecedarian/Desktop/Nerve_Segmentation/train'
dir_list = os.listdir(train_folder)
dir_list = [x for x in dir_list if not('mask' in x)]


#Train in Chunks
train_num = 200

with open('/Images/errors.csv','w') as csvfile:
    error = csv.writer(csvfile, delimiter=',')
    error.writerow(['Train Error'])    
    
    for passNumber in range(1800):
        X_train = np.empty((train_num*100, 1, 42,58))
        y_train = np.empty(train_num*100)        
        
        np.random.seed(passNumber)       
        shuffled = np.random.choice(range(len(dir_list)), train_num, replace = False)
        
        for i in range(train_num): 
            for j in range(100):
                k = j % 10
                l = j//10
                X_train[100*i+j,0] = misc.imread(train_folder+os.sep+dir_list[shuffled[i]])[42*k:(42*k+42),58*l:(58*l+58)]
                y_train[100*i+j] = int(np.sum(misc.imread(train_folder + os.sep+dir_list[shuffled[i]].split('.')[0] + '_mask.tif')[42*k:(42*k+42),58*l:(58*l+58)])/(4.2*58*255))
                
                y_train[100*i+j] = int(y_train[100*i+j] > 0)

            if sum(y_train[100*i:100*(i+1)]) > 0:
                for j in range(100):
                    if y_train[100*i+j] == 0:
                        y_train[100*i+j] = 2
            else:
                indicesToRemove = np.random.choice(range(100*i,100*(i+1)), 94, replace = False)
                y_train[indicesToRemove] = 2

        X_train, y_train = X_train[y_train != 2], y_train[y_train != 2]
        print(X_train.shape[0],y_train.shape[0])
                                
                ###Testing this code
#                temp = np.zeros((420,580))                
#                for m in range(100):
#                    k = m % 10
#                    l = (m-k)/10
#                    temp[42*k:(42*k+42),58*l:(58*l+58)] = (y_train[i*100+m])*25.5
#                plt.imshow(temp)
#                plt.imshow(misc.imread(train_folder + os.sep+dir_list[shuffled[i]].split('.')[0] + '_mask.tif'))
#       
        # train network
        num_epochs = 1
        batch_size = 200
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
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        
        np.savez('/Images/model_random_chunks_binaryBlocksTrimmed.npz', *lasagne.layers.get_all_param_values(network))
        
        predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1),allow_input_downcast = True)
        
        ## Print out something useful
        
        Neel_Error = sum(abs(predict_fn(X_train[0:1000]) - y_train[0:1000]))
        print("Neel_Error for 10 images: %r" %Neel_Error)
        error.writerow([train_err / train_batches])

## Saving 
with open('/Images/Block_Profiles.csv','w') as csvfile:
    imagesub = csv.writer(csvfile, delimiter=',')
    imagesub.writerow(['img','Block_Num','Block_Profile'])
    for i in range(test_num):
        imageNum = dir_list_test[i].split('.')[0]
        for j in range(100):
            k = j % 10
            l = (j - k)/10
            X_test[j,0] = misc.imread(test_folder+os.sep+dir_list_test[i])[42*k:(42*k+42),58*l:(58*l+58)]        
            imagesub.writerow([imageNum, j, predict_fn(X_test[j:j+1])[0]])

##

######## Making a submission
#blockInfo = np.zeros((test_num*100,3))
#
##with open('/Users/Abecedarian/Desktop/Block_Profiles.csv', 'rb') as f:
#with open('/Images/Block_Profiles.csv', 'rb') as f:
#    reader = csv.reader(f)
#    l = 0    
#    for row in reader:
#        if l != 0:
#            blockInfo[l-1] = row
#        l = l+1
#
## Looking at Some Images

Block_Profiles = np.genfromtxt('/Users/tiruviluamala/Downloads/Block_Profiles.csv', delimiter=",")
i = 3
temp = np.zeros((420,580))                
for m in range(100):
    k = m % 10
    l = m//10
    temp[42*k:(42*k+42),58*l:(58*l+58)] = (blockInfo[i*100+m])*25.5
plt.imshow(temp)
#
#
#with open('/Images/Blocks_Submission.csv','w', newline='') as csvfile:
#    imagesub = csv.writer(csvfile)
#    imagesub.writerow(['img','pixels'])
#    for im in testImages:
#        imageNum = int(im.split(os.sep)[6].split('.')[0])        
#        image = misc.imread(im)
#        image = misc.imresize(image, (42,58))
#        
#        imagesub.writerow([imageNum,'1 243600'])

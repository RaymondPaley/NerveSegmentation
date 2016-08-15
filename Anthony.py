# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:18:55 2016

@author: Abecedarian
"""

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

#Set up Test Data
test_folder = '/Images/test'
#test_folder = '/Users/tiruviluamala/Desktop/UltrasoundNerveSegmentation/test'
dir_list_test = os.listdir(test_folder)
dir_list_test = [x for x in dir_list_test if 'tif' in x]

test_num = len(dir_list_test)



X_test = np.empty((1, 1, 42,58))

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

Inp = lasagne.layers.InputLayer((None, 1, 42, 58), input_var)


Conv1a= lasagne.layers.Conv2DLayer(Inp, 32, (3,3),nonlinearity=leaky_rectify)
Conv1b = lasagne.layers.Conv2DLayer(Conv1a, 32, (3,3),nonlinearity=leaky_rectify)

Pool1 = lasagne.layers.Pool2DLayer(Conv1b, (2, 2), mode='max')

Conv2a = lasagne.layers.Conv2DLayer(Pool1, 32, (3,3),nonlinearity=leaky_rectify)
Conv2b = lasagne.layers.Conv2DLayer(Pool1, 32, (3,3),nonlinearity=leaky_rectify)

Pool2 = lasagne.layers.Pool2DLayer(Conv2b, (2, 2), mode='max')

Conv3a = lasagne.layers.Conv2DLayer(Pool2, 32, (3,3),nonlinearity=leaky_rectify)
Conv3b = lasagne.layers.Conv2DLayer(Conv3a), 32, (3,3),nonlinearity=leaky_rectify)

Pool3 = lasagne.layers.Pool2DLayer(Conv3b, (2, 2), mode='max')

Conv4a = lasagne.layers.Conv2DLayer(Pool3, 32, (3,3),nonlinearity=leaky_rectify)
Conv4b = lasagne.layers.Conv2DLayer(Conv4a, 32, (3,3),nonlinearity=leaky_rectify)

Pool4 = lasagne.layers.Pool2DLayer(Conv4b, (2, 2), mode='max')

Conv5a = lasagne.layers.Conv2DLayer(Pool4, 32, (3,3),nonlinearity=leaky_rectify)
Conv5b = lasagne.layers.Conv2DLayer(Conv5a, 32, (3,3),nonlinearity=leaky_rectify)

UpConv1 = lasagne.layers.TransposedConv2DLayer(Conv5b,32,(2,2),nonlinearity=leaky_rectify)


#merge layer here merges bottom of U
merge4 = lasagne.layers.ConcatLayer((Conv4b,UpConv1,1,None))
Conv6a = lasagne.layers.Conv2DLayer(merge4,32, (3,3), nonlinearity=leaky_rectify)
Conv6b = lasagne.layers.Conv2DLayer(Conv6a, 32, (3,3), nonlinearity=leaky_rectify)

UpConv2 = lasagne.layers.TransposedConv2DLayer(Conv6b,32,(2,2),nonlinearity=leaky_rectify)

#Merge layer here merges Next one up from bottom merge layer
merge3 = lasagne.layers.ConcatLayer((Conv3b,UpConv2)1,None)
Conv7a = lasagne.layers.Conv2DLayer(merge3,32,(3,3),nonlinearity=leaky_rectify)
Conv7b = lasagne.layers.Conv2DLayer(Conv7a, 32, (3,3), nonlinearity=leaky_rectify)

UpConv3 = lasagne.layers.TransposedConv2DLayer(Conv7b,32,(2,2),nonlinearity=leaky_rectify)

#merge layer here merges next one up from merge3
merge2 = lasagne.layers.ConcatLayer((Conv2b,UpConv3),1, None)
Conv8a = lasagne.layers.Conv2DLayer(merge2,32,(3,3),nonlinearity=leaky_rectify)
Conv8b = lasagne.layers.Conv2DLayer(Conv8a, 32, (3,3), nonlinearity=leaky_rectify)

UpConv4 = lasagne.layers.TransposedConv2DLayer(Conv8b,32,(2,2),nonlinearity=leaky_rectify)

#merge layer here merges top of U
merge1 = lasagne.layers.ConcatLayer((Conv1b,UpConv4), 1, None)
Conv9a =lasagne.layers.Conv2DLayer(merge1,32,(3,3),nonlinearity=leaky_rectify)
Conv9b = lasagne.layers.Conv2DLayer(Conv9a, 32, (3,3), nonlinearity=leaky_rectify)

FIN  = lasagne.layers.Conv2DLayer(Conv9b,32, (1,1), nonlinearity = leaky_rectify)








# create loss function
prediction = lasagne.layers.get_output(FIN)
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
#
#with np.load('/Images/model_random_chunks_3.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#
#lasagne.layers.set_all_param_values(network, param_values)

#Set up Training Data
train_folder = '/Images/train'
#train_folder = '/Users/Abecedarian/Desktop/Nerve_Segmentation/train'
dir_list = os.listdir(train_folder)
dir_list = [x for x in dir_list if not('mask' in x)]


#Train in Chunks
train_num = 200

X_train = np.empty((train_num, 1, 420,580))
y_train = np.empty(train_num)

with open('/Images/errors.csv','w') as csvfile:
    error = csv.writer(csvfile, delimiter=',')
    error.writerow(['Train Error'])

    for passNumber in range(5):
        np.random.seed(passNumber)
        shuffled = np.random.choice(range(len(dir_list)), train_num, replace = False)

        for i in range(train_num):
            X_train[i,0] = misc.imread(train_folder+os.sep+dir_list[shuffled[i]])
            #reshape to a 420*580 x 1 vector
            X_train[i,0] = X_train[i,0].reshape(420*580, 1)
            #Binarize the image
            for j in range(420*580):
                if X_train[i,0][j] > 127.5:
                    X_train[i,0][j] = 255
                else
                    X_train[i,0][j] = 0



        y_train[i] = int(np.sum(misc.imread(train_folder + os.sep+dir_list[shuffled[i]].split('.')[0] + '_mask.tif')) > 0)


        # train network
        num_epochs = 1
        batch_size = 2
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

        np.savez('/Images/model_random_chunks_3.npz', *lasagne.layers.get_all_param_values(network))

        predict_fn_show = theano.function([input_var], T.argmax(test_prediction, axis=1),allow_input_downcast = True)
        predict_fn = theano.function([input_var], test_prediction[:,1],allow_input_downcast = True)

        print("Predicted class for first train input: %r" % predict_fn_show(X_train[0:10]))
        print(y_train[0:10])
        dif = predict_fn_show(X_train[0:train_num])-y_train[0:train_num]
        print(sum(dif != 0))
        error.writerow([train_err / train_batches])

        print "5"

## Saving
with open('/Images/Predicted_Classes.csv','w') as csvfile:
    imagesub = csv.writer(csvfile, delimiter=',')
    imagesub.writerow(['img','Prediction'])
    for i in range(test_num):
        imageNum = dir_list_test[i].split('.')[0]
        X_test[0,0] = misc.imread(test_folder+os.sep+dir_list_test[i])

        imagesub.writerow([imageNum,predict_fn(X_test[0:1])[0]])

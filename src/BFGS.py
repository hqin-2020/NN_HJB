import json
import tensorflow as tf 
import time 
import os
import pickle
from para import *
from training import *

workdir = os.path.dirname(os.getcwd())
model = 'ah_0135'
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + model + '/'
outputdir = workdir + '/output/' + model + '/'
docdir = workdir + '/doc/' + model + '/'

json_location =  datadir + 'parameters.json'

with open(json_location) as json_file:
  paramsFromFile= json.load(json_file)
params = setModelParametersFromFile(paramsFromFile)

batchSize = 2048

## Use float64 by default
tf.keras.backend.set_floatx("float64")

logXiE_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[3,]),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(1,  activation= None,  kernel_initializer='glorot_normal')])

logXiH_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[3,]),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh', kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(1, activation=None   , kernel_initializer='glorot_normal')])

kappa_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[3,]),
      tf.keras.layers.Dense(16, activation='tanh',    kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh',    kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh',    kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(16, activation='tanh',    kernel_initializer='glorot_normal'),
      tf.keras.layers.Dense(1,  activation='sigmoid', kernel_initializer='glorot_normal')])

start = time.time()
targets = tf.zeros(shape=(batchSize,1), dtype=tf.float64)
for iter in range(10):
  W = tf.random.uniform(shape = (batchSize,1), minval = params['wMin'], maxval = params['wMax'], dtype=tf.float64)
  Z = tf.random.uniform(shape = (batchSize,1), minval = params['zMin'], maxval = params['zMax'], dtype=tf.float64)
  V = tf.random.uniform(shape = (batchSize,1), minval = params['vMin'], maxval = params['vMax'], dtype=tf.float64)
  
  print('Iteration', iter)
  training_step_BFGS(logXiH_NN, logXiE_NN, kappa_NN, W, Z, V, params, targets)
end = time.time()
print('Elapsed time for training {:.4f} sec'.format(end - start))

## Save trained neural network approximations and respective model parameters
save_results = True
if save_results:

  VF_h_Name = 'logXiH_NN'
  VF_e_Name = 'logXiE_NN'
  policy_Name = 'kappa_NN'

  tf.saved_model.save(logXiH_NN, outputdir   + VF_h_Name)
  tf.saved_model.save(logXiE_NN, outputdir   + VF_e_Name)
  tf.saved_model.save(kappa_NN, outputdir + policy_Name)

  with open(outputdir + 'params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


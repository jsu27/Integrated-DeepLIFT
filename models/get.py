import keras
from keras.models import load_model
from keras.models import Model
print("keras version:", keras.__version__)
import tensorflow as tf
print("tensorflow version:", tf.__version__)
import wget
import os
import sys
from keras.datasets import cifar10
import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
x_valid = x_train[:10000]
y_valid = y_train[:10000]

model_files = [
    "model_cifar10_balanced_seed-0_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-10_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-20_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-30_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-40_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-50_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-60_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-70_bestbefore-100_currentepoch-100_valacc-91_vgg.h5",
    "model_cifar10_balanced_seed-80_bestbefore-100_currentepoch-100_valacc-90_vgg.h5",
    "model_cifar10_balanced_seed-90_bestbefore-100_currentepoch-100_valacc-91_vgg.h5"
]

for model_file in model_files:
  print("On model", model_file)
  if (os.path.isfile(model_file)==False):
    print("Downloading", model_file)
    wget.download("https://zenodo.org/record/2648107/files/"
                  +model_file+"?download=1", out=model_file)
  model = load_model(model_file)

  pre_softmax_model = Model(input=model.input,
                            output=model.layers[-2].output)
  print("Making predictions on validation set")
  valid_preacts = pre_softmax_model.predict(x_valid)
  print("Making predictions on test set")
  test_preacts = pre_softmax_model.predict(x_test)

  print("Test accuracy",np.mean(np.argmax(test_preacts,axis=1)
                                == np.squeeze(y_test)))
  print("Valid accuracy",np.mean(np.argmax(valid_preacts,axis=1)
                                 == np.squeeze(y_valid)))

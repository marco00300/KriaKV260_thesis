'''
Quantize the floating-point model
'''

import argparse
import os
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np
import pickle
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope


DIVIDER = '-----------------------------------------'

def input_fn(input_data,batchsize,is_training):
  '''
  Dataset creation and augmentation for training
  '''
  dataset = tf.data.Dataset.from_tensor_slices(input_data)
  if is_training:
    dataset = dataset.shuffle(buffer_size=1000,seed=42)
  dataset = dataset.batch(batchsize, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  if is_training:
      dataset = dataset.repeat()
  return dataset

def quant_model(float_model,quant_model,batchsize, x_pickle_file, y_pickle_file, evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''
    with open(x_pickle_file, 'rb') as f:
        X = pickle.load(f)
    
    with open(y_pickle_file, 'rb') as f:
        y = pickle.load(f)
    
    # make folder for saving quantized model
    head_tail = os.path.split(quant_model) 
    os.makedirs(head_tail[0], exist_ok = True)

    # load the floating point trained model
    float_model = load_model(float_model)
    # Print the model input shape
    print("Model input shape:", float_model.input_shape)
    float_model.summary()

    quant_dataset = input_fn((X, y), batchsize, False)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)


    if (evaluate):
        '''
        Evaluate quantized model
        '''
        print('\n'+DIVIDER)
        print ('Evaluating quantized model..')
        print(DIVIDER+'\n')

        test_dataset = input_fn((X, y), batchsize, False)

        quantized_model.compile(optimizer=Adam(),
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

        scores = quantized_model.evaluate(test_dataset,
                                          verbose=0)

        print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        print('\n'+DIVIDER)

    return



def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--float_model',  type=str, default='build/float_model/my_newnewmodel.h5', help='Full path of floating-point model. Default is build/float_model/k_model.h5')
    'path to save the float model'
    ap.add_argument('-q', '--quant_model',  type=str, default='build/quant_model/q_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    'path to save the quant model'
    ap.add_argument('-b', '--batchsize',    type=int, default=50,                       help='Batchsize for quantization. Default is 50')
    ap.add_argument('-x', '--x_pickle_file', type=str, default='build/dataset/704_calibration_1000_samples_X.pickle',
                    help='Full path to the pickle file containing the features. Default is build/dataset/calibration_data_X.pickle')
    ap.add_argument('-y', '--y_pickle_file', type=str, default='build/dataset/704_calibration_1000_samples_y.pickle',
                    help='Full path to the pickle file containing the labels. Default is build/dataset/calibration_data_y.pickle')
    ap.add_argument('-e', '--evaluate',     action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --float_model  : ', args.float_model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print(' --x_pickle_file: ', args.x_pickle_file)
    print(' --y_pickle_file: ', args.y_pickle_file)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')


    quant_model(args.float_model, args.quant_model, args.batchsize, args.x_pickle_file, args.y_pickle_file, args.evaluate)


if __name__ ==  "__main__":
    main()

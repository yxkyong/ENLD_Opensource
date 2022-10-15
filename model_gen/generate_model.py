import sys
sys.path.append("your path")
######################################base
import numpy as np
from keras.models import load_model
from base_class import dataset,metric_function,base_function,aug_function
from model_class import *
from enld_class import enld_function
import pandas as pd
from sklearn.model_selection import train_test_split
########################################model
from model_class import my_model, keras_model, my_model_keras_fix
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KDTree,KNeighborsClassifier,KernelDensity,KNeighborsTransformer
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import time
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#####################################train model
#model = load_model(model_path)
def save_and_train(model, X_train, Y_train, batch_size=64, save_path = './model_results/cifar_100/test.h5', epoches=50):
    model.fit(X_train, Y_train, batch_size, epochs=epoches)#validation_data=(x_val, y_val)
    model.save(save_path)
    return model
def save_and_train_aug(datagen, callbacks, model, X_train, Y_train, X_test, Y_test, batch_size=64, save_path = './model_results/cifar_100/test.h5', epoches=50):
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), validation_data=(X_test, Y_test),
                        epochs=epoches, verbose=2, workers=4, callbacks=callbacks,steps_per_epoch=X_train.shape[0] // batch_size)
    model.save(save_path)
    return model
def model_init(input_shape, n_classes, architecture='ResNet164'):
    optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    res_model = my_model.create_model(input_shape=input_shape, classes=n_classes, name=architecture, architecture=architecture)
    res_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
    weights_initial = res_model.get_weights()
    return res_model
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
def get_all_one_hot(label, class_num =100):
    if type(label)!=list:
        label = label.tolist()
    if type(label[0])==np.ndarray:
        label = [data.tolist() for data in label]
        label = sum(label,[])
    make_up = [i for i in range(class_num)]
    label = label + make_up
    label = pd.get_dummies(label)
    label_onehot = np.array(label)
    
    return label_onehot[:-class_num]





def train_cifar_100(data_path, noise_rate, save_path):
    ####################################################################################
    start = time.time()
    random_seed = 66
    base = data_path
    ###################################### step 0 ######################################
    ########################################################################## load data
    des = 'cifar_100_noisy{}_ratio2_1'.format(noise_rate)
    path = base+'/cifar100_source/'
    noisy_path = base + des +'/'
    x_train_inventory, y_train_inventory = np.load(path+'x_train_inventory_2.npy'), np.load(noisy_path+'noisy_y_train_inventory.npy')
    #### on hot code
    y_train_inventory = pd.get_dummies(y_train_inventory)
    y_train_inventory = np.array(y_train_inventory)

    ########################################################Generate X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(x_train_inventory, y_train_inventory, test_size=0.5, random_state=random_seed)
    np.save( base+ des+'/X_train55.npy', X_train)
    np.save( base+ des+'/X_test55.npy', X_test)
    np.save( base+ des+'/Y_train55.npy', Y_train)
    np.save( base+ des+'/Y_test55.npy', Y_test)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #--------------------------------------------------------------------------------------
    #########################################################Load X_train, X_test, Y_train, Y_test

    # X_train, X_test, Y_train, Y_test = np.load(base+ des+'/X_train55.npy'), np.load(base+ des+'/X_test55.npy'), \
    #                                     np.load( base+ des+'/Y_train55.npy'), np.load( base + des+'/Y_test55.npy')
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #--------------------------------------------------------------------------------------


    ###################################### step 0 ######################################

    #####################################Define optimizer and compile model
    input_shape = X_train[0].shape
    n_classes = Y_train.shape[1]
    ######################## 
    # 3.3M
    res_model = my_model.create_model(input_shape=input_shape, classes=n_classes, name='ResNet110', architecture='ResNet110')
    res_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    # res_model = keras_model.ENLD_Resnet50(input_shape=input_shape, classes=n_classes)
    res_model.summary()

    #####################################save and train 
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    T = np.load(base + des +'/'+ des +'_T.npy')
    acc = []

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    save_path = save_path
    if os.path.exists(save_path) == False:
            os.makedirs(save_path)
    save_callbacks = keras.callbacks.ModelCheckpoint(save_path+'AUG_R110_weights{epoch:03d}.h5', 
                                         save_weights_only=False, period=10)
    callbacks = [lr_reducer, lr_scheduler,save_callbacks]

    batch_size = 32
    epoches = 200
    print(X_train.shape, Y_train.shape)
    training_generator = aug_function.MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.2, datagen=datagen)()
    res_model.fit_generator(training_generator, validation_data=(X_test, Y_test),
                            epochs=epoches, verbose=2, callbacks=callbacks,steps_per_epoch=X_train.shape[0] // batch_size)


    end = time.time()
    print("time:",end-start)
    
    
def train_emnist(data_path, noise_rate, save_path):
    start = time.time()
    random_seed = 66
    base = data_path
    ###################################### step 0 ######################################
    ########################################################################## load data
    des = 'eminst_noisy{}_ratio2_1'.format(noise_rate)
    path = base+'eminst_source/'
    noisy_path = base + des +'/'
    x_train_inventory, y_train_inventory = np.load(path+'x_train_inventory_2.npy'), np.load(noisy_path+'noisy_y_train_inventory.npy')
    #### on hot code
    y_train_inventory = pd.get_dummies(y_train_inventory)
    y_train_inventory = np.array(y_train_inventory)

    #--------------------------------------------------------------------------------------
    #########################################################Load X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(x_train_inventory, y_train_inventory, test_size=0.5, random_state=random_seed)
    np.save( base+ des+'/X_train55.npy', X_train)
    np.save( base+ des+'/X_test55.npy', X_test)
    np.save( base+ des+'/Y_train55.npy', Y_train)
    np.save( base+ des+'/Y_test55.npy', Y_test)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    # X_train, X_test, Y_train, Y_test = np.load(base+'/process_result/'+ des+'/X_train55.npy'), np.load(base+'/process_result/'+ des+'/X_test55.npy'), \
    #                                     np.load( base+'/process_result/'+ des+'/Y_train55.npy'), np.load( base+'/process_result/'+ des+'/Y_test55.npy')
    # # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #--------------------------------------------------------------------------------------


    ###################################### step 0 ######################################

    #####################################Define optimizer and compile model
    input_shape = X_train[0].shape
    n_classes = Y_train.shape[1]
    ######################## 
    # res_model = my_model_keras_fix.create_model(input_shape=input_shape, classes=n_classes, name='ResNet164', architecture='ResNet164')
    # 3.3M
    res_model = mymode_eminst.create_model(input_shape=input_shape, classes=n_classes, name='ResNet110', architecture='ResNet110')
    res_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    # res_model = keras_model.ENLD_Resnet50(input_shape=input_shape, classes=n_classes)
    res_model.summary()

    #####################################save and train 
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    T = np.load(base + des +'/'+ des +'_T.npy')
    acc = []

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    save_path = save_path
    save_callbacks = keras.callbacks.ModelCheckpoint(save_path+'AUG_R110_weights{epoch:03d}.h5', 
                                         save_weights_only=False, period=5)
    callbacks = [lr_reducer, lr_scheduler,save_callbacks]

    batch_size = 32
    epoches = 30
    print(X_train.shape, Y_train.shape)
    training_generator = aug_function.MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.2, datagen=datagen)()
    res_model.fit_generator(training_generator, validation_data=(X_test, Y_test),
                            epochs=epoches, verbose=2, callbacks=callbacks,steps_per_epoch=X_train.shape[0] // batch_size)



    end = time.time()
    print("time:",end-start)
    
    
def train_tiny_imagenet(data_path, noise_rate, save_path):
    ####################################################################################
    start = time.time()
    random_seed = 66
    base = data_path
    ###################################### step 0 ######################################
    ########################################################################## load data
    des = 'tiny_imagenet_noisy{}_ratio2_1'.format(noise_rate)
    path = base+'/tiny_imagenet_source/'
    noisy_path = base + des +'/'
    x_train_inventory, y_train_inventory = np.load(path+'x_train_inventory_2.npy'), np.load(noisy_path+'noisy_y_train_inventory.npy')
    #### on hot code
    y_train_inventory = pd.get_dummies(y_train_inventory)
    y_train_inventory = np.array(y_train_inventory)

    ########################################################Generate X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(x_train_inventory, y_train_inventory, test_size=0.5, random_state=random_seed)
    np.save( base+ des+'/X_train55.npy', X_train)
    np.save( base+ des+'/X_test55.npy', X_test)
    np.save( base+ des+'/Y_train55.npy', Y_train)
    np.save( base+ des+'/Y_test55.npy', Y_test)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #--------------------------------------------------------------------------------------
    #########################################################Load X_train, X_test, Y_train, Y_test

    # X_train, X_test, Y_train, Y_test = np.load(base+ des+'/X_train55.npy'), np.load(base+ des+'/X_test55.npy'), \
    #                                     np.load( base+ des+'/Y_train55.npy'), np.load( base + des+'/Y_test55.npy')
    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
    #--------------------------------------------------------------------------------------


    ###################################### step 0 ######################################

    #####################################Define optimizer and compile model
    input_shape = X_train[0].shape
    n_classes = Y_train.shape[1]
    ######################## 
    # 3.3M
    res_model = my_model_imagenet.create_model(input_shape=input_shape, classes=n_classes, name='ResNet110', architecture='ResNet110')
    res_model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    # res_model = keras_model.ENLD_Resnet50(input_shape=input_shape, classes=n_classes)
    res_model.summary()

    #####################################save and train 
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    T = np.load(base + des +'/'+ des +'_T.npy')
    acc = []

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.0)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    save_path = save_path
    if os.path.exists(save_path) == False:
            os.makedirs(save_path)
    save_callbacks = keras.callbacks.ModelCheckpoint(save_path+'AUG_R110_weights{epoch:03d}.h5', 
                                         save_weights_only=False, period=10)
    callbacks = [lr_reducer, lr_scheduler,save_callbacks]

    batch_size = 32
    epoches = 100
    print(X_train.shape, Y_train.shape)
    training_generator = aug_function.MixupGenerator(X_train, Y_train, batch_size=batch_size, alpha=0.2, datagen=datagen)()
    res_model.fit_generator(training_generator, validation_data=(X_test, Y_test),
                            epochs=epoches, verbose=2, callbacks=callbacks,steps_per_epoch=X_train.shape[0] // batch_size)


    end = time.time()
    print("time:",end-start)


################################### data preprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar100")
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--noise_rate', type=str,default="0.1")
args = parser.parse_args()


if args.dataset=='emnist':
    train_emnist(data_path, noise_rate, save_path)
    
if args.dataset=='cifar100':
    train_cifar_100(args.data_path, args.noise_rate, args.save_path)
    
if args.dataset=='tiny_imagenet':
    train_tiny_imagenet(data_path, noise_rate, save_path)

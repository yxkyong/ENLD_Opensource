import sys
sys.path.append("/home/3005yxk/ENLD_project_8_31/ENLD_code/")
########################### enld base.py
######################################base
import numpy as np
from keras.models import load_model
from base_class import dataset,metric_function,base_function
from model_class import *
from enld_class import enld_function
import pandas as pd
from sklearn.model_selection import train_test_split
########################################model
from model_class import my_model, keras_model
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
from keras.layers import Input,Conv2D,Dense,BatchNormalization,Activation,add,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Model
from keras import backend as K
import heapq
import random
import pickle
import time
import math
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

################################test 
def get_high_quality_data(res_model, X_test, Y_test):
    pred_probs = res_model.predict(X_test)
    observe_labels = np.argmax(np.array(Y_test), axis=1)
    high_quality_index = enld_function.get_high_quality(observe_labels, pred_probs)
    high_quality_x = X_test[high_quality_index]
    high_quality_y = observe_labels[high_quality_index]
    high_quality_y_onehost = Y_test[high_quality_index]
    ########################get P and T
    pred_probs = res_model.predict(X_test)
    observe_labels = np.argmax(np.array(Y_test), axis=1)
    # print(pred_probs.shape, observe_labels.shape)
    J = enld_function.Joint_Estimate(observe_labels, pred_probs)
    J_T = enld_function.Joint_Estimate_T(observe_labels, pred_probs)
    
    P_observed_true = enld_function.normalized_T(J)
    P_true_observed = enld_function.normalized_T(J_T)
    
    
    return high_quality_x,high_quality_y,high_quality_index,high_quality_y_onehost, P_observed_true, P_true_observed

def get_high_quality_data_only(res_model, X_test, Y_test, layer_name):
    pred_probs, feature = enld_function.feature_extraction(res_model, X_test, layer_name = layer_name)
    observe_labels = np.argmax(np.array(Y_test), axis=1)
    high_quality_index = enld_function.get_high_quality(observe_labels, pred_probs)
    high_quality_x = X_test[high_quality_index]
    high_quality_y = observe_labels[high_quality_index]
    high_quality_y_onehost = Y_test[high_quality_index]
    
    return_feature = feature[high_quality_index]
    return high_quality_x,high_quality_y,high_quality_index,high_quality_y_onehost, return_feature

def get_high_quality_data_only_confidence(res_model, X_test, Y_test, layer_name, confidence=0):
    pred_probs, feature = enld_function.feature_extraction(res_model, X_test, layer_name = layer_name)
    observe_labels = np.argmax(np.array(Y_test), axis=1)
    high_quality_index = enld_function.get_high_quality_confidence(observe_labels, pred_probs,confidence)
    high_quality_x = X_test[high_quality_index]
    high_quality_y = observe_labels[high_quality_index]
    high_quality_y_onehost = Y_test[high_quality_index]
    
    return_feature = feature[high_quality_index]
    return high_quality_x,high_quality_y,high_quality_index,high_quality_y_onehost, return_feature

def get_high_quality_data_only_confidence_auto(res_model, X_test, Y_test, layer_name, confidence=0, auto=False):
    pred_probs, feature = enld_function.feature_extraction(res_model, X_test, layer_name = layer_name)
    observe_labels = np.argmax(np.array(Y_test), axis=1)
    high_quality_index = enld_function.get_high_quality_confidence_auto(observe_labels, pred_probs, auto=auto)
    high_quality_x = X_test[high_quality_index]
    high_quality_y = observe_labels[high_quality_index]
    high_quality_y_onehost = Y_test[high_quality_index]
    
    return_feature = feature[high_quality_index]
    return high_quality_x,high_quality_y,high_quality_index,high_quality_y_onehost, return_feature

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
def flatten(sample_x):
    list_s = []
    for data in sample_x:
        for e in data:
            list_s.append(e)
    return np.array(list_s)


def print_noisy(observed_label, true_label):
    groud_truth_index =  [i for i in range(len(observed_label)) if observed_label[i]!=true_label[i]]
    print(len(groud_truth_index)/len(observed_label))
    

def eval_enld_main_cifar100(noise_rate, args): # step =5
################################################ load data
    noisy_rate = noise_rate
    des = 'cifar_100_noisy{}_ratio2_1'.format(noisy_rate)
    base_dir = args.data_path
    X_train, X_test, Y_train, Y_test = np.load(base_dir + des+'/X_train55.npy'), np.load( base_dir + des+'/X_test55.npy'), \
                                        np.load( base_dir + des+'/Y_train55.npy'), np.load( base_dir + des+'/Y_test55.npy')  
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    input_shape = X_train[0].shape
    n_classes = Y_train.shape[1]



    layer_name = 'flatten'
    save_path = args.model_path
    res_model = load_model( save_path)
    
    ##############################################init
    high_quality_x,high_quality_y,high_quality_index,high_quality_y_onehost, P_observed_true, P_true_observed = get_high_quality_data(res_model, X_test, Y_test)
    pre_list = []
    recall_list = []
    time_list =[]
    his_all = []
    validate_list = []
    for k in range(1,21):
    ################################################ load incremental data
    # k = 1
        start = time.time()
        incremental_path = base_dir + 'divide_incremental_905/' + str(k)+'/'
        incremental_x, incremental_y = np.load(incremental_path + 'x_incremental.npy'), np.load(incremental_path + 'y_incremental.npy')
        #noisy rate
        noisy_y = np.load(incremental_path + 'y_incremental_{}.npy'.format(noisy_rate))
        noisy_y = noisy_y.astype('int')
        incremental_x = incremental_x.astype('float32') / 255
        incremental_y_onehot = get_all_one_hot(incremental_y, class_num=100)
        print(incremental_x.shape)
        print_noisy(noisy_y, incremental_y)
        ################################################ load model
        # incremental_x_total, incremental_y_total = np.load('./process_result_cifar100/cifar_100_source/x_train_incremental_1.npy'), np.load('./process_result_cifar100/cifar_100_source/y_train_incremental_1.npy')
        # incremental_x_total = incremental_x_total.astype('float32') / 255
        # incremental_y_total_onehot = get_all_one_hot(incremental_y_total, class_num=100)

        ############### multi round flush back base version

        
        def return_label_distribution(y):
            y_set = set(y)
            labe_count_dic  = {}
            for label in y:
                if label not in labe_count_dic:
                    labe_count_dic[label] = 0
                else: 
                    labe_count_dic[label] = labe_count_dic[label] + 1
            return labe_count_dic
        def count_clean_update(clean_count, index_list):
            for index in index_list:
                clean_count[index]  = clean_count[index] + 1
            return clean_count

        def update_clean_index(clean_index, clean_count, steps):
            new_clean_index = []
            for index in clean_index:
                if clean_count[index]>=steps:
                    new_clean_index.append(index)
            return new_clean_index
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
                validation_split=0.0
        )

        ############ 
        n_classes = Y_train.shape[1]
        iteration = args.iteration
        step = 5
        confidence = 0
        batch_size = args.batch_size_set
        size = args.size
        warm_step = 2
        vote = args.vote
        filter_para = 0.6
        name = './tmp/tmp_mix_size1_917_withoutclean.h5'
        # if noisy_rate in ['0.1', '0.2']:
        #     auto_para = False
        # if noisy_rate in ['0.3', '0.4']:
        #     auto_para = True
        auto_para = True
        ################################################################################ naive training
        res_model = load_model(save_path)
        ### lr = 0.001
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        res_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
        # high_quality_x high_quality_y_onehost sampled_x, sampled_y_one_hot
        clean_index = []
        noisy_index = []
        selected_index = []
        his = []
        validate = []
        ##################### obtain subset

        print("noisy label distribution:", return_label_distribution(noisy_y))
        print("true label distribution:", return_label_distribution(incremental_y))
        #######################################################

        noisy_y_set = set(noisy_y)
        print("noisy label size:", len(noisy_y_set),len(noisy_y))
        Y_test_label  = np.argmax(np.array(Y_test), axis=1)
        subset_noisy_index = [index for index in range(len(Y_test_label)) if Y_test_label[index] in noisy_y_set]
        sub_x = X_test[subset_noisy_index]
        sub_y = Y_test_label[subset_noisy_index]
        sub_y_onhot = Y_test[subset_noisy_index]
        print("sub size:", sub_x.shape)

        subset_noisy_index = np.array(subset_noisy_index)
        ###############################################################################################
        #####################################obtain am data
        predict_prob, incremental_feature = enld_function.feature_extraction(res_model, incremental_x, layer_name = layer_name)
        predict_label = np.argmax(np.array(predict_prob), axis=1)
        am_index = enld_function.get_am_data(noisy_y, predict_prob)
        am_x = incremental_x[am_index]
        am_y = noisy_y[am_index]
        ################################################################################ 
        # if am_x.shape[0]/incremental_x.shape[0]>=filter_para:
        #     auto_para = True
        # else: 
        #     auto_para = False


        ################################################################################ 
        pre_label = np.argmax(np.array(predict_prob), axis=1)
        am_pre = pre_label[am_index]
        incremental_am_feature = incremental_feature[am_index]  #enld_function.feature_extraction(res_model, am_x, layer_name = layer_name)[1]
        #################################################################################

        sub_high_x,sub_high_y,sub_high_index,sub_high_y_onehost, sub_inventory_feature= get_high_quality_data_only_confidence_auto(res_model, sub_x, sub_y_onhot, layer_name = layer_name, confidence=confidence, auto=auto_para)
        sub_high_y_onhot = get_all_one_hot(sub_high_y, class_num=n_classes)
        sub_high_quality_dic, sub_KD_tree_list = enld_function.generate_dic(sub_inventory_feature, sub_high_x, sub_high_y)
        # _,_,sub_high_index_fix,_, _= get_high_quality_data_only_confidence(res_model, sub_x, sub_y_onhot, layer_name = layer_name, confidence=0.9)
        # selected_index = list(set(selected_index)|set(subset_noisy_index[sub_high_index_fix]))

        ## init size 
        # incremental_am_feature = incremental_feature
        sampled_x, sampled_y = enld_function.contrastive_sampling_sub_fix(sub_KD_tree_list, sub_high_quality_dic, incremental_am_feature,am_y,P_observed_true, incremental_am_feature.shape[0]*size, noisy_y_set)
        sampled_x = flatten(sampled_x)
        sampled_y = flatten(sampled_y)
        print("sampled label distribution:", return_label_distribution(sampled_y))
        sampled_y_one_hot = get_all_one_hot(sampled_y, class_num=n_classes)
        print(sampled_x.shape,sampled_y_one_hot.shape)
        ###############################################################################################
        noisy_y_onehot = get_all_one_hot(noisy_y, class_num=n_classes)
        ###model init
        best = 0

        ######################################## best init
        for i in range(5): #sub_high_x sub_high_y_onhot


            history = res_model.fit_generator(datagen.flow(sampled_x, sampled_y_one_hot,batch_size=batch_size),validation_data=(incremental_x, noisy_y_onehot),epochs=2,
                                   steps_per_epoch=sampled_x.shape[0] // batch_size)
            print(history.history['val_accuracy'])
            if history.history['val_accuracy'][1]>best:
                best = history.history['val_accuracy'][1]
                res_model.save(name)
                ############################################################################################-----------------------------------------
                sub_high_x,sub_high_y,sub_high_index,sub_high_y_onehost, sub_inventory_feature= get_high_quality_data_only_confidence_auto(res_model, sub_x, sub_y_onhot, layer_name = layer_name, confidence=confidence, auto=auto_para)
                sub_high_quality_dic, sub_KD_tree_list = enld_function.generate_dic(sub_inventory_feature, sub_high_x, sub_high_y)
            # ########################## update sample


        #####################################obtain am data
        predict_prob, incremental_feature = enld_function.feature_extraction(res_model, incremental_x, layer_name = layer_name)
        predict_label = np.argmax(np.array(predict_prob), axis=1)
        am_index = enld_function.get_am_data(noisy_y, predict_prob)
        am_x = incremental_x[am_index]
        am_y = noisy_y[am_index]

        pre_label = np.argmax(np.array(predict_prob), axis=1)
        am_pre = pre_label[am_index]
        incremental_am_feature = incremental_feature[am_index]  #enld_function.feature_extraction(res_model, am_x, layer_name = layer_name)[1]
        ## init size 
        # incremental_am_feature = incremental_feature
        sampled_x, sampled_y = enld_function.contrastive_sampling_sub_fix(sub_KD_tree_list, sub_high_quality_dic, incremental_am_feature,am_y,P_observed_true, incremental_am_feature.shape[0]*size, noisy_y_set)
        sampled_x = flatten(sampled_x)
        sampled_y = flatten(sampled_y)
        print("sampled label distribution:", return_label_distribution(sampled_y))
        sampled_y_one_hot = get_all_one_hot(sampled_y, class_num=n_classes)
        print(sampled_x.shape,sampled_y_one_hot.shape)
        ###############################################################################################
        ############################################################################################-----------------------------------------

        # res_model.save(name)

        res_model = load_model(name)
        optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        res_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
        clean_count_select = np.zeros((X_test.shape[0],))

        for i in range(iteration):
            clean_count = np.zeros((incremental_x.shape[0],))

            for j in range(step):# validation_data=(incremental_x, incremental_y_onehot)
                history = res_model.fit_generator(datagen.flow(sampled_x, sampled_y_one_hot,batch_size=batch_size),epochs=2, #validation_data=(incremental_x, incremental_y_onehot),
                                           steps_per_epoch=sampled_x.shape[0] // batch_size)
                # validate.append(history.history['val_accuracy'])
                predict_prob, incremental_feature = enld_function.feature_extraction(res_model, incremental_x, layer_name = layer_name)#res_model.predict(incremental_x)
                predict_label = np.argmax(np.array(predict_prob), axis=1)

                clean_update =  [i for i in range(len(noisy_y)) if noisy_y[i]==predict_label[i]]# and predict_prob[i][predict_label[i]]>0.95
                ### clean count & update###############################################
                clean_count = count_clean_update(clean_count, clean_update)
                if i<warm_step:
                    clean_update = update_clean_index(clean_update, clean_count, steps=step)
                else:
                    clean_update = update_clean_index(clean_update, clean_count, steps=vote)
                ########################################################################
                clean_index = list(set(clean_index)|set(clean_update))
                print(len(clean_index))
                #####################################################
                #######################################################
                noisy_index = [i for i in range(len(noisy_y)) if i not in clean_index]
                print(len(noisy_index))
                ################## print precision & recall
                noisy_index = noisy_index
                groud_truth_index =  [i for i in range(len(noisy_y)) if noisy_y[i]!=incremental_y[i]]
                real = list(set(noisy_index) & set(groud_truth_index))
                print(len(real)/len(noisy_index))
                print(len(real)/len(groud_truth_index))
            # if i%5==0 and i>0:
            #     size = size + 1
            # ########################## update sample
            # confidence = confidence - (i+1)*(confidence/iteration)
            sub_high_x,sub_high_y,sub_high_index,sub_high_y_onehost, sub_inventory_feature= get_high_quality_data_only_confidence_auto(res_model, sub_x, sub_y_onhot, layer_name = layer_name, confidence=confidence, auto=auto_para)
            sub_high_quality_dic, sub_KD_tree_list = enld_function.generate_dic(sub_inventory_feature, sub_high_x, sub_high_y)
            #####################################obtain am data
            am_index = noisy_index #enld_function.get_am_data(noisy_y, predict_prob)
            am_x = incremental_x[am_index]
            am_y = noisy_y[am_index]
            pre_label = np.argmax(np.array(predict_prob), axis=1)
            am_pre = pre_label[am_index]
            incremental_am_feature = incremental_feature[am_index] #enld_function.feature_extraction(res_model, am_x, layer_name = layer_name)[1]
            sampled_x, sampled_y = enld_function.contrastive_sampling_sub_fix(sub_KD_tree_list, sub_high_quality_dic, incremental_am_feature,am_y,P_observed_true,am_x.shape[0]*size, noisy_y_set)
            sampled_x = flatten(sampled_x)
            sampled_y = flatten(sampled_y)
            sampled_y_one_hot = get_all_one_hot(sampled_y, class_num=n_classes)
            print(sampled_x.shape,sampled_y_one_hot.shape)
            # ###############################################################################################
            # _,_,sub_high_index_fix,_, _= get_high_quality_data_only_confidence(res_model, sub_x, sub_y_onhot, layer_name = layer_name, confidence=0.9)
            ####################################################################################################
            clean_select_update = subset_noisy_index[sub_high_index]
            clean_count_select = count_clean_update(clean_count_select, clean_select_update)
            clean_select_update = update_clean_index(clean_select_update, clean_count_select, steps=15)

            selected_index = list(set(selected_index)|set(clean_select_update))
            # precision(Y_test, y_inventory_test_gd, selected_index)
            ####################################################################################################
            # ##################### add clean sample
            # sampled_x = np.vstack([sampled_x, incremental_x[clean_index]])
            # sampled_y_one_hot = np.vstack([sampled_y_one_hot, noisy_y_onehot[clean_index]])
            print("selected_index num", len(selected_index))
            end_tmp = time.time()
            his.append([len(real)/len(noisy_index), len(real)/len(groud_truth_index), end_tmp-start])
        end = time.time()
        print("time:",end-start)
        print(validate)
        his_all.append(his)
        validate_list.append(validate)
        # return result
        print("file id:",k)
        print("noise rate:",noise_rate)
        time_list.append(end-start)
        pre_list.append(len(real)/len(noisy_index))
        recall_list.append(len(real)/len(groud_truth_index))
        selected_index = np.array(selected_index)
        # np.save('./selected_clean_816/'+des+'/'+str(k)+'.npy', selected_index)
    return pre_list, recall_list, time_list, his_all, validate_list


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vote', type=int, default=3)
parser.add_argument('--size', type=int, default=3)
parser.add_argument('--batch_size_set', type=int, default=64)
parser.add_argument('--iteration', type=int, default=17)
parser.add_argument('--noise_rate', type=str,default="0.1")
parser.add_argument('--data_path', type=str)
parser.add_argument('--model_path', type=str)

args = parser.parse_args()
print("parameters: vote, size, batch_size_set,iteration")
print(args.vote, args.size, args.batch_size_set,args.iteration)


############################################
pre_list, recall_list, time_list, his_all, validate_list = eval_enld_main_cifar100(noise_rate=args.noise_rate,args=args)
print("{}_result:".format(args.noise_rate))
print("precision:",pre_list)
print("recall:",recall_list)
print("time:",time_list)
print("history:", his_all)
print("validate_list:", validate_list)

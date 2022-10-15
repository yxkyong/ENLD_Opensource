
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import os.path
import pickle
import hashlib
import errno
import numpy as np
import pandas as pd
import sys
import random

from numpy.testing import assert_array_almost_equal
from six.moves import urllib

import tensorflow as tf

# def prepare_cifar10_data(data_dir, val_size=0, rescale=True):
#     # swap, orginally should be reshape to 3*32*32
#     train_data = []
#     train_labels = []
#     for i in range(5):
#         file = data_dir+'/data_batch_'+str(i+1)
#         with open(file, 'rb') as fo:
#             d = pickle.load(fo, encoding='bytes')
#         train_data.append(d[b'data'])
#         train_labels.append(d[b'labels'])
#     train_data = np.concatenate(train_data,axis=0)
#     train_labels = np.concatenate(train_labels,axis=0)

#     file = data_dir+'/test_batch'
#     with open(file, 'rb') as fo:
#         d = pickle.load(fo, encoding='bytes')
#     test_data = d[b'data']
#     test_labels = np.array(d[b'labels'])
    
#     validation_data = train_data[:val_size, :]
#     validation_labels = train_labels[:val_size]
#     train_data = train_data[val_size:, :]
#     train_labels = train_labels[val_size:]

#     # convert to one-hot labels
#     n_class=10
#     train_labels = np.eye(n_class)[train_labels]
#     validation_labels = np.eye(n_class)[validation_labels]
#     test_labels = np.eye(n_class)[test_labels]
    
#     if rescale:
#         train_data = train_data/255
#         validation_data = validation_data/255
#         test_data = test_data/255

#     train_data = train_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    
#     if val_size>0:
#         validation_data = validation_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    
#     test_data = test_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
       
#     return train_data, train_labels, validation_data, validation_labels, test_data, test_labels
 
def divide_data(data, labels, n_class=100, each_class=600, ratio=0.5):
    #ratio  the ratio of incremental data
    d_inc = []
    l_inc = []
    d_inv = []
    l_inv = []
    all_num=[]
    for i in range(each_class):
        all_num.append(i)
    for n in range(n_class):
        da=[]
        la=[]
        for i in range(len(data)):
            if labels[i] == n:
                da.append(data[i])
                la.append(labels[i])
        
        swich = random.sample(all_num, int(each_class*ratio))
        c = 0
        v = 0
        for i in range(len(la)):
            if i in swich:
                d_inc.append(da[i])
                l_inc.append(la[i])
                c = c + 1
            else:
                d_inv.append(da[i])
                l_inv.append(la[i])
                v = v+ 1
    d_inv = np.concatenate(d_inv)
    d_inv = d_inv.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    d_inc = np.concatenate(d_inc)
    d_inc = d_inc.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    return d_inv, l_inv, d_inc, l_inc
        
def get_incremental_data(data, labels,noisy_labels, num_class=10, ratio=0.01):
    y_train = pd.get_dummies(labels)
    y_train = np.array(y_train)
    n_class = y_train.shape[1]
    each_class = int(y_train.shape[0]/y_train.shape[1])
    all_class=[]
    dict_all = {}
    for i in range(n_class):
        num = 0
        for n in labels:
            if n == i:
                num = num + 1
        dict_all[i] = num
    all_num = int(ratio*(len(labels)))#要筛选的总数
    print(all_num)
    for i in range(n_class):
        all_class.append(i)
    
    dict1={}
    sum = 0
    select_data = []
    select_labels = []
    select_noisy_labels = []
    t = True
    while t:
        switch_class=random.sample(all_class,num_class)#随机出类序号
        for i in switch_class:
            dict1[i]=random.random()#计算每一类分别选取的个数，先随机一个数
            sum = sum + dict1[i]
        xx = 0
        al = 0
        for i in switch_class:
            dict1[i]=int(all_num*dict1[i]/sum)#计算每一类分别选取的个数，这里normalized所有随机的数，并乘总数
            al = al + dict1[i]
            if dict1[i] > dict_all[i]:
                xx = xx + 1
        if xx == 0 and al > int(0.95*all_num):
            t =False
            print(al,dict1)
    for i in switch_class:
        each = []
        for ii in range(dict_all[i]):
            each.append(ii)
        switch = random.sample(each,dict1[i])#每一类随机出选择序号
        this_num = 0
        nn = 0
        for x in range(len(data)):#从总量中抽取数据
            # print(labels[x])
            if labels[x] == i :
                this_num = this_num + 1 
                if this_num in switch:
                    nn = nn + 1
                    select_data.append(data[x])
                    select_labels.append(labels[x])
                    select_noisy_labels.append(noisy_labels[x])
        print(i,dict1[i],nn)
        print("this:",this_num,"all:",dict_all[i],switch)
    return select_data,  select_labels,  select_noisy_labels           
        
        
def get_incremental_data_2(data, labels, noisy_labels, num_class=10):
    y_train = pd.get_dummies(labels)
    y_train = np.array(y_train)
    n_class = y_train.shape[1]
    each_class = int(y_train.shape[0]/y_train.shape[1])
    all_class=[]
    dict_all = {}
    for i in range(n_class):
        num = 0
        for n in labels:
            if n == i:
                num = num + 1
        dict_all[i] = num

    for i in range(n_class):
        all_class.append(i)
    
    dict1={}
    sum = 0
    select_data = []
    select_labels = []
    select_noisy_labels = []

    
    switch_class=random.sample(all_class,num_class)#随机出类序号
    for i in switch_class:
        dict1[i]=random.random()#计算每一类分别选取的个数，先随机一个数
        sum = sum + dict1[i]
    xx = 0
    al = 0
    for i in switch_class:
        dict1[i]=int(dict_all[i] * dict1[i])#计算每一类分别选取的个数，这里normalized所有随机的数，并乘总数
        al = al + dict1[i]
    # print(al,dict1)
    for i in switch_class:
        each = []
        for ii in range(dict_all[i]):
            each.append(ii)
        switch = random.sample(each,dict1[i])#每一类随机出选择序号
        this_num = 0
        nn = 0
        for x in range(len(data)):#从总量中抽取数据
            # print(labels[x])
            if labels[x] == i :
                this_num = this_num + 1 
                if this_num in switch:
                    nn = nn + 1
                    select_data.append(data[x])
                    select_labels.append(labels[x])
                    select_noisy_labels.append(noisy_labels[x])
        # print(i,dict1[i],nn)
        # print("this:",this_num,"all:",dict_all[i],switch)
    return select_data,  select_labels,   select_noisy_labels          
               
        
        
        
def prepare_cifar100_data(data_dir,n_class=100,val_size=0, rescale=True):
    # swap, orginally should be reshape to 3*32*32
    train_data = []
    train_labels = []
    # for i in range(5):
    #     file = data_dir+'/data_batch_'+str(i+1)
    #     with open(file, 'rb') as fo:
    #         d = pickle.load(fo, encoding='bytes')
    #     train_data.append(d[b'data'])
    #     train_labels.append(d[b'labels'])
    with open(data_dir, 'rb') as f:
        if sys.version_info[0] == 2:
            entry = pickle.load(f)
        else:
            entry = pickle.load(f, encoding='latin1')
        train_data.append(entry['data'])
        if 'labels' in entry:
            train_labels.extend(entry['labels'])
        else:
            train_labels.extend(entry['fine_labels'])
    f.close()
    # train_data = np.concatenate(train_data,axis=0)
    # train_labels = np.concatenate(train_labels,axis=0)

    # file = data_dir+'/test_batch'*
    # with open(file, 'rb') as fo:
    #     d = pickle.load(fo, encoding='bytes')
    # test_data = d[b'data']
    # test_labels = np.array(d[b'labels'])
    
    # validation_data = train_data[:val_size, :]
    # validation_labels = train_labels[:val_size]
    # train_data = train_data[val_size:, :]
    # train_labels = train_labels[val_size:]

    # # convert to one-hot labels
    # # n_class=100
    # train_labels = np.eye(n_class)[train_labels]
    # validation_labels = np.eye(n_class)[validation_labels]
    # test_labels = np.eye(n_class)[test_labels]
    
    # if rescale:
    #     train_data = train_data/255
    #     validation_data = validation_data/255
    #     test_data = test_data/255
    train_data = np.concatenate(train_data)
    # # train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
    # # train_data = train_data.transpose((0, 2, 3, 1)) 
    # train_data = train_data.reshape((50000, 3, 32, 32))
	# train_data = train_data.transpose((0, 2, 3, 1)) 
    train_data = train_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])

    # # 选取部分类别数据
    # la=[]
    # da=[]
    # for i in range(0,len(train_labels)):
    #     if train_labels[i] <n_class:
    #         la.append(train_labels[i])
    #         da.append(train_data[i])
    # train_data = da
    # train_labels = la
    # max_x =0 
    # for i in range(0,len(train_labels)):
    #     if train_labels[i] > max_x :
    #         max_x = train_labels[i]
    # print("now the max labels is :",max_x)
    
    # #one hot
    # x_train_labels = np.eye(n_class)[train_labels]
    

    
    # if val_size>0:
    #     validation_data = validation_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    
    # test_data = test_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
       
    return train_data, train_labels






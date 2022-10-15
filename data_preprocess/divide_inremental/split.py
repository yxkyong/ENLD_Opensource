import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='cifar100')
parser.add_argument('--data_path',type=str)
parser.add_argument('--save_path',type=str)
args = parser.parse_args()




def divide_emnist(data_path,save_path):
    x=np.load(data_path+"/eminst/eminst_source/x_train_incremental_1.npy")
    y=np.load(data_path+"/eminst/eminst_source/y_train_incremental_1.npy")
    y1=np.load(data_path+"/eminst/eminst_noisy0.1_ratio2_1/noisy_y_train_incremental.npy")
    y2=np.load(data_path+"/eminst/eminst_noisy0.2_ratio2_1/noisy_y_train_incremental.npy")
    y3=np.load(data_path+"/eminst/eminst_noisy0.3_ratio2_1/noisy_y_train_incremental.npy")
    y4=np.load(data_path+"/eminst/eminst_noisy0.4_ratio2_1/noisy_y_train_incremental.npy")
# print(x[1])
    seed = 66
    label =  []
    d = {}
    for i in range(0,26):
        d[i] = 0
    for i in y:
        d[int(i)] = d[int(i)] + 1
    print(d)
    for i in range(0,26):
        label.append(i)
    for n in range(1,6):
        print("begin")
        print(x.shape)
        lx, ly, ly1, ly2, ly3, ly4 = [], [], [], [], [], []
        lxx, lyy, lyy1, lyy2, lyy3, lyy4 = [], [], [], [], [], []
        random.seed(seed)
        seed = seed + 1
        if n < 5:
            l_num = 5
        else:
            l_num = 6
        choose_label = random.sample(label,l_num)
        # print(1111111111111,choose_label)
        for t in range(0,l_num):
            random.seed(seed)
            seed = seed + 1
            ratio = random.random()
            
            this_label = choose_label[0]
            
            choose_label.remove(this_label)
            label.remove(this_label)
            
            ind = 0
            number = d[int(this_label)] * ratio
            for num in range(len(y)):
                if y[num] == this_label:
                    ind = ind + 1
                    if ind <= number:
                        lx.append(x[num])
                        ly.append(y[num])
                        ly1.append(y1[num])
                        ly2.append(y2[num])
                        ly3.append(y3[num])
                        ly4.append(y4[num])
                    else:
                        lxx.append(x[num])
                        lyy.append(y[num])
                        lyy1.append(y1[num])
                        lyy2.append(y2[num])
                        lyy3.append(y3[num])
                        lyy4.append(y4[num])
        print(len(lx),len(ly1),len(ly2),len(ly3),len(ly4))
        print(len(lxx),len(lyy1),len(lyy2),len(lyy3),len(lyy4))
        lx = np.array(lx)
        ly = np.array(ly)
        ly1 = np.array(ly1)
        ly2 = np.array(ly2)
        ly3 = np.array(ly3)
        ly4 = np.array(ly4)
        lxx = np.array(lxx)
        lyy = np.array(lyy)
        lyy1 = np.array(lyy1)
        lyy2 = np.array(lyy2)
        lyy3 = np.array(lyy3)
        lyy4 = np.array(lyy4)
        print(lx.shape)
        np.save(save_path+str(2*n-1)+"/x_incremental.npy",lx)
        np.save(save_path+str(2*n-1)+"/y_incremental.npy",ly)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.1.npy",ly1)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.2.npy",ly2)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.3.npy",ly3)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.4.npy",ly4)
    
        np.save(save_path+str(2*n)+"/x_incremental.npy",lxx)
        np.save(save_path+str(2*n)+"/y_incremental.npy",lyy)
        np.save(save_path+str(2*n)+"/y_incremental_0.1.npy",lyy1)
        np.save(save_path+str(2*n)+"/y_incremental_0.2.npy",lyy2)
        np.save(save_path+str(2*n)+"/y_incremental_0.3.npy",lyy3)
        np.save(save_path++str(2*n)+"/y_incremental_0.4.npy",lyy4)

def divide_cifar100(data_path,save_path):
    x=np.load(data_path+"/cifar100/cifar100_source/x_train_incremental_1.npy")
    y=np.load(data_path+data_path+"/cifar100/cifar100_source/y_train_incremental_1.npy")
    y1=np.load(data_path+"/cifar100/cifar_100_noisy0.1_ratio2_1/noisy_y_train_incremental.npy")
    y2=np.load(data_path+"/cifar100/cifar_100_noisy0.2_ratio2_1/noisy_y_train_incremental.npy")
    y3=np.load(data_path+"/cifar100/cifar_100_noisy0.3_ratio2_1/noisy_y_train_incremental.npy")
    y4=np.load(data_path+"/cifar100/cifar_100_noisy0.4_ratio2_1/noisy_y_train_incremental.npy")
    # print(x[1])
    seed = 66
    label =  []
    d = {}
    for i in range(0,100):
        d[i] = 0
    for i in y:
        d[int(i)] = d[int(i)] + 1
    print(d)
    for i in range(0,100):
        label.append(i)
    for n in range(1,11):
        print("begin")
        print(x.shape)
        lx, ly, ly1, ly2, ly3, ly4 = [], [], [], [], [], []
        lxx, lyy, lyy1, lyy2, lyy3, lyy4 = [], [], [], [], [], []
        random.seed(seed)
        seed = seed + 1
        l_num = 10
    # if n < 5:
    #     l_num = 5
    # else:
    #     l_num = 6
        choose_label = random.sample(label,l_num)
    # print(1111111111111,choose_label)
        for t in range(0,l_num):
            random.seed(seed)
            seed = seed + 1
            ratio = random.random()
            
            this_label = choose_label[0]
            
            choose_label.remove(this_label)
            label.remove(this_label)
            
            ind = 0
            number = d[int(this_label)] * ratio
            for num in range(len(y)):
                if y[num] == this_label:
                    ind = ind + 1
                    if ind <= number:
                        lx.append(x[num])
                        ly.append(y[num])
                        ly1.append(y1[num])
                        ly2.append(y2[num])
                        ly3.append(y3[num])
                        ly4.append(y4[num])
                    else:
                        lxx.append(x[num])
                        lyy.append(y[num])
                        lyy1.append(y1[num])
                        lyy2.append(y2[num])
                        lyy3.append(y3[num])
                        lyy4.append(y4[num])
        print(len(lx),len(ly1),len(ly2),len(ly3),len(ly4))
        print(len(lxx),len(lyy1),len(lyy2),len(lyy3),len(lyy4))
        lx = np.array(lx)
        ly = np.array(ly)
        ly1 = np.array(ly1)
        ly2 = np.array(ly2)
        ly3 = np.array(ly3)
        ly4 = np.array(ly4)
        lxx = np.array(lxx)
        lyy = np.array(lyy)
        lyy1 = np.array(lyy1)
        lyy2 = np.array(lyy2)
        lyy3 = np.array(lyy3)
        lyy4 = np.array(lyy4)
        print(lx.shape)
        np.save(save_path+str(2*n-1)+"/x_incremental.npy",lx)
        np.save(save_path+str(2*n-1)+"/y_incremental.npy",ly)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.1.npy",ly1)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.2.npy",ly2)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.3.npy",ly3)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.4.npy",ly4)
    
        np.save(save_path+str(2*n)+"/x_incremental.npy",lxx)
        np.save(save_path+str(2*n)+"/y_incremental.npy",lyy)
        np.save(save_path+str(2*n)+"/y_incremental_0.1.npy",lyy1)
        np.save(save_path+str(2*n)+"/y_incremental_0.2.npy",lyy2)
        np.save(save_path+str(2*n)+"/y_incremental_0.3.npy",lyy3)
        np.save(save_path+str(2*n)+"/y_incremental_0.4.npy",lyy4)

        
        
def divide_tiny_imagenet(data_path,save_path):
    x=np.load(data_path+"/tiny_imagenet/tiny_imagenet_source/x_train_incremental_1.npy")
    y=np.load(data_path+"/tiny_imagenet/tiny_imagenet_source/y_train_incremental_1.npy")
    y1=np.load(data_path+"/tiny_imagenet/tiny_imagenet_noisy0.1_ratio2_1/noisy_y_train_incremental.npy")
    y2=np.load(data_path+"/tiny_imagenet/tiny_imagenet_noisy0.2_ratio2_1/noisy_y_train_incremental.npy")
    y3=np.load(data_path+"/tiny_imagenet/tiny_imagenet_noisy0.3_ratio2_1/noisy_y_train_incremental.npy")
    y4=np.load(data_path+"/tiny_imagenet/tiny_imagenet_noisy0.4_ratio2_1/noisy_y_train_incremental.npy")
    # print(x[1])
    seed = 66
    label =  []
    d = {}
    for i in range(0,200):
        d[i] = 0
    for i in y:
        d[int(i)] = d[int(i)] + 1
    print(d)
    for i in range(0,200):
        label.append(i)
    for n in range(1,11):
        print("begin")
        print(x.shape)
        lx, ly, ly1, ly2, ly3, ly4 = [], [], [], [], [], []
        lxx, lyy, lyy1, lyy2, lyy3, lyy4 = [], [], [], [], [], []
        random.seed(seed)
        seed = seed + 1
        l_num = 20
    # if n < 5:
    #     l_num = 5
    # else:
    #     l_num = 6
        choose_label = random.sample(label,l_num)
    # print(1111111111111,choose_label)
        for t in range(0,l_num):
            random.seed(seed)
            seed = seed + 1
            ratio = random.random()
        
            this_label = choose_label[0]
        
            choose_label.remove(this_label)
            label.remove(this_label)
        
            ind = 0
            number = d[int(this_label)] * ratio
            for num in range(len(y)):
                if y[num] == this_label:
                    ind = ind + 1
                    if ind <= number:
                        lx.append(x[num])
                        ly.append(y[num])
                        ly1.append(y1[num])
                        ly2.append(y2[num])
                        ly3.append(y3[num])
                        ly4.append(y4[num])
                    else:
                        lxx.append(x[num])
                        lyy.append(y[num])
                        lyy1.append(y1[num])
                        lyy2.append(y2[num])
                        lyy3.append(y3[num])
                        lyy4.append(y4[num])
        print(len(lx),len(ly1),len(ly2),len(ly3),len(ly4))
        print(len(lxx),len(lyy1),len(lyy2),len(lyy3),len(lyy4))
        lx = np.array(lx)
        ly = np.array(ly)
        ly1 = np.array(ly1)
        ly2 = np.array(ly2)
        ly3 = np.array(ly3)
        ly4 = np.array(ly4)
        lxx = np.array(lxx)
        lyy = np.array(lyy)
        lyy1 = np.array(lyy1)
        lyy2 = np.array(lyy2)
        lyy3 = np.array(lyy3)
        lyy4 = np.array(lyy4)
        print(lx.shape)
        np.save(save_path+str(2*n-1)+"/x_incremental.npy",lx)
        np.save(save_path+str(2*n-1)+"/y_incremental.npy",ly)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.1.npy",ly1)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.2.npy",ly2)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.3.npy",ly3)
        np.save(save_path+str(2*n-1)+"/y_incremental_0.4.npy",ly4)
    
        np.save(save_path+str(2*n)+"/x_incremental.npy",lxx)
        np.save(save_path+str(2*n)+"/y_incremental.npy",lyy)
        np.save(save_path+str(2*n)+"/y_incremental_0.1.npy",lyy1)
        np.save(save_path+str(2*n)+"/y_incremental_0.2.npy",lyy2)
        np.save(save_path+str(2*n)+"/y_incremental_0.3.npy",lyy3)
        np.save(save_path+str(2*n)+"/y_incremental_0.4.npy",lyy4)
    
    
    
if args.dataset == 'emnist':
    divide_emnist(args.data_path,args.save_path)
    
if args.dataset == 'cifar100':
    divide_cifar100(args.data_path,args.save_path)
    
if args.dataset == 'tiny-imagenet':
    divide_tiny_imagenet(args.data_path,args.save_path)

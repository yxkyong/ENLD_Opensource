import sys
sys.path.append("ur path")


#### Cifar100 Preprocess
import numpy as np
from base_class import base_function, metric_function
from base_class import dataset
from sklearn.model_selection import train_test_split
from enld_class import *
import os


def emnist(data_path, save_path):
    #### Cifar100 Preprocess
    import numpy as np
    from base_class import base_function, metric_function
    from base_class import dataset
    from sklearn.model_selection import train_test_split
    from enld_class import *
    import os

    # Get Data
    dataset_path = data_path
    source_save_path =  save_path
    random_seed = 66

    x_total = np.load(dataset_path + 'train_data.npy')
    y_total = np.load(dataset_path + 'train_labels.npy')


    ####################################################################################################################
    # Divide Inventory Data and Incremental Data
    if os.path.exists(source_save_path) == False:
            os.makedirs(source_save_path)
    x_train_inventory, x_train_incremental, y_train_inventory, y_train_incremental = train_test_split(x_total, y_total, test_size = 1.0/3, random_state= random_seed)
    # print(x_train_inventory.shape, x_train_incremental.shape)
    np.save(source_save_path+'x_train_inventory_2.npy', x_train_inventory)
    np.save(source_save_path+'y_train_inventory_2.npy', y_train_inventory)
    np.save(source_save_path+'x_train_incremental_1.npy', x_train_incremental)
    np.save(source_save_path+'y_train_incremental_1.npy', y_train_incremental)

    print(x_train_inventory.shape, x_train_incremental.shape)
    ####################################################################################################################


    noisy_list = [0.1, 0.2, 0.3, 0.4]
    for noisy_rate in noisy_list:
        random_seed = 66
        n_classes = 26
        noisy_path = source_save_path +'/eminst_noisy{}_ratio2_1/'.format(str(noisy_rate))
        if os.path.exists(noisy_path) == False:
                os.makedirs(noisy_path)
        x_train_inventory, y_train_inventory, x_train_incremental, y_train_incremental = np.load(source_save_path+'x_train_inventory_2.npy'), np.load(source_save_path+'y_train_inventory_2.npy'),\
                                                                                    np.load(source_save_path+'x_train_incremental_1.npy'), np.load(source_save_path+'y_train_incremental_1.npy')
        T = base_function.generate_asymmetri_transition_matrix(t=noisy_rate, n_class=n_classes, co_class=1, random_seed=random_seed)
        np.save( noisy_path + "eminst_noisy{}_ratio2_1_T.npy".format(str(noisy_rate)), T)

        noisy_y_train_inventory  = base_function.generate_transimited_data(y_train_inventory, T)
        noisy_y_train_incremental  = base_function.generate_transimited_data(y_train_incremental, T)
        np.save(noisy_path+'noisy_y_train_inventory.npy', noisy_y_train_inventory)
        np.save(noisy_path+'noisy_y_train_incremental.npy', noisy_y_train_incremental)
        print("accuracy between noisy label and true label:")
        print(metric_function.enld_accuracy(y_train_inventory, noisy_y_train_inventory))



def cifar100(data_path, save_path):
    # Get Data
    dataset_path = data_path
    source_save_path =  save_pathv
    random_seed = 66

    x_train, y_train = dataset.prepare_cifar100_data(data_dir=dataset_path + 'cifar-100-python/train',n_class=100)
    x_test, y_test = dataset.prepare_cifar100_data(data_dir=dataset_path + 'cifar-100-python/test',n_class=100)
    x_total = np.vstack((x_train,x_test))
    y_total = np.hstack((y_train, y_test))


    ####################################################################################################################
    # Divide Inventory Data and Incremental Data
    if os.path.exists(source_save_path) == False:
            os.makedirs(source_save_path)
    x_train_inventory, x_train_incremental, y_train_inventory, y_train_incremental = train_test_split(x_total, y_total, test_size = 1.0/3, random_state= random_seed)
    # print(x_train_inventory.shape, x_train_incremental.shape)
    np.save(source_save_path+'x_train_inventory_2.npy', x_train_inventory)
    np.save(source_save_path+'y_train_inventory_2.npy', y_train_inventory)
    np.save(source_save_path+'x_train_incremental_1.npy', x_train_incremental)
    np.save(source_save_path+'y_train_incremental_1.npy', y_train_incremental)

    print(x_train_inventory.shape, x_train_incremental.shape)


    noisy_list = [0.1, 0.2, 0.3, 0.4]
    for noisy_rate in noisy_list:
        random_seed = 66
        n_classes = 100
        noisy_path = save_path +'/cifar_100_noisy{}_ratio2_1/'.format(str(noisy_rate))
        if os.path.exists(noisy_path) == False:
                os.makedirs(noisy_path)
        x_train_inventory, y_train_inventory, x_train_incremental, y_train_incremental = np.load(source_save_path+'x_train_inventory_2.npy'), np.load(source_save_path+'y_train_inventory_2.npy'),\
                                                                                    np.load(source_save_path+'x_train_incremental_1.npy'), np.load(source_save_path+'y_train_incremental_1.npy')
        T = base_function.generate_asymmetri_transition_matrix(t=noisy_rate, n_class=n_classes, co_class=1, random_seed=random_seed)
        np.save( noisy_path + "cifar_100_noisy{}_ratio2_1_T.npy".format(str(noisy_rate)), T)

        noisy_y_train_inventory  = base_function.generate_transimited_data(y_train_inventory, T)
        noisy_y_train_incremental  = base_function.generate_transimited_data(y_train_incremental, T)
        np.save(noisy_path+'noisy_y_train_inventory.npy', noisy_y_train_inventory)
        np.save(noisy_path+'noisy_y_train_incremental.npy', noisy_y_train_incremental)
        print("accuracy between noisy label and true label:")
        print(metric_function.enld_accuracy(y_train_inventory, noisy_y_train_inventory))
        
        
        
def tiny_imagenet(data_path, save_path):
    import numpy as np
    from base_class import base_function, metric_function
    from base_class import dataset
    from sklearn.model_selection import train_test_split
    from enld_class import *
    import os

    # Get Data
    dataset_path = data_path
    source_save_path =  save_path
    random_seed = 66

    x_total = np.load(dataset_path)
    y_total = np.load(dataset_path)



    ####################################################################################################################
    # Divide Inventory Data and Incremental Data
    if os.path.exists(source_save_path) == False:
            os.makedirs(source_save_path)
    x_train_inventory, x_train_incremental, y_train_inventory, y_train_incremental = train_test_split(x_total, y_total, test_size = 1.0/3, random_state= random_seed)
    # print(x_train_inventory.shape, x_train_incremental.shape)
    np.save(source_save_path+'x_train_inventory_2.npy', x_train_inventory)
    np.save(source_save_path+'y_train_inventory_2.npy', y_train_inventory)
    np.save(source_save_path+'x_train_incremental_1.npy', x_train_incremental)
    np.save(source_save_path+'y_train_incremental_1.npy', y_train_incremental)

    print(x_train_inventory.shape, x_train_incremental.shape)

    noisy_list = [0.1, 0.2, 0.3, 0.4]
    for noisy_rate in noisy_list:
        random_seed = 66
        n_classes = 200
        noisy_path = save_path + '/tiny_imagenet_noisy{}_ratio2_1/'.format(str(noisy_rate))
        if os.path.exists(noisy_path) == False:
                os.makedirs(noisy_path)
        x_train_inventory, y_train_inventory, x_train_incremental, y_train_incremental = np.load(source_save_path+'x_train_inventory_2.npy'), np.load(source_save_path+'y_train_inventory_2.npy'),\
                                                                                    np.load(source_save_path+'x_train_incremental_1.npy'), np.load(source_save_path+'y_train_incremental_1.npy')
        T = base_function.generate_asymmetri_transition_matrix(t=noisy_rate, n_class=n_classes, co_class=1, random_seed=random_seed)
        np.save( noisy_path + "tiny_imagenet_noisy{}_ratio2_1_T.npy".format(str(noisy_rate)), T)

        noisy_y_train_inventory  = base_function.generate_transimited_data(y_train_inventory, T)
        noisy_y_train_incremental  = base_function.generate_transimited_data(y_train_incremental, T)
        np.save(noisy_path+'noisy_y_train_inventory.npy', noisy_y_train_inventory)
        np.save(noisy_path+'noisy_y_train_incremental.npy', noisy_y_train_incremental)
        print("accuracy between noisy label and true label:")
        print(metric_function.enld_accuracy(y_train_inventory, noisy_y_train_inventory))


        
        
################################### data preprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cifar100")
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()


if args.dataset=='emnist':
    emnist(args.data_path, args.save_path)
    
if args.dataset=='cifar100':
    cifar100(args.data_path, args.save_path)
    
if args.dataset=='tiny_imagenet':
    tiny_imagenet(args.data_path, args.save_path)

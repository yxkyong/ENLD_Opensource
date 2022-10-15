import numpy as np
from keras.models import Model
from sklearn.neighbors import KDTree,KNeighborsClassifier,KernelDensity,KNeighborsTransformer
import random
import math

def norm(count_dic, num_dic):
    for key in count_dic:
        count_dic[key] = count_dic[key]/num_dic[key]
    return count_dic

############################################################ estimate transition matrix 
# INTUITION version, probability adding
def estimate_transition_matrix(observed_label, softmax_output):
    count_dic={}
    num_dic={}
    for i in range(len(observed_label)):
        if observed_label[i] not in count_dic:
            count_dic[observed_label[i]] = softmax_output[i]
            num_dic[observed_label[i]] = 1
        else:
            count_dic[observed_label[i]] = count_dic[observed_label[i]] + softmax_output[i]
            num_dic[observed_label[i]] = num_dic[observed_label[i]] + 1
            
    print(count_dic, num_dic)
    
    norm_dic  = norm(count_dic, num_dic)
    print(norm_dic)
    set_label = set(observed_label)
    T = np.zeros((len(set_label), len(set_label)))
    
    for key in norm_dic:
        for i in range(len(norm_dic[key])):
            T[i,key] = norm_dic[key][i]
            
    return T
#### 
# First Version
# observed label-true label
def Joint_Estimate(labels, pred_probs):
    set_label = set(labels)
    if len(set_label) > 100:
        J = np.zeros((len(set_label), len(set_label)))
    else:
        J = np.zeros((100, 100))
    for i in range(len(pred_probs)):
        index = np.where(pred_probs[i]==np.max(pred_probs[i]))
        J[int(labels[i]),index] = J[int(labels[i]),index] + 1
    return J

#true_label - observed_label 
def Joint_Estimate_T(labels, pred_probs):
    set_label = set(labels)  
    if len(set_label) > 100:
        J = np.zeros((len(set_label), len(set_label)))
    else:
        J = np.zeros((100, 100))
    for i in range(len(pred_probs)):
        index = np.where(pred_probs[i]==np.max(pred_probs[i]))
        J[index, int(labels[i])] = J[index, int(labels[i])] + 1
    return J

def normalized_T(J):
    for i in range(len(J)):
        norm_n = sum(J[i])
        if norm_n!=0:
            for j in range(len(J)):
                J[i,j] = J[i,j]/norm_n
            
    return J
#--------------------------------------------------------------------------------------

############################################################ obtain ambigious data
############################################################ obtain high quality data
def get_am_data(observed_label, pred_probs):
    index = []
    for i in range(len(observed_label)):
        predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
        if observed_label[i] != predict_label:
            index.append(i)
            
    return index

def get_high_quality(observed_label, pred_probs):
    index = []
    for i in range(len(observed_label)):
        predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
        if observed_label[i] == predict_label:
            index.append(i)
            
    return index

def get_high_quality_confidence(observed_label, pred_probs, confidence):
    index = []
    for i in range(len(observed_label)):
        predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
        if observed_label[i] == predict_label and pred_probs[i][predict_label]>confidence:
            index.append(i)
            
    return index

def get_high_quality_confidence_auto(observed_label, pred_probs,auto=False):
    if auto == True:
        index = []
        con_dic = {}
        label_set = [i for i in range(200)]
        for l in label_set:
            con_dic[l] = []
        # print(con_dic)
        for i in range(len(observed_label)):
            predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
            con_dic[predict_label[0][0]].append(pred_probs[i][predict_label])
        ################ calculate average
        confidence_dic = {}
        for l in label_set:
            if len(con_dic[l])==0:
                confidence_dic[l] = 0
            else:
                confidence_dic[l] = sum(con_dic[l])/len(con_dic[l])
        for i in range(len(observed_label)):
            predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
            if observed_label[i] == predict_label and pred_probs[i][predict_label]>=confidence_dic[predict_label[0][0]]:
                    index.append(i)
    else:
        confidence = 0
        index = []
        for i in range(len(observed_label)):
            predict_label =   np.where(pred_probs[i]==np.max(pred_probs[i]))
            if observed_label[i] == predict_label and pred_probs[i][predict_label]>=confidence:
                index.append(i)
                
    return index
#--------------------------------------------------------------------------------------
############################################################ generate_dic
# generate high quality data dic
def generate_dic(feature, x, y):
    label_set = set(y)
    dic = {}
    for label in label_set:
        dic[label] = [[],[],[]]
    for i in range(len(feature)):
        dic[y[i]][0].append(x[i])
        dic[y[i]][1].append(y[i])
        dic[y[i]][2].append(feature[i])
        
    KD_tree_list = {}
    for label in label_set:
        dic[label][0] = np.array(dic[label][0])
        dic[label][1] = np.array(dic[label][1])
        KD_tree_list[label] = KDTree(dic[label][2])
        
    count = {}
    for label in label_set:
        count[label] = len(dic[label][1])
    print(count)
        
    return dic, KD_tree_list

############################################################ contrastive sampling
# return contrastive samples set C
def contrastive_sampling(KD_tree_dic, high_dic, am_data, am_data_label, P, sample_num):
    index_set = []
    n = math.ceil(sample_num/len(am_data))
    ##### obtain data
    sampled_x = []
    sampled_y = []
    for i in range(len(am_data_label)):
            ##### get label id
            label = random_label(P, am_data_label[i])
            # print(am_data_label[i])
            # print(label)
            # get data
            dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            # index array([[46, 90]])
            sampled_x.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y.append(high_dic[label][1][idx_of_knn[0]])
    return sampled_x, sampled_y
##################################################################### while ture to sample
def contrastive_sampling_sub(KD_tree_dic, high_dic, am_data, am_data_label, P, sample_num, nosiy_set):
    index_set = []
    n = math.ceil(sample_num/len(am_data))
    ##### obtain data
    sampled_x = []
    sampled_y = []
    all_index = {}
    ###############################################################debug
    high_label_set = []
    for label in high_dic:
        high_label_set.append(label)
     #####################################################################
    for label in nosiy_set:
        all_index[label] = []
    #####################################################################
    for i in range(len(am_data_label)):
            ##### get label id
            while True:
                label = random_label(P, am_data_label[i])
                if label in nosiy_set and label in high_label_set:
                    break
            num = len(high_dic[label][0])
            if n<=num:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            else:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=num)
            # print(n,len(KD_tree_dic[label]))
            # index array([[46, 90]])
            sampled_x.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y.append(high_dic[label][1][idx_of_knn[0]])
            all_index[label] = list(set(all_index[label])|set(idx_of_knn[0]))
    count = 0
    for label in all_index:
        all_index[label] = len(all_index[label])
        count = count + all_index[label]
    print("sampled:",count)
    return sampled_x, sampled_y
#####################################################################
def contrastive_sampling_sub_fix(KD_tree_dic, high_dic, am_data, am_data_label, P, sample_num, nosiy_set):
    index_set = []
    n = math.ceil(sample_num/len(am_data))
    ##### obtain data
    sampled_x = []
    sampled_y = []
    all_index = {}
    ###############################################################debug
    high_label_set = []
    for label in high_dic:
        high_label_set.append(label)
     #####################################################################
    for label in nosiy_set:
        all_index[label] = []
    #####################################################################
    for i in range(len(am_data_label)):
            ##### get label id
            while True:
                label = random_label_fix(P, am_data_label[i], nosiy_set)
                if label in high_label_set:
                    break
            num = len(high_dic[label][0])
            if n<=num:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            else:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=num)
            # print(n,len(KD_tree_dic[label]))
            # index array([[46, 90]])
            sampled_x.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y.append(high_dic[label][1][idx_of_knn[0]])
            all_index[label] = list(set(all_index[label])|set(idx_of_knn[0]))
    count = 0
    for label in all_index:
        all_index[label] = len(all_index[label])
        count = count + all_index[label]
    print("sampled:",count)
    return sampled_x, sampled_y

def random_label_fix(P, label_i, noisy_set):
    noisy_set_list = list(noisy_set)
    temp_list = np.array(list(P[label_i,:]))
    # normalize
    temp = temp_list[noisy_set_list]
    total = sum(temp)
    for i in range(len(temp)):
        temp[i] = temp[i]/total
    random_index = random.random()
    for j in range(len(P[label_i,:])):
        res = random_index - sum(temp[0:j+1])
        if res <=0:
            temp_noisy_label = j
            break
    return_temp_index = noisy_set_list[temp_noisy_label]
    return return_temp_index

def contrastive_sampling_sub_onlypositive(KD_tree_dic, high_dic, am_data, am_data_label, P, sample_num, nosiy_set):
    index_set = []
    n = math.ceil(sample_num/len(am_data))
    ##### obtain data
    sampled_x = []
    sampled_y = []
    all_index = {}
    ###############################################################debug
    high_label_set = []
    for label in high_dic:
        high_label_set.append(label)
     #####################################################################
    for label in nosiy_set:
        all_index[label] = []
    #####################################################################
    for i in range(len(am_data_label)):
            ##### get label id
            # while True:
            #     label = random_label(P, am_data_label[i])
            #     if label in nosiy_set and label in high_label_set:
            #         break
            label = am_data_label[i]
            num = len(high_dic[label][0])
            if n<=num:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            else:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=num)
            # print(n,len(KD_tree_dic[label]))
            # index array([[46, 90]])
            sampled_x.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y.append(high_dic[label][1][idx_of_knn[0]])
            all_index[label] = list(set(all_index[label])|set(idx_of_knn[0]))
    count = 0
    for label in all_index:
        all_index[label] = len(all_index[label])
        count = count + all_index[label]
    print("sampled:",count)
    return sampled_x, sampled_y


def contrastive_sampling_sub_balance(KD_tree_dic, high_dic, am_data, am_data_label, P, sample_num_positive, sample_num_negative, nosiy_set):
    #################################################################################
    n = math.ceil(sample_num_positive/len(am_data))
    ##### obtain data
    sampled_x_positive = []
    sampled_y_positive = []
    all_index = {}
    ###############################################################debug
    high_label_set = []
    for label in high_dic:
        high_label_set.append(label)
     #####################################################################
    for label in nosiy_set:
        all_index[label] = []
    #####################################################################
    for i in range(len(am_data_label)):
            label = am_data_label[i]

            num = len(high_dic[label][0])
            if n<=num:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            else:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=num)
            # print(n,len(KD_tree_dic[label]))
            # index array([[46, 90]])
            sampled_x_positive.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y_positive.append(high_dic[label][1][idx_of_knn[0]])
            all_index[label] = list(set(all_index[label])|set(idx_of_knn[0]))
    sampled_x_positive = flatten(sampled_x_positive)
    sampled_y_positive = flatten(sampled_y_positive)
    
    ########################################################################
    n = math.ceil(sample_num_negative/len(am_data))
    ##### obtain data
    sampled_x_negative = []
    sampled_y_negative = []
    all_index = {}
    ###############################################################debug
    high_label_set = []
    for label in high_dic:
        high_label_set.append(label)
     #####################################################################
    for label in nosiy_set:
        all_index[label] = []
    #####################################################################
    for i in range(len(am_data_label)):
        
            while True:
                label = random_negative_only(P, am_data_label[i])
                if label in nosiy_set and label in high_label_set:
                    break

            num = len(high_dic[label][0])
            if n<=num:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=n)
            else:
                dist_to_knn, idx_of_knn = KD_tree_dic[label].query(am_data[[i]], k=num)
            # print(n,len(KD_tree_dic[label]))
            # index array([[46, 90]])
            sampled_x_negative.append(high_dic[label][0][idx_of_knn[0]])
            sampled_y_negative.append(high_dic[label][1][idx_of_knn[0]])
            all_index[label] = list(set(all_index[label])|set(idx_of_knn[0]))
    sampled_x_negative = flatten(sampled_x_negative)
    sampled_y_negative = flatten(sampled_y_negative)
    
    sampled_x = np.vstack([sampled_x_positive, sampled_x_negative])
    sampled_y = np.hstack([sampled_y_positive, sampled_y_negative])
    return sampled_x, sampled_y
def flatten(sample_x):
    list_s = []
    for data in sample_x:
        for e in data:
            list_s.append(e)
    return np.array(list_s)
################################################################
def random_label(P, label_i):
    temp_noisy_label = 0
    temp = P[label_i,:]
    random_index = random.random()
    for j in range(len(P[label_i,:])):
        res = random_index - sum(temp[0:j+1])
        if res <=0:
            temp_noisy_label = j
            break
    return temp_noisy_label

def random_negative_only(P, label_i):
    temp_noisy_label = 0
    temp = list(P[label_i,:])
    ###### positve 0 and normalize
    temp[label_i] = 0
    total = sum(temp)
    for i in range(len(temp)):
        temp[i] = temp[i]/total
    ##########################################################
    random_index = random.random()
    for j in range(len(temp)):
        res = random_index - sum(temp[0:j+1])
        if res <=0:
            temp_noisy_label = j
            break
    return temp_noisy_label

# rank method
# return fine grained model
    
def contrastive_fine_tune(source_model, contrastive_samples):
    pass

#--------------------------------------------------------------------------------------


############################################################ model output

def feature_extraction(model, X, layer_name = 'flatten_1'):
    new_model = Model(inputs=model.inputs, outputs=[model.output, model.get_layer(layer_name).output])
    output = new_model.predict(X)
    return output

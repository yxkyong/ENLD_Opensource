import numpy as np
import random

#################### 
def generate_asymmetri_transition_matrix(t=0.0, n_class=10, co_class=1, random_seed = 66):
    T = np.zeros((n_class, n_class))
    for i in range(n_class):
        T[i,i] = 1-t
        index_list = [l for l in range(n_class)]
        index_list.pop(i)
        random.seed(random_seed)
        class_index = random.sample(index_list,co_class)
        
        ratio = generate_k_ratio(co_class, total_ratio = t, random_seed=random_seed)
        for n in range(len(class_index)):
            T[i, class_index[n]] =  ratio[n]
            
        random_seed = random_seed + 1
            
    return T
        
        
def generate_k_ratio(k=3, total_ratio= 1,random_seed=66):
    # p = []
    # for i in range(k):
    #     random_number = random.random()
    #     p.append(random_number)
    np.random.seed(random_seed)
    p = np.random.rand(k,).tolist()
    total = sum(p)
    return [total_ratio*e/total for e in p]
        


#input: data_id, label, T
#output: data_id, noisy_label
def generate_transimited_data(label, T):
    noisy_label = np.zeros((len(label)))
    for i in range(len(label)):
        random.seed(i)
        random_index = random.random()
        for j in range(len(T[label[i],:])):
            temp = T[label[i],:]
            res = random_index - sum(temp[0:j+1])
            if res <=0:
                temp_noisy_label = j
                break

                
        noisy_label[i] = int(temp_noisy_label)
    return noisy_label



#######################base
##return second large index
def return_index(a):
    max2 = np.sort(a)[-2]
    for i in range(len(a)):
        if a[i]==max2:
            return i
        

def return_co_label(T):
    co_index = []
    for i in range(len(T)):
        co_index.append(return_index(T[i]))
    return co_index 
# cal if in the top 2
def if_top(T, J):
    a = []
    for i in range(len(T)):
        index = return_index(T[i])
        if J[i][index] >=  np.sort(J[i])[-2]:
            a.append(1)
        else:
            a.append(0)
            
    return sum(a)/len(a)



